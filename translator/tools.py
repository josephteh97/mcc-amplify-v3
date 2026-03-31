"""
translator/tools.py — Private toolset for the BIM-Translator Agent
====================================================================
Tools:
  revit_schema_mapper   — converts validated geometry → Revit Transaction JSON
  coordinate_transformer — pixel → real-world mm using grid-derived scale
  revit_api_client      — communicates with the Windows Revit Add-in; captures C# errors
  memory_io             — read/write for translator/memory.sqlite
                          (stores "API Success Patterns")

All tools return plain dicts.  No tool raises — failures surface as
{"ok": False, "error": "..."}.

Revit Transaction JSON schema (consumed by ModelBuilder.cs on Windows):
  { levels, grids, walls, columns, doors, windows, floors, ceilings, metadata }
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_DIR     = Path(__file__).parent
_DB_PATH = _DIR / "memory.sqlite"

# ── DB connection with idempotent schema init ─────────────────────────────────

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS api_success_patterns (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT NOT NULL,
        element_type    TEXT NOT NULL,
        family_name     TEXT,
        type_name       TEXT,
        parameters_json TEXT,
        revit_outcome   TEXT NOT NULL,
        error_message   TEXT,
        correction_applied TEXT,
        success_count   INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS translation_runs (
        run_id          TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        job_id          TEXT,
        status          TEXT,
        elements_placed INTEGER DEFAULT 0,
        revit_warnings  TEXT,
        error_message   TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_asp_type ON api_success_patterns(element_type);
    CREATE INDEX IF NOT EXISTS idx_asp_fam  ON api_success_patterns(family_name);
    CREATE INDEX IF NOT EXISTS idx_tr_ts    ON translation_runs(timestamp);
    """)
    return con


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — coordinate_transformer
# ══════════════════════════════════════════════════════════════════════════════

def coordinate_transformer(
    validated_geometry: dict,
    project_context: dict | None = None,
) -> dict:
    """
    Convert pixel-space column/wall coordinates to real-world mm using the
    structural grid as the coordinate reference.

    Strategy:
      1. Map grid labels to cumulative mm positions using DfMA standard bay widths
         or detected spacings from the grid result.
      2. For each element, snap its pixel centre to the nearest grid intersection
         and replace with mm coordinates.
      3. If pixel-to-mm ratio cannot be derived (no grid spacings), fall back to
         a safe default of 1 px = 1 mm with a warning.

    Args:
        validated_geometry: Output of ValidationAgent (geometry key).
        project_context:    May contain "bay_widths_x_mm" and "bay_widths_y_mm"
                            lists — explicit grid bay dimensions from the BIM brief.

    Returns:
        {
          "ok":           bool,
          "world_geometry": dict,   — geometry with mm coordinates
          "scale_info":   {         — derived scale metadata
            "px_per_mm":  float,
            "origin_px":  [x, y],
            "source":     "grid_derived" | "context_override" | "fallback_1:1",
          },
          "warnings": [str],
        }
    """
    ctx       = project_context or {}
    warnings: list[str] = []
    geometry  = _deep_copy(validated_geometry)
    grid      = geometry.get("grid", {})

    # ── Derive px/mm scale from grid ──────────────────────────────────────────
    # Grid spacings may come from project_context (if provided by controller)
    bay_x = ctx.get("bay_widths_x_mm", [])
    bay_y = ctx.get("bay_widths_y_mm", [])
    img_w = geometry.get("metadata", {}).get("image_w")
    img_h = geometry.get("metadata", {}).get("image_h")

    # Build cumulative mm positions for each grid line
    v_labels = grid.get("vertical_labels", [])
    h_labels = grid.get("horizontal_labels", [])
    n_v = len(v_labels)
    n_h = len(h_labels)

    # Use bay widths if provided; else distribute image pixels equally
    if bay_x and len(bay_x) >= n_v - 1:
        x_mm = [sum(bay_x[:i]) for i in range(n_v)] if n_v else []
        total_x_mm = sum(bay_x)
        px_per_mm_x = (img_w / total_x_mm) if (img_w and total_x_mm) else 1.0
        source = "context_override"
    elif img_w and n_v >= 2:
        # Equal spacing fallback
        spacing_px = img_w / (n_v - 1) if n_v > 1 else img_w
        # Assume a typical structural bay of 7500 mm
        assumed_bay_mm = ctx.get("assumed_bay_mm", 7500)
        px_per_mm_x    = spacing_px / assumed_bay_mm
        x_mm           = [i * assumed_bay_mm for i in range(n_v)]
        total_x_mm     = (n_v - 1) * assumed_bay_mm
        warnings.append(
            f"No bay widths provided. Assumed {assumed_bay_mm} mm per bay for "
            f"{n_v} vertical grid lines. Coordinates are approximate."
        )
        source = "fallback_assumed_bay"
    else:
        px_per_mm_x = 1.0
        x_mm        = list(range(n_v))
        total_x_mm  = n_v
        warnings.append("Cannot derive X scale — using 1 px = 1 mm fallback.")
        source = "fallback_1:1"

    if bay_y and len(bay_y) >= n_h - 1:
        y_mm = [sum(bay_y[:i]) for i in range(n_h)] if n_h else []
        total_y_mm  = sum(bay_y)
        px_per_mm_y = (img_h / total_y_mm) if (img_h and total_y_mm) else 1.0
    elif img_h and n_h >= 2:
        spacing_px  = img_h / (n_h - 1) if n_h > 1 else img_h
        assumed_bay = ctx.get("assumed_bay_mm", 7500)
        px_per_mm_y = spacing_px / assumed_bay
        y_mm        = [i * assumed_bay for i in range(n_h)]
        total_y_mm  = (n_h - 1) * assumed_bay
        if source == "fallback_assumed_bay":
            pass  # already warned above
        else:
            warnings.append(
                f"No Y bay widths provided. Assumed {assumed_bay} mm per bay."
            )
    else:
        px_per_mm_y = 1.0
        y_mm        = list(range(n_h))
        total_y_mm  = n_h

    # Average the two axes for a single px_per_mm value
    px_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0 if px_per_mm_x and px_per_mm_y else 1.0

    def px_to_mm(px_x: float, px_y: float) -> tuple[float, float]:
        """Convert raw pixel coordinates to world mm."""
        return round(px_x / px_per_mm, 1), round(px_y / px_per_mm, 1)

    # ── Transform columns ─────────────────────────────────────────────────────
    for col in geometry.get("columns", []):
        centre = col.get("center") or [None, None]
        if centre[0] is not None:
            mx, my = px_to_mm(float(centre[0]), float(centre[1]))
            col["location_mm"] = {"x": mx, "y": my, "z": 0.0}

    # ── Transform walls ───────────────────────────────────────────────────────
    for wall in geometry.get("walls", []):
        for pt_key in ("start_point", "end_point"):
            pt = wall.get(pt_key) or {}
            px_x = pt.get("px_x") or pt.get("x")
            px_y = pt.get("px_y") or pt.get("y")
            if px_x is not None:
                mx, my = px_to_mm(float(px_x), float(px_y))
                pt["x"], pt["y"], pt["z"] = mx, my, 0.0

    # ── Build grid world positions ────────────────────────────────────────────
    geometry["grid_world"] = {
        "vertical":   [
            {"label": v_labels[i], "x_mm": x_mm[i]} for i in range(len(x_mm))
        ],
        "horizontal": [
            {"label": h_labels[i], "y_mm": y_mm[i]} for i in range(len(y_mm))
        ],
        "total_x_mm": total_x_mm,
        "total_y_mm": total_y_mm if n_h > 0 else 0,
    }

    scale_info = {
        "px_per_mm":  round(px_per_mm, 4),
        "px_per_mm_x": round(px_per_mm_x, 4),
        "px_per_mm_y": round(px_per_mm_y, 4),
        "origin_px":   [0, 0],
        "source":      source,
    }

    return {
        "ok":           True,
        "world_geometry": geometry,
        "scale_info":   scale_info,
        "warnings":     warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — revit_schema_mapper
# ══════════════════════════════════════════════════════════════════════════════

# v1 family mappings (extracted from mcc-amplify-ai/backend/core/family_mapping.json)
_FAMILY_MAP = {
    "doors": {
        "single":  {"family": "M_Single-Flush",
                    "sizes":  {"800": "0800 x 2100mm", "900": "0900 x 2100mm", "1000": "1000 x 2100mm"}},
        "double":  {"family": "M_Double-Flush",
                    "sizes":  {"1800": "1800 x 2100mm"}},
    },
    "windows": {
        "fixed":   {"family": "M_Fixed",
                    "sizes":  {"1200x1500": "1200 x 1500mm"}},
    },
    "walls": {
        "Interior": "Generic - 200mm",
        "Exterior": "Generic - 300mm",
        "Shear":    "Generic - 250mm",
    },
    "columns": {
        "rectangular": "M_Concrete-Rectangular-Column",
        "circular":    "M_Concrete-Round-Column",
        "column":      "M_Concrete-Rectangular-Column",
    },
}

# DfMA dimension defaults (mm)
_DEFAULTS = {
    "wall_height":       2800,
    "wall_thickness":    200,
    "door_height":       2100,
    "window_height":     1500,
    "sill_height":       900,
    "floor_thickness":   200,
    "storey_height":     3000,
    "column_min_section": 200,
}


def revit_schema_mapper(
    world_geometry: dict,
    project_context: dict | None = None,
    scale_info: dict | None = None,
) -> dict:
    """
    Map world-coordinate validated geometry to a Revit Transaction JSON.

    The output dict is consumed directly by ModelBuilder.cs on the Windows
    Revit Add-in server (POST /build-model or file-drop mode).

    Args:
        world_geometry:  Geometry with mm coordinates (output of coordinate_transformer).
        project_context: Project overrides (storey height, default thicknesses).
        scale_info:      Scale metadata (for audit trail only).

    Returns:
        {
          "ok":                bool,
          "transaction_json":  dict,  — Revit Transaction ready for the Add-in
          "element_counts":    dict,  — { levels, grids, walls, columns, ... }
          "unmapped_warnings": [str], — families or types that could not be resolved
        }
    """
    ctx = {**_DEFAULTS, **(project_context or {})}
    unmapped: list[str] = []
    geom     = world_geometry

    storey_h = ctx.get("storey_height", _DEFAULTS["storey_height"])

    # ── Levels ────────────────────────────────────────────────────────────────
    levels = [
        {"name": "Level 0", "elevation": 0},
        {"name": "Level 1", "elevation": storey_h},
    ]

    # ── Grid lines ────────────────────────────────────────────────────────────
    gw    = geom.get("grid_world", {})
    grids = []
    ext   = 10000  # extend grid lines 10 m beyond boundary

    total_x = gw.get("total_x_mm", 0)
    total_y = gw.get("total_y_mm", 0)

    for vg in gw.get("vertical", []):
        x = vg["x_mm"]
        grids.append({
            "name":  vg["label"],
            "start": {"x": x, "y": -ext,         "z": 0.0},
            "end":   {"x": x, "y": total_y + ext, "z": 0.0},
        })
    for hg in gw.get("horizontal", []):
        y = hg["y_mm"]
        grids.append({
            "name":  hg["label"],
            "start": {"x": -ext,         "y": y, "z": 0.0},
            "end":   {"x": total_x + ext, "y": y, "z": 0.0},
        })

    # ── Columns ───────────────────────────────────────────────────────────────
    columns = []
    for col in geom.get("columns", []):
        loc   = col.get("location_mm", {})
        shape = col.get("shape", "rectangular")
        fam   = _FAMILY_MAP["columns"].get(shape, _FAMILY_MAP["columns"]["rectangular"])

        if col.get("is_circular") and col.get("diameter_mm"):
            w = d = float(col["diameter_mm"])
        else:
            w = float(col.get("width_mm") or ctx["column_min_section"])
            d = float(col.get("depth_mm") or ctx["column_min_section"])
        w = max(w, ctx["column_min_section"])
        d = max(d, ctx["column_min_section"])

        columns.append({
            "id":        col.get("id"),
            "type_mark": col.get("type_mark"),
            "family":    fam,
            "shape":     shape,
            "location":  {"x": loc.get("x", 0.0), "y": loc.get("y", 0.0), "z": 0.0},
            "width":     round(w, 1),
            "depth":     round(d, 1),
            "height":    ctx["wall_height"],
            "material":  col.get("material", "Concrete"),
            "level":     "Level 0",
            "top_level": "Level 1",
        })

    # ── Walls ─────────────────────────────────────────────────────────────────
    walls = []
    for wall in geom.get("walls", []):
        func     = wall.get("function", "Interior")
        type_str = _FAMILY_MAP["walls"].get(func, _FAMILY_MAP["walls"]["Interior"])
        thick    = float(wall.get("thickness_mm", ctx["wall_thickness"]))
        s        = wall.get("start_point", {})
        e        = wall.get("end_point",   {})
        walls.append({
            "id":            wall.get("id"),
            "type_name":     type_str,
            "start_point":   {"x": s.get("x", 0), "y": s.get("y", 0), "z": 0.0},
            "end_point":     {"x": e.get("x", 0), "y": e.get("y", 0), "z": 0.0},
            "thickness":     round(thick, 1),
            "height":        wall.get("height", ctx["wall_height"]),
            "material":      wall.get("material", "Concrete"),
            "is_structural": wall.get("is_structural", False),
            "function":      func,
            "level":         "Level 0",
        })

    # ── Doors ─────────────────────────────────────────────────────────────────
    doors = []
    for door in geom.get("doors", []):
        w_px  = door.get("width", 900)
        fam   = _FAMILY_MAP["doors"]["single"]["family"]
        doors.append({
            "id":             door.get("id"),
            "family":         fam,
            "type_name":      "0900 x 2100mm",
            "location":       door.get("location", {"x": 0, "y": 0, "z": 0}),
            "width":          round(float(w_px), 1),
            "height":         ctx["door_height"],
            "swing_direction": door.get("swing_direction", "Right"),
            "host_wall_id":   door.get("host_wall_id"),
            "level":          "Level 0",
        })

    # ── Windows ───────────────────────────────────────────────────────────────
    windows = []
    for win in geom.get("windows", []):
        windows.append({
            "id":        win.get("id"),
            "family":    _FAMILY_MAP["windows"]["fixed"]["family"],
            "type_name": "1200 x 1500mm",
            "location":  win.get("location", {"x": 0, "y": 0, "z": ctx["sill_height"]}),
            "width":     round(float(win.get("width", 1200)), 1),
            "height":    ctx["window_height"],
            "level":     "Level 0",
        })

    # ── Floors (one per room if rooms present) ────────────────────────────────
    floors = _build_slabs(geom.get("rooms", []), ctx, "floor")

    # ── Ceilings ─────────────────────────────────────────────────────────────
    ceilings = _build_slabs(geom.get("rooms", []), ctx, "ceiling")

    # ── Assemble transaction ──────────────────────────────────────────────────
    transaction = {
        "levels":    levels,
        "grids":     grids,
        "walls":     walls,
        "columns":   columns,
        "doors":     doors,
        "windows":   windows,
        "floors":    floors,
        "ceilings":  ceilings,
        "metadata":  {
            **geom.get("metadata", {}),
            "scale_info":      scale_info or {},
            "project_context": project_context or {},
            "standard":        "SS CP 65 / BCA DfMA 2021",
            "generated_by":    "MCC-Amplify-v2 BIM-Translator",
        },
    }

    element_counts = {
        "levels":   len(levels),
        "grids":    len(grids),
        "walls":    len(walls),
        "columns":  len(columns),
        "doors":    len(doors),
        "windows":  len(windows),
        "floors":   len(floors),
        "ceilings": len(ceilings),
    }

    return {
        "ok":                True,
        "transaction_json":  transaction,
        "element_counts":    element_counts,
        "unmapped_warnings": unmapped,
    }


def _build_slabs(rooms: list, ctx: dict, slab_type: str) -> list:
    slabs = []
    for i, room in enumerate(rooms):
        boundary_px = room.get("boundary", [])
        if not boundary_px and room.get("bbox"):
            x1, y1, x2, y2 = room["bbox"]
            boundary_px = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if not boundary_px:
            continue
        boundary_mm = [{"x": pt[0], "y": pt[1]} for pt in boundary_px]
        elevation = 0.0 if slab_type == "floor" else room.get("ceiling_height", ctx["wall_height"])
        slabs.append({
            "id":              f"{slab_type}_{i}",
            "boundary_points": boundary_mm,
            "thickness":       ctx["floor_thickness"] if slab_type == "floor" else 20,
            "elevation":       elevation,
            "level":           "Level 0",
        })
    return slabs


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — revit_api_client
# ══════════════════════════════════════════════════════════════════════════════

def revit_api_client(
    transaction_json: dict,
    job_id: str | None = None,
    output_dir: str = "data/models/rvt",
) -> dict:
    """
    Send a Revit Transaction JSON to the Windows Revit Add-in server.

    Attempts to import the RevitClient from the v1 project (mcc-amplify-ai)
    via PYTHONPATH.  Falls back to file-drop mode if the HTTP server is
    unreachable.  Returns a structured result including any C#/.NET error text
    captured by the WarningCollector in the Add-in.

    Args:
        transaction_json: Dict produced by revit_schema_mapper().
        job_id:           Unique job identifier (auto-generated if None).
        output_dir:       Where to write the resulting .rvt file.

    Returns:
        {
          "ok":        bool,
          "rvt_path":  str | None,
          "warnings":  [str],       — Revit build warnings (non-fatal)
          "error_log": str | None,  — C#/.NET exception text (if failed)
          "job_id":    str,
        }
    """
    job_id   = job_id or str(uuid.uuid4())
    out_path = Path(output_dir)
    tmp_json = out_path / f"{job_id}_transaction.json"

    try:
        out_path.mkdir(parents=True, exist_ok=True)
        tmp_json.write_text(json.dumps(transaction_json, indent=2), encoding="utf-8")

        client = _load_revit_client()
        if client is None:
            return {
                "ok":        False,
                "rvt_path":  None,
                "warnings":  [],
                "error_log": (
                    "RevitClient not available. "
                    "Add ~/Documents/mcc-amplify-ai to PYTHONPATH or set "
                    "REVIT_CLIENT_PATH env var to the backend directory."
                ),
                "job_id": job_id,
            }

        rvt_path, warnings = asyncio.run(
            client.build_model(str(tmp_json), job_id)
        )
        return {
            "ok":        True,
            "rvt_path":  rvt_path,
            "warnings":  warnings,
            "error_log": None,
            "job_id":    job_id,
        }

    except Exception as exc:
        return {
            "ok":        False,
            "rvt_path":  None,
            "warnings":  [],
            "error_log": f"{type(exc).__name__}: {exc}",
            "job_id":    job_id,
        }
    finally:
        try:
            tmp_json.unlink(missing_ok=True)
        except Exception:
            pass


def _load_revit_client():
    """Try to import RevitClient from v1 project on PYTHONPATH."""
    v1_path = os.getenv("REVIT_CLIENT_PATH",
        str(Path(__file__).parent.parent.parent / "mcc-amplify-ai"))
    if v1_path not in sys.path:
        sys.path.insert(0, v1_path)
    try:
        from backend.services.revit_client import RevitClient
        return RevitClient()
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — memory_io
# ══════════════════════════════════════════════════════════════════════════════

class memory_io:
    """
    Read/write helpers for translator/memory.sqlite.
    Stores "API Success Patterns" — which C# parameter combinations worked,
    and what corrections resolved prior Revit API failures.
    """

    @staticmethod
    def save_pattern(
        element_type: str,
        family_name: str,
        type_name: str,
        parameters: dict,
        outcome: str,
        error_message: str = "",
        correction_applied: str = "",
    ) -> dict:
        """
        Record a Revit API interaction outcome.
        Increments success_count for existing successful patterns.
        """
        ts        = datetime.now().isoformat()
        p_json    = json.dumps(parameters)
        con       = _db()
        with con:
            if outcome == "success":
                existing = con.execute(
                    "SELECT id FROM api_success_patterns "
                    "WHERE element_type=? AND family_name=? AND type_name=? LIMIT 1",
                    (element_type, family_name, type_name),
                ).fetchone()
                if existing:
                    con.execute(
                        "UPDATE api_success_patterns "
                        "SET success_count=success_count+1, timestamp=? WHERE id=?",
                        (ts, existing["id"]),
                    )
                    return {"ok": True, "id": existing["id"], "action": "incremented"}
            cur = con.execute(
                "INSERT INTO api_success_patterns "
                "(timestamp, element_type, family_name, type_name, parameters_json, "
                "revit_outcome, error_message, correction_applied) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, element_type, family_name, type_name, p_json,
                 outcome, error_message, correction_applied),
            )
        con.close()
        return {"ok": True, "id": cur.lastrowid, "action": "inserted"}

    @staticmethod
    def query_patterns(
        element_type: str,
        outcome: str = "success",
        error_substring: str = "",
    ) -> list[dict]:
        """
        Retrieve known good (or failure) patterns for a given element type.
        Optionally filter failures by an error substring to find matching corrections.
        """
        clauses, params = ["element_type=?", "revit_outcome=?"], [element_type, outcome]
        if error_substring and outcome == "failure":
            clauses.append("error_message LIKE ?")
            params.append(f"%{error_substring}%")
        con  = _db()
        rows = [dict(r) for r in con.execute(
            f"SELECT * FROM api_success_patterns WHERE {' AND '.join(clauses)} "
            "ORDER BY success_count DESC, timestamp DESC LIMIT 10",
            params,
        ).fetchall()]
        con.close()
        for row in rows:
            try:
                row["parameters_json"] = json.loads(row["parameters_json"] or "{}")
            except json.JSONDecodeError:
                pass
        return rows

    @staticmethod
    def save_run(
        run_id: str,
        job_id: str,
        status: str,
        elements_placed: int,
        revit_warnings: list,
        error_message: str = "",
    ) -> None:
        ts  = datetime.now().isoformat()
        con = _db()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO translation_runs "
                "(run_id, timestamp, job_id, status, elements_placed, revit_warnings, error_message) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, ts, job_id, status, elements_placed,
                 json.dumps(revit_warnings), error_message),
            )
        con.close()

    @staticmethod
    def recent_runs(limit: int = 20) -> list[dict]:
        con  = _db()
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM translation_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()]
        con.close()
        for row in rows:
            try:
                row["revit_warnings"] = json.loads(row["revit_warnings"] or "[]")
            except json.JSONDecodeError:
                pass
        return rows

    @staticmethod
    def stats() -> dict:
        con = _db()
        row = con.execute("""
            SELECT
                (SELECT COUNT(*) FROM api_success_patterns WHERE revit_outcome='success') AS successes,
                (SELECT COUNT(*) FROM api_success_patterns WHERE revit_outcome='failure') AS failures,
                (SELECT COUNT(*) FROM translation_runs) AS total_runs
        """).fetchone()
        con.close()
        return {
            "success_patterns": row["successes"] or 0,
            "failure_patterns": row["failures"]  or 0,
            "total_runs":       row["total_runs"] or 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _deep_copy(obj: Any) -> Any:
    return copy.deepcopy(obj)
