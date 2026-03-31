"""
validation/tools.py — Private toolset for the Validation Agent
================================================================
Tools:
  geometry_checker          — runs all DfMA rule checks on raw geometry
  loop_closer               — detects and closes open wall loops
  standard_thickness_lookup — returns Singapore / MCC DfMA wall thickness standards
  memory_io                 — read/write helpers for validation/memory.sqlite
                              (stores "Geometric Conflict Resolutions")

All tools return plain dicts.  No tool raises — failures surface as
{"ok": False, "error": "..."}.

Singapore standards reference: SS CP 65, BCA DfMA Advisory 2021.
"""

from __future__ import annotations

import copy
import json
import math
import sqlite3
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
    CREATE TABLE IF NOT EXISTS conflict_resolutions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         TEXT NOT NULL,
        feature_signature TEXT NOT NULL,
        element_type      TEXT NOT NULL,
        rule_code         TEXT NOT NULL,
        original_value    TEXT,
        corrected_value   TEXT,
        rule_applied      TEXT NOT NULL,
        success_count     INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS validation_runs (
        run_id            TEXT PRIMARY KEY,
        timestamp         TEXT NOT NULL,
        feature_signature TEXT,
        status            TEXT,
        issues_count      INTEGER DEFAULT 0,
        corrections_count INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_cr_feat ON conflict_resolutions(feature_signature);
    CREATE INDEX IF NOT EXISTS idx_cr_rule ON conflict_resolutions(rule_code);
    CREATE INDEX IF NOT EXISTS idx_vr_ts   ON validation_runs(timestamp);
    """)
    return con


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — standard_thickness_lookup
# ══════════════════════════════════════════════════════════════════════════════

# Singapore / MCC DfMA dimension standards (all mm)
_SG_STANDARDS = {
    # Walls
    "wall": {
        "Interior RC":          {"min": 150, "max": 250, "default": 200},
        "Interior Blockwork":   {"min": 100, "max": 150, "default": 100},
        "Exterior RC":          {"min": 200, "max": 350, "default": 300},
        "Exterior Precast":     {"min": 180, "max": 250, "default": 200},
        "Shear Wall":           {"min": 200, "max": 400, "default": 250},
        "Party Wall":           {"min": 200, "max": 300, "default": 200},
    },
    # Columns
    "column": {
        "min_section_mm":       200,    # Revit extrusion floor
        "max_section_mm":       1500,
        "typical_residential":  [600, 800, 1000],
        "typical_commercial":   [800, 1000, 1200],
    },
    # Floors
    "floor": {
        "RC Slab min":          150,
        "RC Slab typical":      200,
        "PT Slab typical":      250,
    },
    # Storey heights
    "storey": {
        "residential_typical_mm": 3000,
        "commercial_typical_mm":  3600,
        "min_mm":                 2700,
        "max_mm":                 5000,
    },
    # Grid
    "grid": {
        "confidence_threshold": 0.75,
        "snap_tolerance_mm":    50,
    },
}


def standard_thickness_lookup(
    element_type: str,
    sub_type: str = "",
) -> dict:
    """
    Return DfMA dimension standards for a given element type.

    Args:
        element_type: One of 'wall', 'column', 'floor', 'storey', 'grid'.
        sub_type:     Wall sub-type string (e.g. 'Interior RC', 'Exterior RC').
                      For other element types, leave empty.

    Returns:
        Standards dict, or {"ok": False, "error": "..."} for unknown types.

    Examples:
        standard_thickness_lookup("wall", "Interior RC")
        → {"min": 150, "max": 250, "default": 200}

        standard_thickness_lookup("column")
        → {"min_section_mm": 200, "max_section_mm": 1500, ...}

        standard_thickness_lookup("storey")
        → {"residential_typical_mm": 3000, ...}
    """
    if element_type not in _SG_STANDARDS:
        return {
            "ok":    False,
            "error": (
                f"Unknown element_type '{element_type}'. "
                f"Valid: {list(_SG_STANDARDS.keys())}"
            ),
        }

    standards = _SG_STANDARDS[element_type]

    if element_type == "wall" and sub_type:
        if sub_type in standards:
            return {"ok": True, "element_type": element_type, "sub_type": sub_type,
                    **standards[sub_type]}
        # Fuzzy match — find the closest sub-type key
        sub_lower = sub_type.lower()
        for key, vals in standards.items():
            if any(word in sub_lower for word in key.lower().split()):
                return {"ok": True, "element_type": element_type, "sub_type": key,
                        **vals, "note": f"Matched '{sub_type}' → '{key}'"}
        # Default to Interior RC
        default_vals = standards["Interior RC"]
        return {"ok": True, "element_type": element_type, "sub_type": "Interior RC",
                **default_vals, "note": f"Sub-type '{sub_type}' unknown — using Interior RC default"}

    return {"ok": True, "element_type": element_type, **standards}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — geometry_checker
# ══════════════════════════════════════════════════════════════════════════════

def geometry_checker(raw_geometry: dict, project_context: dict | None = None) -> dict:
    """
    Run all DfMA geometry checks on the raw geometry payload.

    Rule codes:
      G1  Grid confidence threshold
      G2  Both grid axes present
      C1  Column shape is known
      C2  Column section dimensions in range
      C3  Column centre coordinates present
      D1  Duplicate column locations
      W1  Wall thickness within DfMA range
      W2  Non-zero wall length
      O1  Opening dimensions in range

    Args:
        raw_geometry:    Output of detection_parser (from Detection Agent).
        project_context: Optional overrides to DfMA standards (e.g. storey height).

    Returns:
        {
          "status":      "passed" | "warnings" | "failed",
          "issues":      [{"code", "severity", "element_type", "element_id", "msg"}],
          "corrections": [{"field", "original", "corrected", "rule"}],
          "geometry":    <deep copy of raw_geometry with corrections applied>,
        }
    """
    ctx      = _merge_context(project_context)
    issues:      list[dict] = []
    corrections: list[dict] = []
    geometry = _deep_copy(raw_geometry)

    # ── G1: Grid confidence ───────────────────────────────────────────────────
    grid = geometry.get("grid", {})
    conf = float(grid.get("confidence", 0.0))
    thresh = _SG_STANDARDS["grid"]["confidence_threshold"]
    if conf < thresh:
        issues.append(_issue("G1", "warning", "grid", "grid",
            f"Grid confidence {conf:.2f} < {thresh:.2f}. "
            "Detection may be unreliable — check for scanned or low-resolution PDF."))

    # ── G2: Both axes ─────────────────────────────────────────────────────────
    if not grid.get("vertical_labels"):
        issues.append(_issue("G2", "warning", "grid", "vertical",
            "No vertical grid lines detected."))
    if not grid.get("horizontal_labels"):
        issues.append(_issue("G2", "warning", "grid", "horizontal",
            "No horizontal grid lines detected."))

    # ── Column checks ─────────────────────────────────────────────────────────
    seen_centres: list[tuple] = []
    col_std = _SG_STANDARDS["column"]

    for i, col in enumerate(geometry.get("columns", [])):
        col_id = col.get("id", i)

        # C1 — known shape
        shape = col.get("shape", "")
        if shape not in ("rectangular", "circular", "column"):
            issues.append(_issue("C1", "warning", "column", col_id,
                f"Unknown shape '{shape}'. Correcting to 'rectangular'."))
            corrections.append(_correction(
                f"columns[{i}].shape", shape, "rectangular", "C1"))
            col["shape"] = "rectangular"

        # C2 — section dimensions
        lo, hi = col_std["min_section_mm"], col_std["max_section_mm"]
        if col.get("is_circular"):
            diam = col.get("diameter_mm")
            if diam is None:
                defval = ctx.get("default_column_size_mm", col_std["min_section_mm"])
                issues.append(_issue("C2", "warning", "column", col_id,
                    f"Circular column diameter unknown. Applying DfMA minimum {defval} mm."))
                corrections.append(_correction(
                    f"columns[{i}].diameter_mm", None, defval, "C2"))
                col["diameter_mm"] = defval
            else:
                diam = float(diam)
                if not (lo <= diam <= hi):
                    clamped = _clamp(diam, lo, hi)
                    issues.append(_issue("C2", "error", "column", col_id,
                        f"Circular diameter {diam:.0f} mm outside DfMA range [{lo},{hi}]. Clamped to {clamped:.0f} mm."))
                    corrections.append(_correction(
                        f"columns[{i}].diameter_mm", diam, clamped, "C2"))
                    col["diameter_mm"] = clamped
        else:
            for dim_key in ("width_mm", "depth_mm"):
                val = col.get(dim_key)
                if val is None:
                    defval = ctx.get("default_column_size_mm", col_std["min_section_mm"])
                    issues.append(_issue("C2", "warning", "column", col_id,
                        f"{dim_key} unknown. Applying DfMA minimum {defval} mm."))
                    corrections.append(_correction(
                        f"columns[{i}].{dim_key}", None, defval, "C2"))
                    col[dim_key] = defval
                else:
                    val = float(val)
                    if not (lo <= val <= hi):
                        clamped = _clamp(val, lo, hi)
                        issues.append(_issue("C2", "error", "column", col_id,
                            f"{dim_key} {val:.0f} mm outside range [{lo},{hi}]. Clamped to {clamped:.0f} mm."))
                        corrections.append(_correction(
                            f"columns[{i}].{dim_key}", val, clamped, "C2"))
                        col[dim_key] = clamped

        # C3 — centre present
        centre = col.get("center") or [None, None]
        if centre[0] is None or centre[1] is None:
            issues.append(_issue("C3", "error", "column", col_id,
                "Centre coordinates missing. Column cannot be placed in Revit."))
        else:
            # D1 — duplicate (round to 50 px grid)
            snap = (round(centre[0] / 50) * 50, round(centre[1] / 50) * 50)
            if snap in seen_centres:
                issues.append(_issue("D1", "warning", "column", col_id,
                    f"Likely duplicate at pixel centre ≈ {snap}. Check detection overlap."))
            seen_centres.append(snap)

    # ── Wall checks ───────────────────────────────────────────────────────────
    for i, wall in enumerate(geometry.get("walls", [])):
        wall_id = wall.get("id", i)
        func    = wall.get("function", "Interior")
        sub_key = f"{'Exterior' if 'Exterior' in func else 'Interior'} RC"
        w_std   = _SG_STANDARDS["wall"][sub_key]

        thick = wall.get("thickness_mm") or wall.get("thickness")
        if thick is not None:
            thick = float(thick)
            lo_w, hi_w, def_w = w_std["min"], w_std["max"], w_std["default"]
            if not (lo_w <= thick <= hi_w):
                issues.append(_issue("W1", "warning", "wall", wall_id,
                    f"Thickness {thick:.0f} mm outside DfMA [{lo_w},{hi_w}]. Using default {def_w} mm."))
                corrections.append(_correction(
                    f"walls[{i}].thickness_mm", thick, def_w, "W1"))
                wall["thickness_mm"] = def_w

        # W2 — non-zero length
        s = wall.get("start_point", {}) or {}
        e = wall.get("end_point",   {}) or {}
        sx, sy = s.get("x", 0), s.get("y", 0)
        ex, ey = e.get("x", 0), e.get("y", 0)
        if abs(sx - ex) < 1e-6 and abs(sy - ey) < 1e-6:
            issues.append(_issue("W2", "error", "wall", wall_id,
                "Start == End — zero-length wall will be rejected by Revit API."))

    # ── Openings checks (O1) ─────────────────────────────────────────────────
    for i, opening in enumerate(
        geometry.get("doors", []) + geometry.get("windows", [])
    ):
        o_id  = opening.get("id", i)
        width = opening.get("width")
        if width is not None and float(width) < 200:
            issues.append(_issue("O1", "warning", "opening", o_id,
                f"Opening width {width} mm < 200 mm minimum. Revit family may reject."))

    # ── Status ────────────────────────────────────────────────────────────────
    severities = {iss["severity"] for iss in issues}
    status = "failed" if "error" in severities else ("warnings" if severities else "passed")

    return {
        "status":      status,
        "issues":      issues,
        "corrections": corrections,
        "geometry":    geometry,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — loop_closer
# ══════════════════════════════════════════════════════════════════════════════

_ENDPOINT_SNAP_PX = 10.0  # pixels — two endpoints within this are considered shared


def loop_closer(geometry: dict) -> dict:
    """
    Detect open wall loops and close them by snapping nearby endpoints.

    An "open loop" occurs when a wall's endpoint is close to — but not exactly
    coincident with — another wall's endpoint, creating a gap in the floor plan
    boundary.  This causes Revit's Room Bounding and area calculation to fail.

    Algorithm:
      1. Collect all wall endpoints.
      2. For each endpoint, find other endpoints within ENDPOINT_SNAP_PX.
      3. Average the cluster to a single point and update all affected walls.

    Args:
        geometry: Validated geometry dict (output of geometry_checker).
                  Must have a "walls" key containing wall dicts with
                  "start_point" and "end_point" (x, y coords in px or mm).

    Returns:
        {
          "ok":          bool,
          "gaps_closed": int,   — number of endpoint pairs snapped together
          "geometry":    dict,  — geometry with corrected wall endpoints
          "log":         [str], — description of each closure
        }
    """
    walls  = geometry.get("walls", [])
    log:   list[str] = []
    snaps  = 0

    if not walls:
        return {"ok": True, "gaps_closed": 0, "geometry": geometry, "log": ["No walls to process."]}

    # Build flat list of all endpoints with back-references
    # Each entry: (wall_idx, endpoint_key, x, y)
    endpoints: list[tuple] = []
    for wi, wall in enumerate(walls):
        for ep_key in ("start_point", "end_point"):
            pt = wall.get(ep_key) or {}
            x  = pt.get("x") or pt.get("px_x")
            y  = pt.get("y") or pt.get("px_y")
            if x is not None and y is not None:
                endpoints.append((wi, ep_key, float(x), float(y)))

    processed: set[int] = set()

    for i, (wi, wk, x1, y1) in enumerate(endpoints):
        if i in processed:
            continue
        cluster = [(wi, wk, x1, y1)]
        cluster_idxs = [i]

        for j, (wj, wkj, x2, y2) in enumerate(endpoints[i+1:], i+1):
            if j in processed:
                continue
            if math.hypot(x2 - x1, y2 - y1) <= _ENDPOINT_SNAP_PX:
                cluster.append((wj, wkj, x2, y2))
                cluster_idxs.append(j)

        if len(cluster) > 1:
            # Snap all to centroid
            avg_x = sum(c[2] for c in cluster) / len(cluster)
            avg_y = sum(c[3] for c in cluster) / len(cluster)
            for c_wi, c_wk, cx, cy in cluster:
                pt = walls[c_wi].setdefault(c_wk, {})
                if "x" in pt or "y" in pt:
                    pt["x"], pt["y"] = avg_x, avg_y
                else:
                    pt["px_x"], pt["px_y"] = avg_x, avg_y
            log.append(
                f"Snapped {len(cluster)} endpoints → centroid ({avg_x:.1f}, {avg_y:.1f})"
            )
            snaps += len(cluster) - 1
            for idx in cluster_idxs:
                processed.add(idx)

    return {
        "ok":          True,
        "gaps_closed": snaps,
        "geometry":    geometry,
        "log":         log or ["No gaps found."],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — memory_io
# ══════════════════════════════════════════════════════════════════════════════

class memory_io:
    """
    Read/write helpers for validation/memory.sqlite.
    Stores "Geometric Conflict Resolutions" — how the agent fixed past issues
    so it can proactively apply known good solutions on future similar drawings.
    """

    @staticmethod
    def save_resolution(
        feature_signature: str,
        element_type: str,
        rule_code: str,
        original_value: Any,
        corrected_value: Any,
        rule_applied: str,
    ) -> dict:
        """
        Record a successful conflict resolution.
        Increments success_count if the same resolution already exists.
        """
        ts  = datetime.now().isoformat()
        con = _db()
        with con:
            existing = con.execute(
                "SELECT id FROM conflict_resolutions "
                "WHERE feature_signature=? AND element_type=? AND rule_code=? LIMIT 1",
                (feature_signature, element_type, rule_code),
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE conflict_resolutions SET success_count=success_count+1, timestamp=? WHERE id=?",
                    (ts, existing["id"]),
                )
                rid = existing["id"]
            else:
                cur = con.execute(
                    "INSERT INTO conflict_resolutions "
                    "(timestamp, feature_signature, element_type, rule_code, "
                    "original_value, corrected_value, rule_applied) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (ts, feature_signature, element_type, rule_code,
                     str(original_value), str(corrected_value), rule_applied),
                )
                rid = cur.lastrowid
        con.close()
        return {"ok": True, "id": rid}

    @staticmethod
    def query_resolutions(
        feature_signature: str,
        rule_code: str = "",
        element_type: str = "",
    ) -> list[dict]:
        """
        Find prior resolutions for a given feature signature.
        Returns most-successful entries first.
        """
        clauses, params = ["feature_signature=?"], [feature_signature]
        if rule_code:
            clauses.append("rule_code=?")
            params.append(rule_code)
        if element_type:
            clauses.append("element_type=?")
            params.append(element_type)
        con  = _db()
        rows = [dict(r) for r in con.execute(
            f"SELECT * FROM conflict_resolutions WHERE {' AND '.join(clauses)} "
            "ORDER BY success_count DESC, timestamp DESC LIMIT 20",
            params,
        ).fetchall()]
        con.close()
        return rows

    @staticmethod
    def save_run(
        run_id: str,
        feature_signature: str,
        status: str,
        issues_count: int,
        corrections_count: int,
    ) -> None:
        ts  = datetime.now().isoformat()
        con = _db()
        with con:
            con.execute(
                "INSERT OR REPLACE INTO validation_runs "
                "(run_id, timestamp, feature_signature, status, issues_count, corrections_count) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, ts, feature_signature, status, issues_count, corrections_count),
            )
        con.close()

    @staticmethod
    def recent_runs(limit: int = 20) -> list[dict]:
        con  = _db()
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM validation_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()]
        con.close()
        return rows

    @staticmethod
    def stats() -> dict:
        con = _db()
        row = con.execute("""
            SELECT
                (SELECT COUNT(*) FROM conflict_resolutions) AS total_resolutions,
                (SELECT SUM(success_count) FROM conflict_resolutions) AS total_uses,
                (SELECT COUNT(*) FROM validation_runs) AS total_runs,
                (SELECT COUNT(*) FROM validation_runs WHERE status='passed') AS passed
        """).fetchone()
        con.close()
        return {
            "total_resolutions": row["total_resolutions"] or 0,
            "total_uses":        row["total_uses"]        or 0,
            "total_runs":        row["total_runs"]        or 0,
            "passed_runs":       row["passed"]            or 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _issue(code, severity, element_type, element_id, msg) -> dict:
    return {
        "code":         code,
        "severity":     severity,
        "element_type": element_type,
        "element_id":   element_id,
        "msg":          msg,
    }


def _correction(field, original, corrected, rule) -> dict:
    return {
        "field":     field,
        "original":  original,
        "corrected": corrected,
        "rule":      rule,
    }


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def _merge_context(project_context: dict | None) -> dict:
    base = {
        "default_column_size_mm": 200,
        "storey_height_mm":       3000,
    }
    if project_context:
        base.update(project_context)
    return base


def _deep_copy(obj: Any) -> Any:
    return copy.deepcopy(obj)
