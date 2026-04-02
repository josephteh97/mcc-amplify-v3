"""
controller.py — Pipeline Orchestrator
=======================================
Coordinates the three-agent pipeline:

  [PDF Detection] → [Validation] → [BIM-Translator] → .rvt

The controller acts as a "courier":
  • It injects "Project Context" metadata between agents.
  • It forwards "Refinement Requests" from the BIM-Translator back to
    the Validation Agent (one re-validation attempt per pipeline run).
  • It does NOT share memory or tool references between agents.
    Each agent's memory is strictly private.

Usage (CLI):
    python controller.py <pdf_path> [--context project_context.json] [--page 0]

Usage (Python):
    from controller import run_pipeline
    result = run_pipeline("floor_plan.pdf", project_context={...}, page_num=0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

for _p in (str(_ROOT), str(Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validation.agent  import ValidationAgent
from translator.agent  import BIMTranslatorAgent
import gltf_exporter

# Detection agents
_GRID_AGENT_DIR = _ROOT / "grid-detection-agent"

# YOLO column agent (replaces pdf_detection_agent Ollama-vision approach)
sys.path.insert(0, str(_ROOT))
from yolo_detection_agents.column_agent import YOLOColumnAgent as _YOLOColumnAgent


# ══════════════════════════════════════════════════════════════════════════════
# Detection Agent adapters (wrap existing agents without modifying them)
# ══════════════════════════════════════════════════════════════════════════════

def _run_grid_detection(pdf_path: str, verbose: bool = False) -> dict:
    if not _GRID_AGENT_DIR.exists():
        print(f"  [controller] grid-detection-agent not found at {_GRID_AGENT_DIR}")
        return {}
    sys.path.insert(0, str(_GRID_AGENT_DIR))
    try:
        import agent as grid_agent_module
        result = grid_agent_module.run(pdf_path, verbose=verbose)
        print(
            f"  [controller] Grid: {result.get('total_grid_lines', 0)} lines, "
            f"conf={result.get('confidence', 0):.2f}"
        )
        return result
    except Exception as exc:
        print(f"  [controller] Grid detection failed: {exc}")
        return {}
    finally:
        sys.path.remove(str(_GRID_AGENT_DIR))


def _run_column_detection(pdf_path: str, page_num: int = 0) -> dict:
    try:
        agent  = _YOLOColumnAgent()
        result = agent.detect(pdf_path, page_num=page_num)
        if "error" in result:
            print(f"  [controller] Column detection error: {result['error']}")
            return {}
        print(
            f"  [controller] Columns: {result.get('total_columns', 0)} on page {page_num} "
            f"[model={result.get('model', 'unknown')}]"
        )
        return result
    except Exception as exc:
        print(f"  [controller] Column detection failed: {exc}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# detection_parser (controller-level — no shared tools module)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_detections(
    pdf_path: str,
    grid_result: dict,
    column_result: dict,
) -> dict:
    """
    Normalise Detection Agent outputs into a canonical "Raw Geometry JSON"
    without importing any shared tools module.

    This is an intentionally thin normalisation pass — the heavy geometric
    validation is done by the Validation Agent.
    """
    import hashlib
    import uuid as _uuid
    import os

    def _hash_pdf(path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read(65536)).hexdigest()[:16]
        except OSError:
            return hashlib.md5(path.encode()).hexdigest()[:16]

    def _feature_sig(grid: dict, columns: list) -> str:
        n_lines = grid.get("total_grid_lines", 0)
        n_cols  = len(columns)
        parts   = []
        parts.append("Dense Grid (>16 lines)"   if n_lines > 16
                      else "Medium Grid (9-16)"  if n_lines > 8
                      else "Simple Grid (≤8)"    if n_lines > 0
                      else "No Grid Detected")
        parts.append("High Column Density (>30)" if n_cols > 30
                      else "Medium Column Density (11-30)" if n_cols > 10
                      else "Low Column Density (≤10)"      if n_cols > 0
                      else "No Columns Detected")
        shapes = {c.get("shape") for c in columns}
        if "circular" in shapes and "rectangular" in shapes:
            parts.append("Mixed Column Shapes")
        elif "circular" in shapes:
            parts.append("Circular Columns")
        elif "rectangular" in shapes:
            parts.append("Rectangular Columns")
        return ", ".join(parts)

    grid = {
        "vertical_labels":   grid_result.get("vertical_labels",   []),
        "horizontal_labels": grid_result.get("horizontal_labels", []),
        "total_grid_lines":  grid_result.get("total_grid_lines",  0),
        "confidence":        float(grid_result.get("confidence",  0.0)),
        "notes":             grid_result.get("notes", ""),
    }

    columns = []
    for det in column_result.get("detections", []):
        bb  = det.get("bbox_page") or [None, None, None, None]
        cx  = ((bb[0] or 0) + (bb[2] or 0)) / 2 if bb[0] is not None else None
        cy  = ((bb[1] or 0) + (bb[3] or 0)) / 2 if bb[1] is not None else None
        columns.append({
            "id":          det.get("id"),
            "shape":       det.get("shape", "rectangular"),
            "confidence":  float(det.get("confidence") or 0.0),
            "bbox_page":   bb,
            "center":      [cx, cy],
            "width_mm":    det.get("width_mm"),
            "depth_mm":    det.get("depth_mm"),
            "diameter_mm": det.get("diameter_mm"),
            "is_circular": det.get("is_circular", False),
            "type_mark":   det.get("type_mark"),
            "notes":       det.get("notes", ""),
            "tile_index":  det.get("tile_index"),
        })

    img_size = column_result.get("image_size") or [None, None]

    return {
        "job_id":            str(_uuid.uuid4()),
        "pdf_path":          pdf_path,
        "pdf_hash":          _hash_pdf(pdf_path),
        "feature_signature": _feature_sig(grid, columns),
        "grid":              grid,
        "columns":           columns,
        "walls":             [],
        "doors":             [],
        "windows":           [],
        "metadata": {
            "image_w":       img_size[0],
            "image_h":       img_size[1],
            "source_page":   column_result.get("page", 0),
            "model":         column_result.get("model"),
            "rendered_image": grid_result.get("_rendered_image"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pdf_path: str,
    project_context: dict | None = None,
    page_num: int = 0,
    verbose: bool = False,
    job_id: str | None = None,
) -> dict:
    """
    Execute the full Detection → Validation → Translation pipeline.

    Args:
        pdf_path:        Path to the PDF floor plan.
        project_context: Optional dict overriding DfMA defaults.
                         Keys: storey_height_mm, bay_widths_x_mm, bay_widths_y_mm,
                               wall_thickness_interior_mm, default_column_size_mm, …
        page_num:        PDF page index for column detection (0-based).
        verbose:         If True, detection agents emit verbose step logs.

    Returns:
        {
          "ok":                bool,
          "job_id":            str,
          "pdf_path":          str,
          "stage_reached":     "detection"|"validation"|"translation"|"complete",
          "validation_status": str,
          "rvt_path":          str | None,
          "warnings":          [str],
          "error_log":         str | None,
          "refinement_request": None | dict,
          "element_counts":    dict,
          "timings_s":         dict,
          "raw_geometry":      dict,
          "validated_payload": dict,
          "translator_result": dict,
        }
    """
    timings: dict[str, float] = {}
    ctx     = project_context or {}

    _banner("PIPELINE START")
    print(f"  PDF:  {pdf_path}")
    print(f"  Page: {page_num}")
    print(f"  Context keys: {list(ctx.keys()) or 'defaults'}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Detection
    # ══════════════════════════════════════════════════════════════════════════
    _banner("STAGE 1 — Detection")
    t0 = time.time()

    # Grid and column detection are independent — run in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_grid   = pool.submit(_run_grid_detection,   pdf_path, verbose)
        f_cols   = pool.submit(_run_column_detection, pdf_path, page_num)
        grid_result   = f_grid.result()
        column_result = f_cols.result()
    raw_geometry = _parse_detections(pdf_path, grid_result, column_result)
    if job_id:
        raw_geometry["job_id"] = job_id

    timings["detection_s"] = round(time.time() - t0, 2)
    job_id = raw_geometry["job_id"]

    print(
        f"  Feature signature: {raw_geometry['feature_signature']}\n"
        f"  Grid: {raw_geometry['grid']['total_grid_lines']} lines | "
        f"Columns: {len(raw_geometry['columns'])}"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Validation
    # ══════════════════════════════════════════════════════════════════════════
    _banner("STAGE 2 — Validation")
    t0 = time.time()

    validation_payload = {
        **raw_geometry,
        "project_context": ctx,
    }

    validator         = ValidationAgent()
    validated_payload = validator.run(validation_payload)
    timings["validation_s"] = round(time.time() - t0, 2)

    if not validated_payload.get("ok"):
        return _pipeline_error(
            stage="validation",
            job_id=job_id,
            pdf_path=pdf_path,
            error=validated_payload.get("error", "Validation agent failed"),
            raw_geometry=raw_geometry,
            timings=timings,
        )

    val_status = validated_payload.get("validation_status", "unknown")
    print(
        f"  Validation status: {val_status} | "
        f"{len(validated_payload.get('issues', []))} issue(s) | "
        f"{len(validated_payload.get('corrections', []))} correction(s)"
    )

    # If a refinement request is raised at validation stage, abort early
    ref_req = validated_payload.get("refinement_request")
    if ref_req:
        print(
            f"  [controller] Refinement request from Validation: "
            f"{ref_req.get('reason','')}"
        )
        return {
            "ok":                False,
            "job_id":            job_id,
            "pdf_path":          pdf_path,
            "stage_reached":     "validation",
            "validation_status": val_status,
            "rvt_path":          None,
            "warnings":          [],
            "error_log":         ref_req.get("reason"),
            "refinement_request": ref_req,
            "element_counts":    {},
            "timings_s":         timings,
            "raw_geometry":      raw_geometry,
            "validated_payload": validated_payload,
            "translator_result": {},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — BIM Translation
    # ══════════════════════════════════════════════════════════════════════════
    _banner("STAGE 3 — BIM Translation")
    t0 = time.time()

    # Enrich payload: inject validation issues as context for the translator
    translator_payload = {
        **validated_payload,
        "_validation_issues": validated_payload.get("issues", []),
    }

    translator       = BIMTranslatorAgent()
    translator_result = translator.run(translator_payload)
    timings["translation_s"] = round(time.time() - t0, 2)

    trans_ref_req = translator_result.get("refinement_request")

    # ── Self-healing loop: translator requests re-validation ─────────────────
    if trans_ref_req and trans_ref_req.get("type") == "revit_api_failure":
        _banner("STAGE 3b — Re-Validation (Refinement Request)")
        print(
            f"  [controller] Translator escalated: {trans_ref_req.get('reason','')}\n"
            f"  Requesting re-validation with additional context …"
        )
        t0 = time.time()

        # Inject translator's failure context into the validation payload
        re_val_payload = {
            **validation_payload,
            "project_context": {
                **ctx,
                "_translator_error": trans_ref_req.get("error", ""),
                "_translator_hint":  trans_ref_req.get("hint",  ""),
            },
        }
        validator2       = ValidationAgent()
        re_validated     = validator2.run(re_val_payload)
        timings["re_validation_s"] = round(time.time() - t0, 2)

        if re_validated.get("ok") and not re_validated.get("refinement_request"):
            # Retry translation with freshly validated geometry
            _banner("STAGE 3c — BIM Translation Retry")
            t0 = time.time()
            retry_payload    = {**re_validated, "_validation_issues": re_validated.get("issues", [])}
            translator2      = BIMTranslatorAgent()
            translator_result = translator2.run(retry_payload)
            timings["translation_retry_s"] = round(time.time() - t0, 2)
        else:
            print("  [controller] Re-validation also produced unresolvable issues. Pipeline halted.")

    # ══════════════════════════════════════════════════════════════════════════
    # RESULT
    # ══════════════════════════════════════════════════════════════════════════
    success   = translator_result.get("ok", False) and translator_result.get("rvt_path") is not None
    rvt_path  = translator_result.get("rvt_path")
    warnings  = translator_result.get("warnings", [])
    error_log = translator_result.get("error_log")

    # ── glTF export — always attempted when translation produced geometry ──────
    gltf_path = None
    tx_geometry = translator_result.get("transaction_json", {})
    if tx_geometry:
        try:
            gltf_out = _ROOT / "data" / "models" / "gltf" / f"{job_id}.glb"
            gltf_path = gltf_exporter.export(tx_geometry, str(gltf_out))
        except Exception as exc:
            print(f"  [controller] glTF export skipped: {exc}")

    _banner("PIPELINE COMPLETE" if success else "PIPELINE FAILED")
    print(
        f"  Status:    {'SUCCESS' if success else 'FAILED'}\n"
        f"  RVT path:  {rvt_path or 'N/A'}\n"
        f"  Warnings:  {len(warnings)}\n"
        f"  Timings:   {timings}"
    )

    return {
        "ok":                success,
        "job_id":            job_id,
        "pdf_path":          pdf_path,
        "stage_reached":     "complete" if success else "translation",
        "validation_status": val_status,
        "rvt_path":          rvt_path,
        "gltf_path":         gltf_path,
        "warnings":          warnings,
        "error_log":         error_log,
        "refinement_request": translator_result.get("refinement_request"),
        "element_counts":    translator_result.get("element_counts", {}),
        "timings_s":         timings,
        "raw_geometry":      raw_geometry,
        "validated_payload": validated_payload,
        "translator_result": translator_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def _pipeline_error(
    stage: str,
    job_id: str,
    pdf_path: str,
    error: str,
    raw_geometry: dict,
    timings: dict,
) -> dict:
    print(f"  [controller] Pipeline error at {stage}: {error}")
    return {
        "ok":                False,
        "job_id":            job_id,
        "pdf_path":          pdf_path,
        "stage_reached":     stage,
        "validation_status": "failed",
        "rvt_path":          None,
        "warnings":          [],
        "error_log":         error,
        "refinement_request": None,
        "element_counts":    {},
        "timings_s":         timings,
        "raw_geometry":      raw_geometry,
        "validated_payload": {},
        "translator_result": {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCC-Amplify-v2 Pipeline Controller — PDF → BIM"
    )
    parser.add_argument("pdf_path",  help="Path to the floor plan PDF")
    parser.add_argument("--context", help="Path to project_context.json", default=None)
    parser.add_argument("--page",    help="PDF page index (0-based)", type=int, default=0)
    parser.add_argument("--verbose", help="Enable verbose detection logs", action="store_true")
    parser.add_argument("--output",  help="Write result JSON to this file", default=None)
    args = parser.parse_args()

    project_context = {}
    if args.context:
        ctx_path = Path(args.context)
        if not ctx_path.is_absolute():
            ctx_path = _ROOT / ctx_path
        with open(ctx_path, encoding="utf-8") as f:
            project_context = json.load(f)

    result = run_pipeline(
        pdf_path        = args.pdf_path,
        project_context = project_context,
        page_num        = args.page,
        verbose         = args.verbose,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult written to: {args.output}")
    else:
        print("\n" + json.dumps(result, indent=2, default=str))
