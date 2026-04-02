"""
seed_memory.py — Bootstrap agent memories with known DfMA rules from v1
=========================================================================
Run once after cloning the repo to pre-populate:
  • validation/memory.sqlite  — known geometric conflict resolutions
  • translator/memory.sqlite  — known Revit API success patterns

Extracted from mcc-amplify-ai:
  - backend/core/family_mapping.json  → Revit family patterns
  - backend/services/geometry_generator.py  → dimension defaults + validation logic
  - backend/services/revit_client.py  → retry + error patterns

Usage:
    python seed_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def seed_validation_memory() -> None:
    from validation.tools import memory_io

    print("Seeding validation/memory.sqlite …")

    # ── Known geometric conflict resolutions ──────────────────────────────────
    resolutions = [
        # (feature_signature, element_type, rule_code, original, corrected, rule_applied)
        ("*", "column", "C2",
         "None", "200",
         "BCA DfMA: minimum RC column section 200 mm (Revit extrusion floor)"),

        ("*", "column", "C2",
         "<200", "200",
         "BCA DfMA: column section below 200 mm raises Revit extrusion error — clamped to 200 mm"),

        ("*", "wall", "W1",
         "145", "150",
         "SS CP 65: interior blockwork minimum 100 mm; RC shear minimum 150 mm — corrected to 150 mm"),

        ("*", "wall", "W1",
         "100", "200",
         "BCA DfMA: RC interior walls default to 200 mm for precast compatibility"),

        ("High Column Density (>30), Rectangular Columns", "column", "D1",
         "duplicate_bbox", "removed",
         "High-density drawings produce YOLO overlap detections — duplicates removed"),

        ("Dense Grid (>16 lines), Rectangular Columns", "column", "C2",
         "None", "800",
         "Commercial/complex core: typical column 800x800 mm per MCC Singapore standard"),

        ("Simple Grid (≤8), Rectangular Columns", "column", "C2",
         "None", "600",
         "Residential: typical column 600x600 mm per MCC Singapore standard"),

        ("*", "grid", "G1",
         "confidence<0.75", "flag_for_review",
         "Low-confidence grid from SEA-LION on scanned PDF — flag for human review"),

        ("*", "column", "C1",
         "unknown_shape", "rectangular",
         "Unrecognised shape token defaulted to rectangular per v1 geometry_generator logic"),
    ]

    for (feat_sig, el_type, rule, orig, corr, rule_applied) in resolutions:
        result = memory_io.save_resolution(
            feature_signature=feat_sig,
            element_type=el_type,
            rule_code=rule,
            original_value=orig,
            corrected_value=corr,
            rule_applied=rule_applied,
        )
        print(f"  [{rule}] {el_type:10s} | {rule_applied[:70]} … {result}")

    print(f"Validation memory seeded: {len(resolutions)} resolutions.\n")


def seed_translator_memory() -> None:
    from translator.tools import memory_io

    print("Seeding translator/memory.sqlite …")

    # ── Revit API success patterns (from v1 family_mapping.json) ─────────────
    patterns = [
        # (element_type, family_name, type_name, parameters, outcome)
        ("column", "M_Concrete-Rectangular-Column", "600 x 600mm",
         {"b": 600, "d": 600, "Column Height": 3000}, "success"),

        ("column", "M_Concrete-Rectangular-Column", "800 x 800mm",
         {"b": 800, "d": 800, "Column Height": 3000}, "success"),

        ("column", "M_Concrete-Rectangular-Column", "1000 x 1000mm",
         {"b": 1000, "d": 1000, "Column Height": 3000}, "success"),

        ("column", "M_Concrete-Round-Column", "Ø500mm",
         {"Diameter": 500, "Column Height": 3000}, "success"),

        ("column", "M_Concrete-Round-Column", "Ø600mm",
         {"Diameter": 600, "Column Height": 3000}, "success"),

        ("door",   "M_Single-Flush", "0900 x 2100mm",
         {"Width": 900, "Height": 2100}, "success"),

        ("door",   "M_Single-Flush", "0800 x 2100mm",
         {"Width": 800, "Height": 2100}, "success"),

        ("door",   "M_Double-Flush", "1800 x 2100mm",
         {"Width": 1800, "Height": 2100}, "success"),

        ("window", "M_Fixed", "1200 x 1500mm",
         {"Width": 1200, "Height": 1500}, "success"),

        # Known failure patterns → correction applied (from v1 Revit error logs)
        ("wall",   "", "", {}, "failure"),   # placeholder — errors are runtime-populated

        ("column", "M_Concrete-Rectangular-Column", "",
         {"b": 100, "d": 100}, "failure"),   # below Revit extrusion floor
    ]

    for (el_type, family, type_name, params, outcome) in patterns:
        result = memory_io.save_pattern(
            element_type      = el_type,
            family_name       = family,
            type_name         = type_name,
            parameters        = params,
            outcome           = outcome,
            error_message     = "" if outcome == "success" else "Column section < 200 mm extrusion error",
            correction_applied = "" if outcome == "success" else "Clamp b and d to 200 mm minimum",
        )
        status = "OK" if outcome == "success" else "FAIL-PATTERN"
        print(f"  [{status}] {el_type:8s} | {family or 'N/A'} | {type_name or 'N/A'} … {result}")

    print(f"Translator memory seeded: {len(patterns)} patterns.\n")


if __name__ == "__main__":
    seed_validation_memory()
    seed_translator_memory()

    print("─" * 60)
    print("Memory bootstrap complete.")
    print("  validation/memory.sqlite — DfMA conflict resolutions")
    print("  translator/memory.sqlite — Revit API success patterns")
    print("\nRun the pipeline:")
    print("  python controller.py <floor_plan.pdf> [--context project_context.json]")
