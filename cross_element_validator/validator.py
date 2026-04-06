"""
validator.py — Cross-element validator.

Runs after ALL YOLO detection agents complete, when the full multi-element
picture is available. Applies four independent checks and quarantines
suspicious detections without blocking the pipeline.

Four checks (each attaches a score field to the detection dict):
  1. geometric_plausibility  — bbox shape matches expected element geometry
  2. overlap_conflict        — cross-type bbox overlap (column inside a wall)
  3. grid_intersection       — columns should sit near a grid intersection
  4. neighbourhood_consensus — column outliers (no nearby columns in a grid)

Plausibility score = weighted average of the four check scores.
Detections below QUARANTINE_THRESHOLD are quarantined (soft-flagged).

Usage:
    from cross_element_validator.validator import CrossElementValidator

    validator = CrossElementValidator()
    validated, quarantined = validator.validate(
        all_detections={"column": [...], "wall": [...]},
        grid_result=grid_result,
    )
    # validated  — same structure, detections annotated with plausibility scores
    # quarantined — QuarantineManager, call .to_edit_panel_payload()
"""

from __future__ import annotations

from .quarantine import QuarantineManager
from .checks import (
    geometric_plausibility as _geom_check,
    overlap_conflict       as _overlap_check,
    grid_intersection      as _grid_check,
    neighbourhood_consensus as _nbr_check,
)

# ── Quarantine threshold ───────────────────────────────────────────────────────
# Detections with overall plausibility below this are quarantined.
# 0.40 means at least two checks must be healthy for the detection to pass.
_QUARANTINE_THRESHOLD = 0.40

# ── Check weights (must sum to 1.0) ───────────────────────────────────────────
_WEIGHTS = {
    "geom":     0.30,   # always applicable
    "overlap":  0.30,   # always applicable when multiple element types present
    "grid":     0.25,   # most useful for columns; neutral for others
    "neighbour":0.15,   # useful for columns in dense drawings
}


class CrossElementValidator:
    """
    Runs all four cross-element checks and quarantines suspicious detections.
    """

    def validate(
        self,
        all_detections: dict[str, list[dict]],
        grid_result:    dict,
    ) -> tuple[dict[str, list[dict]], QuarantineManager]:
        """
        Args:
            all_detections: {"column": [...], "wall": [...], ...}
            grid_result:    Output from grid detection agent.

        Returns:
            (all_detections, quarantine_manager)
            all_detections is mutated in-place with plausibility scores.
        """
        qm = QuarantineManager()

        # ── Check 1: geometric plausibility (per element type) ────────────────
        for element_type, dets in all_detections.items():
            _geom_check.check(dets, element_type)

        # ── Check 2: cross-type overlap conflict (needs all types together) ───
        _overlap_check.check(all_detections)

        # ── Check 3 & 4: per-element spatial checks ────────────────────────────
        for element_type, dets in all_detections.items():
            _grid_check.check(dets, grid_result, element_type)
            _nbr_check.check(dets, element_type)

        # ── Compute overall plausibility and quarantine low-scoring detections ─
        for element_type, dets in all_detections.items():
            for det in dets:
                score, reasons, checks_failed = _score(det)
                det["_plausibility"] = round(score, 3)

                if score < _QUARANTINE_THRESHOLD:
                    qm.add(
                        detection     = det,
                        element_type  = element_type,
                        reasons       = reasons,
                        checks_failed = checks_failed,
                        plausibility  = score,
                        suggested_action = _suggest(element_type, reasons),
                    )

        if len(qm) > 0:
            print(f"  [CrossElementValidator] {qm.summary()}")
        else:
            print("  [CrossElementValidator] All detections passed.")

        return all_detections, qm


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _score(det: dict) -> tuple[float, list[str], list[str]]:
    """
    Compute weighted plausibility score, collect failure reasons.
    Returns (score, reasons, checks_failed).
    """
    geom     = det.get("_geom_plausibility",      1.0)
    grid     = det.get("_grid_plausibility",       1.0)
    neighbour = det.get("_neighbour_plausibility", 1.0)
    # Overlap: 0.0 if any conflict, else 1.0
    overlap  = 0.0 if det.get("_overlap_conflict") else 1.0

    score = (
        _WEIGHTS["geom"]      * geom     +
        _WEIGHTS["overlap"]   * overlap  +
        _WEIGHTS["grid"]      * grid     +
        _WEIGHTS["neighbour"] * neighbour
    )

    reasons: list[str]       = []
    checks_failed: list[str] = []

    if geom < 0.5:
        sq = det.get("_squareness", "?")
        reasons.append(f"bbox squareness {sq} unexpected for this element type")
        checks_failed.append("geometric_plausibility")

    if overlap < 1.0:
        for conflict in det.get("_overlap_conflict", []):
            reasons.append(
                f"overlaps {conflict['conflicts_with_type']} id={conflict['conflicts_with_id']} "
                f"(IoU={conflict['iou']})"
            )
        checks_failed.append("overlap_conflict")

    if grid < 0.5:
        dist = det.get("_nearest_grid_dist_px", "?")
        reasons.append(f"far from nearest grid intersection ({dist} px)")
        checks_failed.append("grid_intersection")

    if neighbour < 0.5:
        nn = det.get("_nn_dist_px", "?")
        reasons.append(f"spatial outlier — nearest neighbour is {nn} px away")
        checks_failed.append("neighbourhood_consensus")

    return score, reasons, checks_failed


def _suggest(element_type: str, reasons: list[str]) -> str:
    if any("overlaps" in r for r in reasons):
        return (
            f"This {element_type} overlaps another element type. "
            "Check whether YOLO misclassified an adjacent element."
        )
    if any("grid intersection" in r for r in reasons):
        return (
            f"This {element_type} is far from any grid intersection. "
            "Verify it is a real structural column, not an annotation or symbol."
        )
    if any("squareness" in r for r in reasons):
        return (
            f"The bbox shape is inconsistent with a {element_type}. "
            "Check whether this is a wall section or door jamb."
        )
    if any("outlier" in r for r in reasons):
        return (
            f"Isolated {element_type} with no nearby peers. "
            "Confirm this is not a stray annotation or false positive."
        )
    return f"Review this {element_type} detection against the original drawing."
