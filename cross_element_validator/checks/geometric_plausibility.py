"""
geometric_plausibility.py — Check bbox geometry matches expected element shape.

Each element type has a characteristic aspect ratio in plan view:
  • Column  — roughly square (min/max side ≥ 0.55)
  • Wall    — elongated (max/min side ≥ 3.0)
  • Beam    — elongated, similar to wall

A "column" with aspect ratio 0.15 is almost certainly a misclassified wall
section. This check computes a plausibility score based on how well the
detected bbox geometry matches the expected shape for its element type.
"""

from __future__ import annotations

# Expected aspect ratio ranges per element type.
# (min_squareness, max_squareness) where squareness = min(w,h)/max(w,h)
# 1.0 = perfect square, 0.0 = infinitely thin line
_EXPECTED: dict[str, tuple[float, float]] = {
    "column": (0.55, 1.00),   # mostly square
    "wall":   (0.00, 0.35),   # elongated
    "beam":   (0.00, 0.40),   # elongated
}
_DEFAULT_RANGE = (0.0, 1.0)   # unknown type — neutral


def check(
    detections:   list[dict],
    element_type: str,
) -> list[dict]:
    """
    Attach "_geom_plausibility" (0.0–1.0) to each detection.
    Returns detections (mutated in-place).
    """
    lo, hi = _EXPECTED.get(element_type, _DEFAULT_RANGE)

    for det in detections:
        bb = det.get("bbox_page")
        if not bb or bb[0] is None:
            det["_geom_plausibility"] = 1.0
            continue

        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if w <= 0 or h <= 0:
            det["_geom_plausibility"] = 0.0
            continue

        squareness = min(w, h) / max(w, h)
        det["_squareness"] = round(squareness, 3)

        if lo <= squareness <= hi:
            det["_geom_plausibility"] = 1.0
        else:
            # Distance outside the expected range, normalised
            dist = min(abs(squareness - lo), abs(squareness - hi))
            # Penalise: 0.1 outside range → score 0.5; 0.3+ outside → score 0.0
            score = max(0.0, 1.0 - dist / 0.30)
            det["_geom_plausibility"] = round(score, 3)

    return detections
