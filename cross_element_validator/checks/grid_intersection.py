"""
grid_intersection.py — Plausibility check: columns should be near grid intersections.

Structural columns in plan drawings always sit at or very close to a grid
intersection. A "column" detection far from every intersection is suspicious —
it may be a door symbol, annotation box, or room label.

Penalty is proportional to distance from the nearest intersection. The score
feeds into the overall plausibility; it does not hard-reject the detection.

Requires grid_result to contain pixel positions of grid lines:
    grid_result["_grid_lines_px"] = {
        "vertical":   [x_px, ...],   # x pixel coordinates
        "horizontal": [y_px, ...],   # y pixel coordinates
    }

If _grid_lines_px is absent (grid agent did not expose pixel positions),
this check is skipped gracefully with a neutral score.
"""

from __future__ import annotations

import math

# Maximum pixel distance from a grid intersection before penalising.
# At 300 DPI, 1:400 scale, a typical bay is ~550 px wide.
# Columns on the same grid line should be within 30 px of the line.
_MAX_NEAR_PX       = 40    # within this distance → plausibility 1.0
_MAX_PENALTY_PX    = 200   # beyond this → plausibility 0.0 (fully suspicious)


def check(
    detections:  list[dict],
    grid_result: dict,
    element_type: str = "column",
) -> list[dict]:
    """
    Attach "_grid_plausibility" (0.0–1.0) to each detection.
    Returns detections (mutated in-place).

    Only meaningful for columns — walls and beams span between intersections
    so their endpoints may be near but not their midpoints.
    """
    if element_type != "column":
        for det in detections:
            det["_grid_plausibility"] = 1.0
        return detections

    grid_lines = grid_result.get("_grid_lines_px", {})
    v_lines = grid_lines.get("vertical",   [])
    h_lines = grid_lines.get("horizontal", [])

    if not v_lines or not h_lines:
        # Grid pixel positions unavailable — skip with neutral score
        for det in detections:
            det["_grid_plausibility"] = 1.0
        return detections

    # Precompute all intersection points
    intersections = [(vx, hy) for vx in v_lines for hy in h_lines]

    for det in detections:
        bb = det.get("bbox_page")
        if not bb or bb[0] is None:
            det["_grid_plausibility"] = 1.0
            continue

        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2

        min_dist = min(
            math.hypot(cx - ix, cy - iy)
            for ix, iy in intersections
        )

        if min_dist <= _MAX_NEAR_PX:
            score = 1.0
        elif min_dist >= _MAX_PENALTY_PX:
            score = 0.0
        else:
            # Linear decay from 1.0 → 0.0
            score = 1.0 - (min_dist - _MAX_NEAR_PX) / (_MAX_PENALTY_PX - _MAX_NEAR_PX)

        det["_grid_plausibility"] = round(score, 3)
        det["_nearest_grid_dist_px"] = round(min_dist, 1)

    return detections
