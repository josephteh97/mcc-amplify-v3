"""
neighbourhood_consensus.py — Flag spatial outliers within an element group.

Columns appear in grids — never in true isolation. A single "column" detection
with no other columns within 3× the median inter-column spacing is almost
certainly a false positive (stray mark, door symbol, annotation box).

This check computes the median nearest-neighbour distance for the group, then
flags any detection whose nearest neighbour is > OUTLIER_FACTOR × median.
"""

from __future__ import annotations

import math

# A detection is an outlier if its nearest neighbour distance exceeds this
# multiple of the group's median nearest-neighbour distance.
_OUTLIER_FACTOR = 3.0

# Minimum group size to compute a stable median.
# Smaller groups skip the check (not enough context to judge).
_MIN_GROUP_SIZE = 4


def check(
    detections:   list[dict],
    element_type: str = "column",
) -> list[dict]:
    """
    Attach "_neighbour_plausibility" (0.0–1.0) to each detection.
    Returns detections (mutated in-place).

    Only applied to columns. Walls and beams have topology (they connect to
    other elements) rather than a grid pattern, so outlier-by-isolation
    is not the right heuristic for them.
    """
    if element_type != "column" or len(detections) < _MIN_GROUP_SIZE:
        for det in detections:
            det["_neighbour_plausibility"] = 1.0
        return detections

    centroids = _centroids(detections)

    # Nearest-neighbour distance for every detection
    nn_dists: list[float] = []
    for i, (cx, cy) in enumerate(centroids):
        if cx is None:
            nn_dists.append(0.0)
            continue
        dists = [
            math.hypot(cx - ox, cy - oy)
            for j, (ox, oy) in enumerate(centroids)
            if i != j and ox is not None
        ]
        nn_dists.append(min(dists) if dists else 0.0)

    # Median of non-zero distances
    valid = sorted(d for d in nn_dists if d > 0)
    if not valid:
        for det in detections:
            det["_neighbour_plausibility"] = 1.0
        return detections
    median_nn = valid[len(valid) // 2]

    threshold = _OUTLIER_FACTOR * median_nn

    for det, nn in zip(detections, nn_dists):
        if nn == 0:
            det["_neighbour_plausibility"] = 1.0
            continue
        if nn <= threshold:
            det["_neighbour_plausibility"] = 1.0
        else:
            # Smoothly decay: at 2× threshold → 0.0
            score = max(0.0, 1.0 - (nn - threshold) / threshold)
            det["_neighbour_plausibility"] = round(score, 3)
        det["_nn_dist_px"] = round(nn, 1)

    return detections


def _centroids(
    detections: list[dict],
) -> list[tuple[float | None, float | None]]:
    result = []
    for det in detections:
        bb = det.get("bbox_page")
        if bb and bb[0] is not None:
            result.append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
        else:
            result.append((None, None))
    return result
