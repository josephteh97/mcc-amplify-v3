"""
overlap_conflict.py — Flag detections that overlap across element types.

A column bbox that significantly overlaps a wall bbox is suspicious — YOLO
likely misclassified a wall junction or door jamb as a column.

Both overlapping detections are flagged (both may be wrong, or one may be
correct and the other a false positive — the user decides).

Only cross-type overlap is checked here. Within-type duplicates (two column
detections on the same spot) are handled by DfMA rule D1 in the Validation
Agent.
"""

from __future__ import annotations

# Minimum IoU to flag a cross-type overlap as a conflict.
# 0.3 catches partial overlaps (e.g. column centre inside a wall bbox).
_CONFLICT_IOU_THRESHOLD = 0.30


def check(all_detections: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    Flag cross-type overlapping detections with "_overlap_conflict".

    Args:
        all_detections: {"column": [...], "wall": [...], "beam": [...], ...}

    Returns the same dict with "_overlap_conflict" fields added in-place.
    """
    element_types = list(all_detections.keys())

    for i, type_a in enumerate(element_types):
        for type_b in element_types[i + 1:]:
            dets_a = all_detections[type_a]
            dets_b = all_detections[type_b]

            for det_a in dets_a:
                for det_b in dets_b:
                    iou = _iou(det_a.get("bbox_page"), det_b.get("bbox_page"))
                    if iou >= _CONFLICT_IOU_THRESHOLD:
                        _flag(det_a, type_b, det_b.get("id"), iou)
                        _flag(det_b, type_a, det_a.get("id"), iou)

    return all_detections


def _flag(det: dict, conflict_type: str, conflict_id: int | None, iou: float) -> None:
    conflicts = det.setdefault("_overlap_conflict", [])
    conflicts.append({
        "conflicts_with_type": conflict_type,
        "conflicts_with_id":   conflict_id,
        "iou":                 round(iou, 3),
    })


def _iou(a: list | None, b: list | None) -> float:
    """Intersection-over-Union of two [x1, y1, x2, y2] bboxes."""
    if not a or not b or None in a or None in b:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
