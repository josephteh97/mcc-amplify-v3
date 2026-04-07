"""
validation/grid_snap.py — Snap element coordinates to the nearest grid intersection.

snap_to_grid_mm() works in world-mm space and is robust to scale estimation
errors up to half a bay width — use this as the primary snap.
"""
from __future__ import annotations


def snap_to_grid_mm(
    approx_mm: tuple[float, float],
    x_mm: list[float],
    y_mm: list[float],
    v_labels: list[str],
    h_labels: list[str],
    tolerance_mm: float = 4200.0,
) -> dict:
    """
    Snap approximate world-mm coordinates to the nearest grid intersection.

    Works entirely in mm space so it is immune to scale estimation errors as long
    as the error is less than tolerance_mm (default half of an 8 400 mm bay).

    Args:
        approx_mm:    (x_mm, y_mm) coarse world position (from px_to_mm fallback).
        x_mm:         Sorted list of grid x positions in mm  [0, 8400, 16800, ...].
        y_mm:         Sorted list of grid y positions in mm  [0, 8400, ...].
        v_labels:     Grid labels matching x_mm  ["1", "2", ...].
        h_labels:     Grid labels matching y_mm  ["A", "B", ...].
        tolerance_mm: Maximum mm distance to snap (default 4 200 mm = half an 8 400 mm bay).

    Returns:
        {"ok": bool, "x_mm": float, "y_mm": float, "grid_label": (v_label, h_label) | None}
        ok=False when no grid positions are provided or distance exceeds tolerance.
    """
    if not x_mm or not y_mm:
        return {"ok": False, "x_mm": 0.0, "y_mm": 0.0, "grid_label": None}

    ax, ay = approx_mm
    vi = min(range(len(x_mm)), key=lambda i: abs(ax - x_mm[i]))
    hi = min(range(len(y_mm)), key=lambda i: abs(ay - y_mm[i]))

    if abs(ax - x_mm[vi]) > tolerance_mm or abs(ay - y_mm[hi]) > tolerance_mm:
        return {"ok": False, "x_mm": 0.0, "y_mm": 0.0, "grid_label": None}

    return {
        "ok":         True,
        "x_mm":       x_mm[vi],
        "y_mm":       y_mm[hi],
        "grid_label": (v_labels[vi], h_labels[hi]),
    }
