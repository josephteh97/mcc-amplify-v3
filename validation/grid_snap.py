"""
validation/grid_snap.py — Snap element pixel centers to the nearest grid intersection.

Used by coordinate_transformer() to place columns at exact grid-intersection
mm coordinates rather than scaled pixel positions.
"""
from __future__ import annotations


def snap_to_grid_intersection(
    center_px: tuple[float, float],
    v_lines_px: list[dict],
    h_lines_px: list[dict],
    v_world: dict[str, float],
    h_world: dict[str, float],
    tolerance_px: float = 80.0,
) -> dict:
    """
    Snap (cx, cy) to the nearest grid intersection and return its mm coordinates.

    Args:
        center_px:    Detected element center in pixels (cx, cy).
        v_lines_px:   [{"label": "1", "x_px": 0.0}, ...]  — vertical grid line positions.
        h_lines_px:   [{"label": "A", "y_px": 0.0}, ...]  — horizontal grid line positions.
        v_world:      {"1": 0.0, "2": 7500.0, ...}  — label → x_mm (precomputed by caller).
        h_world:      {"A": 0.0, "B": 5000.0, ...}  — label → y_mm (precomputed by caller).
        tolerance_px: Maximum pixel distance from a grid line to snap (default 80 px).

    Returns:
        {"ok": bool, "x_mm": float, "y_mm": float, "grid_label": (v_label, h_label) | None}
        ok=False when: no grid lines present; center is outside tolerance on either axis;
        or a matched label is absent from v_world/h_world.
    """
    if not v_lines_px or not h_lines_px:
        return {"ok": False, "x_mm": 0.0, "y_mm": 0.0, "grid_label": None}

    cx, cy = center_px
    best_v = min(v_lines_px, key=lambda g: abs(cx - g["x_px"]))
    best_h = min(h_lines_px, key=lambda g: abs(cy - g["y_px"]))

    if abs(cx - best_v["x_px"]) > tolerance_px or abs(cy - best_h["y_px"]) > tolerance_px:
        return {"ok": False, "x_mm": 0.0, "y_mm": 0.0, "grid_label": None}

    v_lbl, h_lbl = best_v["label"], best_h["label"]
    if v_lbl not in v_world or h_lbl not in h_world:
        return {"ok": False, "x_mm": 0.0, "y_mm": 0.0, "grid_label": None}

    return {
        "ok":         True,
        "x_mm":       v_world[v_lbl],
        "y_mm":       h_world[h_lbl],
        "grid_label": (v_lbl, h_lbl),
    }
