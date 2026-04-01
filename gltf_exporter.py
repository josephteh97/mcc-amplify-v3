"""
gltf_exporter.py — GLB export from Revit Transaction JSON geometry
===================================================================
Converts the geometry dict produced by translator/tools.py
(revit_schema_mapper output) into a binary GLB file for web 3D preview.

Mesh naming mirrors the element arrays so the frontend Viewer can map
a clicked mesh back to its recipe element:
    wall_0, wall_1 …   column_0 …   door_0 …   window_0 …   floor_0 …
"""

from __future__ import annotations

import numpy as np
import trimesh
from pathlib import Path


def export(geometry: dict, output_path: str) -> str:
    """
    Build a GLB from the geometry dict and write it to output_path.

    Args:
        geometry:    Dict with keys walls, columns, doors, windows, floors,
                     ceilings — as produced by revit_schema_mapper().
        output_path: Destination .glb file path (parent dir is created).

    Returns:
        The resolved output_path string.
    """
    # (key, builder, RGBA) — key[:-1] gives the singular mesh name: "walls" → "wall_0"
    _TYPES = [
        ("walls",    _wall_mesh,                         [200, 200, 200, 255]),
        ("columns",  _column_mesh,                       [150, 150, 150, 255]),
        ("doors",    lambda e: _opening_mesh(e, 100.0),  [139,  90,  43, 255]),
        ("windows",  lambda e: _opening_mesh(e,  50.0),  [135, 206, 235, 200]),
        ("floors",   _slab_mesh,                         [220, 210, 190, 255]),
        ("ceilings", _slab_mesh,                         [240, 240, 240, 220]),
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()

    for key, builder, color in _TYPES:
        for idx, element in enumerate(geometry.get(key, [])):
            m = builder(element)
            if m is not None:
                m.visual.face_colors = color
                scene.add_geometry(m, geom_name=f"{key[:-1]}_{idx}")

    if len(scene.geometry) == 0:
        # Placeholder so the viewer doesn't receive an empty file
        plane = trimesh.creation.box(extents=[1000, 1000, 1])
        plane.visual.face_colors = [180, 180, 180, 255]
        scene.add_geometry(plane)

    scene.export(output_path)
    print(
        f"  [gltf] exported {len(scene.geometry)} meshes → {output_path}"
    )
    return output_path


# ── Mesh builders ──────────────────────────────────────────────────────────────

def _wall_mesh(wall: dict):
    try:
        s, e  = wall["start_point"], wall["end_point"]
        dx, dy = e["x"] - s["x"], e["y"] - s["y"]
        length = float(np.sqrt(dx**2 + dy**2))
        if length < 1.0:
            return None
        angle     = float(np.arctan2(dy, dx))
        thickness = float(wall.get("thickness", 200))
        height    = float(wall.get("height", 2800))
        box = trimesh.creation.box(extents=[length, thickness, height])
        cx, cy = (s["x"] + e["x"]) / 2, (s["y"] + e["y"]) / 2
        T = trimesh.transformations.translation_matrix([cx, cy, height / 2])
        R = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        box.apply_transform(trimesh.transformations.concatenate_matrices(T, R))
        return box
    except Exception:
        return None


def _column_mesh(col: dict):
    try:
        loc    = col["location"]
        width  = float(col.get("width",  300))
        depth  = float(col.get("depth",  300))
        height = float(col.get("height", 2800))
        mesh = (
            trimesh.creation.cylinder(radius=width / 2, height=height)
            if col.get("shape") == "circular"
            else trimesh.creation.box(extents=[width, depth, height])
        )
        mesh.apply_transform(
            trimesh.transformations.translation_matrix([loc["x"], loc["y"], height / 2])
        )
        return mesh
    except Exception:
        return None


def _opening_mesh(opening: dict, depth: float = 100.0):
    try:
        loc    = opening["location"]
        width  = float(opening.get("width",  900))
        height = float(opening.get("height", 2100))
        z      = float(loc.get("z", 0))
        box    = trimesh.creation.box(extents=[width, depth, height])
        box.apply_transform(
            trimesh.transformations.translation_matrix([loc["x"], loc["y"], z + height / 2])
        )
        return box
    except Exception:
        return None


def _slab_mesh(slab: dict):
    try:
        pts = slab.get("boundary_points", [])
        if len(pts) < 3:
            return None
        xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
        w, d   = max(xs) - min(xs), max(ys) - min(ys)
        if w < 1.0 or d < 1.0:
            return None
        cx, cy    = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        thickness = float(slab.get("thickness", 200))
        elevation = float(slab.get("elevation", 0))
        mesh = trimesh.creation.box(extents=[w, d, thickness])
        mesh.apply_transform(
            trimesh.transformations.translation_matrix([cx, cy, elevation + thickness / 2])
        )
        return mesh
    except Exception:
        return None
