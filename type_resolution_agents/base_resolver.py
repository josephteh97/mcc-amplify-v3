"""
base_resolver.py — Abstract base class for per-element type resolution.

Answers the "what type?" question after YOLO answers "where is it?".

Resolution cascade (highest → lowest trust):
  1. OCR scan      — text tag found adjacent to element (e.g. "C1", "W2")
  2. Geometric     — measure visual geometry of the cropped bbox region
  3. Cluster prop  — group by shape signature; propagate labels within group
  4. Spatial k-NN  — inherit label from k nearest labelled neighbours
  5. Synthetic     — assign GROUP_A/B/C to fully unlabelled clusters

Each resolved detection gains:
    "type_mark":            str | None  — e.g. "C1", "GROUP_A"
    "resolution_method":    str         — audit trail (see constants below)
    "resolution_confidence": float      — 0.0–1.0

Subclasses must implement:
    _extract_tag(text)              — regex for element-specific tag patterns
    _geometric_fingerprint(det, img, scale) → dict  — shape + estimated mm dims
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

# ── Optional OCR (pytesseract + tesseract binary) ─────────────────────────────
try:
    import pytesseract as _tess
    _OCR_AVAILABLE = True
except ImportError:
    _tess = None
    _OCR_AVAILABLE = False

# ── Resolution method audit strings ───────────────────────────────────────────
RES_OCR_DIRECT   = "ocr:direct"       # tag found adjacent to this element
RES_OCR_SCHEDULE = "ocr:schedule"     # tag found + schedule table looked up
RES_GEOMETRIC    = "geometric"        # shape derived from visual measurement
RES_CLUSTER      = "propagated:cluster"   # inherited from cluster majority
RES_SPATIAL      = "propagated:spatial"   # inherited from k nearest neighbours
RES_SYNTHETIC    = "synthetic"        # no label; assigned GROUP_A/B/C

# ── Tuning constants ──────────────────────────────────────────────────────────
_OCR_OUTER_RADIUS = 80   # px — search ring outer radius around bbox
_OCR_INNER_MARGIN = 4    # px — inner gap (exclude the element symbol itself)
_CLUSTER_TOL      = 0.20 # ±20% size tolerance for same-cluster grouping
_PROP_K           = 4    # k nearest neighbours for spatial propagation
_PROP_MIN_VOTE    = 0.60 # majority fraction required to propagate a label


# ══════════════════════════════════════════════════════════════════════════════
# BaseTypeResolver
# ══════════════════════════════════════════════════════════════════════════════

class BaseTypeResolver(ABC):
    """
    Subclass this for each element type (column, wall, beam …).

    Args:
        ocr_enabled:    Whether to attempt OCR tag scan (default True).
                        Disable for faster runs when drawings have no tags.
        scale_mm_per_px: Override the scale estimate (mm per pixel).
                         If None, estimated from project_context at resolve() time.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        scale_mm_per_px: float | None = None,
    ) -> None:
        self._ocr_enabled    = ocr_enabled and _OCR_AVAILABLE
        self._scale_mm_per_px = scale_mm_per_px

    # ── Public entry point ────────────────────────────────────────────────────

    def resolve(
        self,
        detections:      list[dict],
        page_image:      Image.Image,
        grid_result:     dict,
        project_context: dict | None = None,
    ) -> list[dict]:
        """
        Enrich detections in-place with type_mark, resolution_method,
        resolution_confidence.  Returns the same list (mutated).
        """
        if not detections:
            return detections

        ctx   = project_context or {}
        scale = self._scale_mm_per_px or _estimate_scale(detections, ctx)

        # ── Step 1: OCR scan ──────────────────────────────────────────────────
        for det in detections:
            tag = self._ocr_scan(det, page_image) if self._ocr_enabled else None
            det["type_mark"]             = tag
            det["resolution_method"]     = RES_OCR_DIRECT if tag else None
            det["resolution_confidence"] = 0.95 if tag else 0.0

        # ── Step 2: geometric fingerprint (subclass-specific) ─────────────────
        for det in detections:
            fp = self._geometric_fingerprint(det, page_image, scale)
            det.update(fp)   # shape, est_width_mm, est_depth_mm, etc.

        # ── Step 3: cluster by shape signature ───────────────────────────────
        clusters = _build_shape_clusters(detections)

        # ── Step 4: propagate labels within clusters ──────────────────────────
        _propagate_cluster_labels(detections, clusters)

        # ── Step 5: spatial k-NN propagation for remaining unresolved ─────────
        _spatial_propagate(detections)

        # ── Step 6: assign synthetic group names to fully unlabelled clusters ─
        _assign_synthetic_types(detections, clusters)

        return detections

    # ── Abstract hooks (subclasses must implement) ────────────────────────────

    @abstractmethod
    def _extract_tag(self, text: str) -> str | None:
        """
        Extract the element type mark from raw OCR text.
        Return None if no valid tag is found.
        e.g. "C1" from "C1 800x800" → "C1"
        """

    @abstractmethod
    def _geometric_fingerprint(
        self,
        det:           dict,
        page_image:    Image.Image,
        scale_mm_per_px: float | None,
    ) -> dict:
        """
        Analyse the cropped bbox region and return geometry fields.
        Minimum keys to return: {"shape": str, "est_width_mm": float|None,
                                 "est_depth_mm": float|None}
        """

    # ── OCR scan (shared across all element types) ────────────────────────────

    def _ocr_scan(self, det: dict, page_image: Image.Image) -> str | None:
        """
        Crop a donut-shaped region AROUND (not on) the bbox and run OCR.
        Searching outside the symbol avoids reading hatching/fill as text.
        """
        if not _OCR_AVAILABLE:
            return None

        x1, y1, x2, y2 = [int(v) for v in det["bbox_page"]]
        W, H = page_image.size

        # Expand outward by OCR_OUTER_RADIUS, clamp to image bounds
        rx1 = max(0, x1 - _OCR_OUTER_RADIUS)
        ry1 = max(0, y1 - _OCR_OUTER_RADIUS)
        rx2 = min(W, x2 + _OCR_OUTER_RADIUS)
        ry2 = min(H, y2 + _OCR_OUTER_RADIUS)
        region = page_image.crop((rx1, ry1, rx2, ry2))

        # Blank out the inner symbol area to avoid false reads from hatching
        inner_x1 = x1 - rx1 - _OCR_INNER_MARGIN
        inner_y1 = y1 - ry1 - _OCR_INNER_MARGIN
        inner_x2 = x2 - rx1 + _OCR_INNER_MARGIN
        inner_y2 = y2 - ry1 + _OCR_INNER_MARGIN
        region = region.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(region)
        draw.rectangle([inner_x1, inner_y1, inner_x2, inner_y2], fill="white")

        # Scale up (OCR accuracy improves significantly above 150 DPI equivalent)
        region = region.resize(
            (region.width * 2, region.height * 2),
            resample=Image.LANCZOS,
        )

        try:
            text = _tess.image_to_string(
                region.convert("L"),
                config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/",
            )
            return self._extract_tag(text.strip())
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Shared propagation algorithms (module-level, reused by all subclasses)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_scale(detections: list[dict], ctx: dict) -> float | None:
    """
    Estimate mm-per-pixel scale from median column bbox size + project context.
    Returns None if insufficient data.
    """
    default_mm = ctx.get("default_column_size_mm", 800)
    widths = [
        d["bbox_page"][2] - d["bbox_page"][0]
        for d in detections
        if d.get("bbox_page") and d["bbox_page"][0] is not None
    ]
    if not widths:
        return None
    median_px = sorted(widths)[len(widths) // 2]
    return default_mm / median_px if median_px > 0 else None


def _build_shape_clusters(detections: list[dict]) -> dict[int, list[int]]:
    """
    Group detections by similar bbox size (±CLUSTER_TOL).
    Returns {cluster_id: [detection_index, ...]}.

    Algorithm: greedy — each detection joins the first existing cluster whose
    representative size is within tolerance; otherwise starts a new cluster.
    """
    cluster_reps: list[tuple[float, float]] = []   # (w_px, h_px) per cluster
    cluster_map:  list[int]                 = []   # cluster_id per detection

    for det in detections:
        if not det.get("bbox_page") or det["bbox_page"][0] is None:
            cluster_map.append(-1)
            continue
        w = det["bbox_page"][2] - det["bbox_page"][0]
        h = det["bbox_page"][3] - det["bbox_page"][1]

        assigned = -1
        for cid, (rw, rh) in enumerate(cluster_reps):
            if (abs(w - rw) / max(rw, 1e-6) <= _CLUSTER_TOL and
                    abs(h - rh) / max(rh, 1e-6) <= _CLUSTER_TOL):
                assigned = cid
                break
        if assigned == -1:
            assigned = len(cluster_reps)
            cluster_reps.append((w, h))
        cluster_map.append(assigned)

    # Attach cluster_id to each detection and build reverse index
    clusters: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_map):
        detections[idx]["_cluster_id"] = cid
        clusters.setdefault(cid, []).append(idx)

    return clusters


def _propagate_cluster_labels(
    detections: list[dict],
    clusters:   dict[int, list[int]],
) -> None:
    """
    For each cluster: if ≥PROP_MIN_VOTE of labelled members share a type_mark,
    assign that mark to all unresolved members in the cluster.
    Only overwrites detections where type_mark is None.
    """
    for cid, indices in clusters.items():
        if cid == -1:
            continue
        labelled = [
            detections[i]["type_mark"]
            for i in indices
            if detections[i].get("type_mark")
        ]
        if not labelled:
            continue

        # Find the majority label
        counts: dict[str, int] = {}
        for lbl in labelled:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority_lbl, majority_cnt = max(counts.items(), key=lambda kv: kv[1])

        if majority_cnt / len(labelled) < _PROP_MIN_VOTE:
            continue   # disagreement — do not propagate

        confidence = round(majority_cnt / len(indices), 2)
        for i in indices:
            if detections[i].get("type_mark") is None:
                detections[i]["type_mark"]             = majority_lbl
                detections[i]["resolution_method"]     = RES_CLUSTER
                detections[i]["resolution_confidence"] = confidence


def _spatial_propagate(detections: list[dict]) -> None:
    """
    For each still-unresolved detection: find k nearest neighbours by centroid
    distance. If ≥PROP_MIN_VOTE share a type_mark AND the same _cluster_id,
    inherit that label.

    Using same-cluster guard prevents propagation across different element
    types (e.g. a small column near a large wall) even if they are spatially
    close.
    """
    # Build centroid array for all detections
    centroids: list[tuple[float, float] | None] = []
    for det in detections:
        bb = det.get("bbox_page")
        if bb and bb[0] is not None:
            centroids.append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
        else:
            centroids.append(None)

    for i, det in enumerate(detections):
        if det.get("type_mark") is not None:
            continue
        if centroids[i] is None:
            continue

        cx, cy = centroids[i]
        cid = det.get("_cluster_id", -1)

        # Compute distance to all labelled neighbours in the same cluster
        neighbours: list[tuple[float, str]] = []
        for j, other in enumerate(detections):
            if i == j or other.get("type_mark") is None:
                continue
            if other.get("_cluster_id") != cid:
                continue
            if centroids[j] is None:
                continue
            ox, oy = centroids[j]
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            neighbours.append((dist, other["type_mark"]))

        if not neighbours:
            continue

        neighbours.sort(key=lambda x: x[0])
        k_nearest = [lbl for _, lbl in neighbours[:_PROP_K]]
        if not k_nearest:
            continue

        counts: dict[str, int] = {}
        for lbl in k_nearest:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority_lbl, majority_cnt = max(counts.items(), key=lambda kv: kv[1])

        if majority_cnt / len(k_nearest) >= _PROP_MIN_VOTE:
            det["type_mark"]             = majority_lbl
            det["resolution_method"]     = RES_SPATIAL
            det["resolution_confidence"] = round(majority_cnt / len(k_nearest), 2)


def _assign_synthetic_types(
    detections: list[dict],
    clusters:   dict[int, list[int]],
) -> None:
    """
    Any detection still unresolved gets a synthetic group name (GROUP_A, GROUP_B …).
    All members of the same cluster share the same synthetic name — they look the
    same visually even if no label was found.
    """
    _LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    synthetic_counter = 0
    cluster_names: dict[int, str] = {}

    for det in detections:
        if det.get("type_mark") is not None:
            continue
        cid = det.get("_cluster_id", -1)
        if cid not in cluster_names:
            letter = _LETTERS[synthetic_counter % 26]
            cluster_names[cid] = f"GROUP_{letter}"
            synthetic_counter += 1
        det["type_mark"]             = cluster_names[cid]
        det["resolution_method"]     = RES_SYNTHETIC
        det["resolution_confidence"] = 0.30
