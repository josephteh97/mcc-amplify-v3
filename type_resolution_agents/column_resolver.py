"""
column_resolver.py — Type resolver for structural columns.

Extends BaseTypeResolver with column-specific strategies:
  • OCR tag patterns: C1, RC1, SC1, PC1 (Singapore standard notation)
  • Geometric fingerprint: circular detection (Hough) + aspect ratio
  • Estimated mm dimensions from bbox size × scale
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from .base_resolver import BaseTypeResolver, RES_GEOMETRIC

# ── Optional OpenCV for geometric analysis ────────────────────────────────────
try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False

# ── Canonical shape values ────────────────────────────────────────────────────
SHAPE_ROUND     = "round"
SHAPE_SQUARE    = "square"
SHAPE_RECTANGLE = "rectangle"

# ── Column tag regex (Singapore structural drawings) ─────────────────────────
# Matches: C1, C12, RC1, SC2, PC3, BC1 — word-boundary anchored
_TAG_RE = re.compile(
    r"\b([RSPBF]?C\d{1,3})\b",
    re.IGNORECASE,
)

# ── Circular detection thresholds ─────────────────────────────────────────────
# If the detected region has a clear circular boundary it is a round column.
# Hough circles require the inscribed circle to fill ≥70% of the bbox area.
_CIRCLE_FILL_MIN  = 0.70   # inscribed circle area / bbox area
_SQUARENESS_RECT  = 0.65   # min(w,h)/max(w,h) below this → rectangle (not square)


# ══════════════════════════════════════════════════════════════════════════════
# ColumnTypeResolver
# ══════════════════════════════════════════════════════════════════════════════

class ColumnTypeResolver(BaseTypeResolver):
    """
    Per-element type resolver for structural columns.

    Args:
        ocr_enabled:     Scan for type marks (C1, RC1 …) near each column.
        scale_mm_per_px: Override scale. If None, estimated from median bbox size.
    """

    # ── OCR tag extraction ────────────────────────────────────────────────────

    def _extract_tag(self, text: str) -> str | None:
        """Return first column type mark found in OCR text, or None."""
        m = _TAG_RE.search(text.upper())
        return m.group(1).upper() if m else None

    # ── Geometric fingerprint ─────────────────────────────────────────────────

    def _geometric_fingerprint(
        self,
        det:             dict,
        page_image:      Image.Image,
        scale_mm_per_px: float | None,
    ) -> dict:
        """
        Analyse the cropped column bbox to determine:
          • shape        — "square" | "rectangle" | "round"
          • est_width_mm — estimated section width in mm
          • est_depth_mm — estimated section depth in mm
          • is_circular  — bool

        Sets resolution_method to RES_GEOMETRIC if no OCR tag was found.
        """
        x1, y1, x2, y2 = [int(v) for v in det["bbox_page"]]
        w_px = max(x2 - x1, 1)
        h_px = max(y2 - y1, 1)

        # ── Shape determination ───────────────────────────────────────────────
        is_circular = _detect_circular(page_image, x1, y1, x2, y2)

        if is_circular:
            shape = SHAPE_ROUND
        elif min(w_px, h_px) / max(w_px, h_px) >= _SQUARENESS_RECT:
            shape = SHAPE_SQUARE
        else:
            shape = SHAPE_RECTANGLE

        # ── Dimension estimate ────────────────────────────────────────────────
        est_width_mm  = round(w_px * scale_mm_per_px, 0) if scale_mm_per_px else None
        est_depth_mm  = round(h_px * scale_mm_per_px, 0) if scale_mm_per_px else None
        est_diam_mm   = round(((w_px + h_px) / 2) * scale_mm_per_px, 0) if (scale_mm_per_px and is_circular) else None

        result = {
            "shape":         shape,
            "is_circular":   is_circular,
            "est_width_mm":  est_width_mm,
            "est_depth_mm":  est_depth_mm,
            "est_diam_mm":   est_diam_mm,
        }

        # Only overwrite resolution fields if OCR found nothing
        if det.get("type_mark") is None:
            result["resolution_method"]     = RES_GEOMETRIC
            result["resolution_confidence"] = 0.65

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Geometric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_circular(
    img: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
) -> bool:
    """
    Return True if the cropped region appears to contain a circular column.

    Strategy:
      1. OpenCV Hough circle transform (if cv2 available) — most reliable
      2. Fallback: compare filled-circle area to bbox area (pure numpy/PIL)
    """
    # Pad slightly to give Hough room to find the boundary arc
    W, H = img.size
    pad  = max(4, (x2 - x1) // 4)
    cx1  = max(0, x1 - pad)
    cy1  = max(0, y1 - pad)
    cx2  = min(W, x2 + pad)
    cy2  = min(H, y2 + pad)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("L")

    if _CV2_AVAILABLE:
        return _hough_circle(np.array(crop), x2 - x1, y2 - y1)
    return _fill_ratio_circle(np.array(crop))


def _hough_circle(gray: np.ndarray, w_px: int, h_px: int) -> bool:
    """Detect circle using HoughCircles. Returns True if a strong circle found."""
    blurred = _cv2.GaussianBlur(gray, (5, 5), 0)
    min_r   = int(min(w_px, h_px) * 0.35)
    max_r   = int(max(w_px, h_px) * 0.65)
    circles = _cv2.HoughCircles(
        blurred,
        _cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(w_px, h_px),
        param1=50,
        param2=25,
        minRadius=max(min_r, 4),
        maxRadius=max(max_r, 8),
    )
    return circles is not None and len(circles[0]) > 0


def _fill_ratio_circle(gray: np.ndarray) -> bool:
    """
    Fallback without cv2: threshold the region, find the largest contiguous
    dark blob, compare its area to the inscribed circle area.
    If the blob is roughly circular (area ≈ π r²) → column is round.
    """
    threshold = 128
    binary = (gray < threshold).astype(np.uint8)

    # Simple blob area vs bbox area heuristic
    dark_pixels = binary.sum()
    h, w        = binary.shape
    bbox_area   = w * h
    circle_area = np.pi * (min(w, h) / 2) ** 2

    if bbox_area == 0 or circle_area == 0:
        return False

    # Dark pixels should fill roughly the inscribed circle area for a round col
    fill_ratio = dark_pixels / circle_area
    return 0.60 <= fill_ratio <= 1.40  # within ±40% of expected circle fill
