#!/usr/bin/env python3
"""
grid_detector.py — End-to-end structural grid detector for floor plan PDFs.

Pipeline
--------
1. Render PDF page → high-res RGB image
2. Detect long horizontal & vertical lines (the structural grid)
   • Morphological dilation reconnects dashed grid lines before detection
3. Cluster nearby lines → unique H and V grid positions
4. Find every grid intersection (candidate column location)
5. At each intersection analyse a local patch:
   • Blob density, shape circularity, aspect ratio → classify column type
6. Return detections in the same dict format as agent.py

Usage
-----
    python grid_detector.py --pdf path/to/plan.pdf --page 0
    python grid_detector.py --pdf path/to/plan.pdf --page 0 --debug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── Constants ──────────────────────────────────────────────────────────────────

RENDER_DPI       = 300          # 300 DPI gives enough resolution for dashed grid lines
LINE_PROJ_THRESH = 0.20         # projection must be ≥ 20 % of max to count as a grid line
LINE_CLUSTER_GAP = 40           # px at 300 DPI: lines within this gap → same grid line
CROSS_RADIUS     = 28           # px: search patch half-size at each intersection
DENSITY_THRESH   = 0.05         # min dark-pixel fraction to call it a column
CIRCULARITY_MIN  = 0.60         # contour circularity → round column

SHAPE_COLOURS = {               # for debug overlay
    "round":       (0, 200, 80),
    "square":      (0, 120, 255),
    "rectangle":   (255, 165, 0),
    "i_beam":      (200, 0, 200),
    "unknown":     (160, 160, 160),
}

# ── PDF → image ────────────────────────────────────────────────────────────────

def pdf_to_image(pdf_path: str | Path, page_num: int = 0,
                 dpi: int = RENDER_DPI) -> Image.Image:
    import fitz
    doc = fitz.open(str(pdf_path))
    pix = doc[page_num].get_pixmap(
        matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ── Grid line detection ────────────────────────────────────────────────────────

def _binarise(img_gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold → binary (dark features = 255)."""
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=8)
    return binary


def _detect_lines(binary: np.ndarray,
                  img_h: int, img_w: int) -> tuple[list[float], list[float]]:
    """
    Detect dominant horizontal and vertical lines via column/row projection.

    Strategy
    --------
    • Sum each row → peaks = horizontal grid lines
    • Sum each column → peaks = vertical grid lines
    • Threshold at LINE_PROJ_THRESH × max to keep only the longest lines
      (dashed grid lines show up clearly in the projection even without
       morphological dilation, as long as DPI is high enough)
    """
    h_proj = binary.sum(axis=1).astype(np.float32)
    v_proj = binary.sum(axis=0).astype(np.float32)

    h_thresh = float(h_proj.max()) * LINE_PROJ_THRESH
    v_thresh = float(v_proj.max()) * LINE_PROJ_THRESH

    h_lines = _runs_above(h_proj, h_thresh)
    v_lines = _runs_above(v_proj, v_thresh)
    return h_lines, v_lines


def _runs_above(proj: np.ndarray, threshold: float) -> list[float]:
    """Find weighted centroids of consecutive runs above threshold."""
    above  = (proj > threshold).astype(np.uint8)
    n, labels = cv2.connectedComponents(above.reshape(-1, 1))
    positions: list[float] = []
    for label in range(1, n):
        idx     = np.where(labels.flatten() == label)[0]
        weights = proj[idx]
        positions.append(float(np.average(idx, weights=weights)))
    return positions


def _cluster(positions: list[float], gap: int = LINE_CLUSTER_GAP) -> list[float]:
    """Merge lines that are within `gap` pixels of each other."""
    if not positions:
        return []
    positions = sorted(positions)
    clusters: list[list[float]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


# ── Column classification at each intersection ─────────────────────────────────

def _classify_patch(gray_patch: np.ndarray) -> tuple[str, float]:
    """
    Analyse a small grayscale patch centred on a grid intersection.

    Returns (shape, confidence).  shape ∈ {round, square, rectangle, i_beam, unknown}.
    Returns ("", 0.0) if no column symbol is found.
    """
    h, w    = gray_patch.shape
    _, bin_ = cv2.threshold(gray_patch, 160, 255, cv2.THRESH_BINARY_INV)

    density = float(np.sum(bin_ > 0)) / bin_.size
    if density < DENSITY_THRESH:
        return "", 0.0                              # nothing here

    # Find contours of dark blobs
    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown", 0.5

    # Use the largest contour
    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 4:
        return "unknown", 0.4

    perimeter    = cv2.arcLength(cnt, True)
    circularity  = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect       = bw / bh if bh > 0 else 1.0

    # Classify
    if circularity >= CIRCULARITY_MIN:
        shape = "round"
        conf  = 0.5 + circularity * 0.5
    elif 0.65 < aspect < 1.55:
        shape = "square"
        conf  = 0.6
    elif aspect >= 1.55 or aspect <= 0.65:
        shape = "rectangle"
        conf  = 0.55
    else:
        shape = "unknown"
        conf  = 0.4

    # I-beam heuristic: two separated dark blobs on a vertical axis
    if len(contours) >= 2:
        centres = [cv2.moments(c) for c in contours]
        ys      = [m["m01"] / m["m00"] for m in centres if m["m00"] > 0]
        xs      = [m["m10"] / m["m00"] for m in centres if m["m00"] > 0]
        if ys and max(ys) - min(ys) > h * 0.3 and max(xs) - min(xs) < w * 0.3:
            shape = "i_beam"
            conf  = 0.55

    return shape, min(1.0, conf)


# ── Main detection pipeline ────────────────────────────────────────────────────

def detect_grid(
    img: Image.Image,
    debug: bool = False,
) -> dict:
    """
    Run the full grid-detection pipeline on a PIL image.

    Returns
    -------
    {
        "h_lines": [y, ...],          # clustered horizontal grid-line Y positions
        "v_lines": [x, ...],          # clustered vertical grid-line X positions
        "columns": [
            {
                "shape":      str,    # round | square | rectangle | i_beam | unknown
                "confidence": float,
                "bbox":       [x1,y1,x2,y2],   # pixel coords in original image
                "cx":         float,
                "cy":         float,
            }, ...
        ],
        "debug_img": PIL.Image | None,
    }
    """
    img_np   = np.array(img.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    H, W     = img_gray.shape

    binary   = _binarise(img_gray)
    raw_h, raw_v = _detect_lines(binary, H, W)
    h_lines  = _cluster(raw_h, gap=LINE_CLUSTER_GAP)
    v_lines  = _cluster(raw_v, gap=LINE_CLUSTER_GAP)

    columns: list[dict] = []
    for cy in h_lines:
        for cx in v_lines:
            x1 = max(0, int(cx) - CROSS_RADIUS)
            y1 = max(0, int(cy) - CROSS_RADIUS)
            x2 = min(W, int(cx) + CROSS_RADIUS)
            y2 = min(H, int(cy) + CROSS_RADIUS)
            patch = img_gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            shape, conf = _classify_patch(patch)
            if not shape:
                continue
            columns.append({
                "shape":      shape,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
                "cx":         float(cx),
                "cy":         float(cy),
            })

    debug_img = _draw_debug(img, h_lines, v_lines, columns) if debug else None

    return {
        "h_lines":   h_lines,
        "v_lines":   v_lines,
        "columns":   columns,
        "debug_img": debug_img,
    }


def detect_page(
    pdf_path: str | Path,
    page_num: int = 0,
    dpi: int = RENDER_DPI,
    debug: bool = False,
) -> dict:
    """Convenience wrapper: PDF → grid detection result."""
    img    = pdf_to_image(pdf_path, page_num, dpi)
    result = detect_grid(img, debug=debug)
    result["image"] = img
    result["page_num"] = page_num
    result["dpi"] = dpi
    return result


# ── Debug visualisation ────────────────────────────────────────────────────────

def _draw_debug(img: Image.Image,
                h_lines: list[float], v_lines: list[float],
                columns: list[dict]) -> Image.Image:
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    W, H = out.size

    # Grid lines (semi-transparent)
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in h_lines:
        od.line([(0, y), (W, y)], fill=(0, 120, 255, 60), width=1)
    for x in v_lines:
        od.line([(x, 0), (x, H)], fill=(255, 120, 0, 60), width=1)
    out = Image.alpha_composite(out, overlay)
    draw = ImageDraw.Draw(out)

    # Intersections (dots for no-column; boxes for column)
    for cy in h_lines:
        for cx in v_lines:
            draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(180, 180, 180, 120))

    for col in columns:
        x1, y1, x2, y2 = col["bbox"]
        colour = SHAPE_COLOURS.get(col["shape"], SHAPE_COLOURS["unknown"]) + (220,)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        lbl = f"{col['shape'][:3]} {int(col['confidence']*100)}%"
        draw.rectangle([x1, max(0, y1-14), x1+len(lbl)*7+4, y1], fill=colour)
        draw.text((x1+2, max(0, y1-14)), lbl, fill=(0, 0, 0, 255))

    return out.convert("RGB")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end structural grid detector")
    parser.add_argument("--pdf",   required=True, help="Path to PDF or image file")
    parser.add_argument("--page",  type=int, default=0, help="Page number (0-based)")
    parser.add_argument("--dpi",   type=int, default=RENDER_DPI)
    parser.add_argument("--debug", action="store_true", help="Save annotated debug image")
    parser.add_argument("--out",   default="grid_debug.png", help="Debug image output path")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return

    print(f"Processing {path.name} page {args.page} @ {args.dpi} DPI …")
    result = detect_page(path, args.page, args.dpi, debug=args.debug)

    print(f"  H grid lines : {len(result['h_lines'])} → {[round(y) for y in result['h_lines']]}")
    print(f"  V grid lines : {len(result['v_lines'])} → {[round(x) for x in result['v_lines']]}")
    print(f"  Columns found: {len(result['columns'])}")
    for i, col in enumerate(result["columns"], 1):
        print(f"    [{i:3d}] {col['shape']:12s} conf={col['confidence']:.0%}  "
              f"cx={col['cx']:.0f}  cy={col['cy']:.0f}  bbox={col['bbox']}")

    summary = {k: v for k, v in result.items() if k not in ("debug_img", "image")}
    print(json.dumps(summary, indent=2))

    if args.debug and result["debug_img"]:
        result["debug_img"].save(args.out)
        print(f"\nDebug image saved → {args.out}")


if __name__ == "__main__":
    main()
