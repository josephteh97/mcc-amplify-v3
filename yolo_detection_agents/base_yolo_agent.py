"""
base_yolo_agent.py — Abstract base class for per-element YOLO detection agents.

Responsibilities:
  • Load a YOLOv11 .pt weights file at construction time.
  • Render a PDF page (or open an image) at a configurable DPI.
  • Apply CLAHE contrast enhancement (improves faint engineering-drawing lines).
  • Run YOLO inference and return raw detections in pixel-space.
  • Expose a hook `_postprocess()` for subclasses to normalise detections into
    the canonical mcc-amplify-v3 format expected by controller.py.

Canonical per-detection dict (matches what _parse_detections() reads):
    {
        "id":          int,
        "bbox_page":   [x1, y1, x2, y2],   # pixels on the full-page image
        "shape":       str,                  # "square"|"round"|"rectangle"|"unknown"
        "confidence":  float,
        "notes":       str,
        "tile_index":  None,
        "page_num":    int,
        "is_circular": bool,
        "width_mm":    None,
        "depth_mm":    None,
        "diameter_mm": None,
        "type_mark":   None,
    }

Full detect() return dict (matches what controller._run_column_detection returns):
    {
        "file":          str,
        "page":          int,
        "image_size":    [W, H],
        "total_columns": int,
        "detections":    [<canonical det>, ...],
        "stats":         {"by_shape": dict, "avg_confidence": float},
        "model":         str,
        "timestamp":     str,
    }
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ── Optional heavy imports (fail gracefully so tests can mock them) ────────────

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None          # type: ignore
    _CV2_AVAILABLE = False

try:
    import fitz as _fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _fitz = None          # type: ignore
    _FITZ_AVAILABLE = False

try:
    from pdf2image import convert_from_path as _pdf2img
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _pdf2img = None       # type: ignore
    _PDF2IMAGE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# BaseYOLOAgent
# ══════════════════════════════════════════════════════════════════════════════

class BaseYOLOAgent(ABC):
    """
    Subclass this and implement `_postprocess()` to create a per-element agent.

    Args:
        weights_path: Path to the YOLOv11 .pt file.
        conf_threshold: Minimum confidence to keep a detection (default 0.35).
        render_dpi: DPI used when rasterising PDF pages (default 150).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.35,
        render_dpi: int = 150,
    ) -> None:
        self._weights_path   = Path(weights_path)
        self._conf_threshold = conf_threshold
        self._render_dpi     = render_dpi
        self._model: Any     = None   # loaded lazily on first detect()

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, path: str | Path, page_num: int = 0) -> dict:
        """
        Main entry point.  Accepts a PDF or an image file.

        Returns the canonical dict described in the module docstring.
        """
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}

        self._ensure_model_loaded()

        img = self._load_image(path, page_num)
        W, H = img.size

        raw = self._run_inference(img)
        detections = self._postprocess(raw, page_num, W, H)

        # Assign sequential IDs
        for idx, det in enumerate(detections, 1):
            det["id"] = idx

        by_shape: dict[str, int] = {}
        for d in detections:
            by_shape[d["shape"]] = by_shape.get(d["shape"], 0) + 1
        avg_conf = (
            round(sum(d["confidence"] for d in detections) / len(detections), 3)
            if detections else 0.0
        )

        return {
            "file":          str(path),
            "page":          page_num,
            "image_size":    [W, H],
            "total_columns": len(detections),
            "detections":    detections,
            "stats":         {"by_shape": by_shape, "avg_confidence": avg_conf},
            "model":         self._weights_path.name,
            "timestamp":     datetime.now().isoformat(),
        }

    # ── Subclass hook ──────────────────────────────────────────────────────────

    @abstractmethod
    def _postprocess(
        self,
        raw_detections: list[dict],
        page_num: int,
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        """
        Convert raw YOLO detections ({"type", "bbox", "confidence"}) into the
        canonical per-detection dicts expected by controller._parse_detections().

        `raw_detections` bbox values are already in full-image pixel coordinates.
        """

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self._weights_path}\n"
                f"Copy column-detect.pt from Linux:\n"
                f"  cp ~/Document/generate-yolo-training-datasest-columns/column-detect.pt "
                f"<repo>/yolo_detection_agents/weights/column-detect.pt"
            )
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._weights_path))
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

    def _load_image(self, path: Path, page_num: int) -> Image.Image:
        """Render PDF page or open image; always returns RGB PIL.Image."""
        if path.suffix.lower() == ".pdf":
            return self._render_pdf(path, page_num)
        return Image.open(path).convert("RGB")

    def _render_pdf(self, path: Path, page_num: int) -> Image.Image:
        """Rasterise a PDF page to PIL Image at self._render_dpi."""
        if _FITZ_AVAILABLE:
            doc  = _fitz.open(str(path))
            page = doc[page_num]
            mat  = _fitz.Matrix(self._render_dpi / 72.0, self._render_dpi / 72.0)
            pix  = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if _PDF2IMAGE_AVAILABLE:
            pages = _pdf2img(str(path), dpi=self._render_dpi, first_page=page_num + 1,
                             last_page=page_num + 1)
            if pages:
                return pages[0].convert("RGB")

        raise RuntimeError(
            "No PDF renderer available. Install PyMuPDF (pip install pymupdf) "
            "or pdf2image (pip install pdf2image)."
        )

    def _enhance(self, img: Image.Image) -> np.ndarray:
        """
        CLAHE contrast enhancement in LAB colour space.
        Improves visibility of faint structural drawing lines before inference.
        Falls back to plain numpy array if OpenCV is not available.
        """
        arr = np.array(img)  # HxWx3, uint8, RGB
        if not _CV2_AVAILABLE:
            return arr

        lab   = _cv2.cvtColor(arr, _cv2.COLOR_RGB2LAB)
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)

    def _run_inference(self, img: Image.Image) -> list[dict]:
        """
        Run YOLO on the (optionally enhanced) image.

        Returns a list of raw dicts:
            {"type": str, "bbox": [x1, y1, x2, y2], "confidence": float}
        All bbox coordinates are in full-image pixel space.
        """
        enhanced = self._enhance(img)
        results  = self._model(enhanced, verbose=False, conf=self._conf_threshold)
        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "type":       r.names[int(box.cls)],
                    "bbox":       box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                })
        return detections

    # ── NMS utility (available to subclasses) ─────────────────────────────────

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _nms(detections: list[dict], threshold: float = 0.1) -> list[dict]:
        """IoU-based NMS on the bbox_page field."""
        kept: list[dict] = []
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            bb = det["bbox_page"]
            if not any(BaseYOLOAgent._iou(bb, k["bbox_page"]) > threshold for k in kept):
                kept.append(det)
        return kept
