"""
column_agent.py — OpenAI-vision column detection agent for mcc-amplify-v3.

Drop-in replacement for the YOLO-based column detector. Keeps the
`YOLOColumnAgent` class name and `.detect(path, page_num)` contract so the
controller and `_parse_detections()` continue to work unchanged.

Detection is performed by GPT-4o vision: the page is rendered at a moderate
DPI, tiled to stay under the per-image token budget, and each tile is sent
with a strict JSON schema asking for column bounding boxes in tile-local
pixel space. Boxes are lifted back to full-page pixel coordinates, NMS'd,
and returned in the canonical per-detection format.

Env:
    OPENAI_API_KEY  — preferred
    or `openai_key.txt` at the repo root (auto-loaded as a fallback)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base_yolo_agent import render_pdf_page


# ── Config ────────────────────────────────────────────────────────────────────

_AGENT_DIR = Path(__file__).parent
DB_PATH    = _AGENT_DIR / "detections.db"
_REPO_ROOT = _AGENT_DIR.parent
_KEY_FILE  = _REPO_ROOT / "openai_key.txt"

_MODEL        = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
_RENDER_DPI   = 200
_TILE_SIZE    = 1536
_TILE_OVERLAP = 192
_NMS_IOU      = 0.45
_MIN_CONF     = 0.35
_MAX_WORKERS  = 8


_PROMPT = (
    "You are a structural-drawing analyser. The image is a tile from an "
    "architectural floor plan. Identify every STRUCTURAL COLUMN visible in "
    "this tile. Columns appear as small solid-filled (black/hatched) "
    "rectangles or circles, typically 15–60 pixels on a side, placed on a "
    "grid. IGNORE walls, doors, furniture, dimension text, lift shafts, "
    "and legend symbols.\n\n"
    "Return STRICT JSON of the form:\n"
    '{"columns":[{"x1":..,"y1":..,"x2":..,"y2":..,'
    '"shape":"square|round|rectangle","confidence":0.0-1.0}]}\n'
    "Coordinates are integer pixels within THIS tile (origin top-left). "
    "If no columns are present, return {\"columns\":[]}."
)


def _load_api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or create openai_key.txt."
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thr]
    return keep


class YOLOColumnAgent:
    """OpenAI-vision drop-in replacement; keeps the YOLO class name so the
    controller singleton (`_YOLO_AGENT = YOLOColumnAgent()`) is unchanged."""

    def __init__(
        self,
        conf_threshold: float = _MIN_CONF,
        render_dpi: int = _RENDER_DPI,
        model: str = _MODEL,
        save_to_memory: bool = True,
        max_tiles: int | None = None,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._render_dpi     = render_dpi
        self._model_name     = model
        self._save_to_memory = save_to_memory
        self._max_tiles      = max_tiles
        self._client: Any    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, path: str | Path, page_num: int = 0) -> dict:
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}

        img = (render_pdf_page(path, page_num, dpi=self._render_dpi)
               if path.suffix.lower() == ".pdf"
               else Image.open(path).convert("RGB"))
        W, H = img.size

        try:
            raw, api_calls = self._run_vision(img)
        except Exception as exc:
            return {"error": f"OpenAI vision call failed: {exc}"}

        detections = self._postprocess(raw, page_num)

        for idx, det in enumerate(detections, 1):
            det["id"] = idx

        by_shape: dict[str, int] = {}
        for d in detections:
            by_shape[d["shape"]] = by_shape.get(d["shape"], 0) + 1
        avg_conf = (round(sum(d["confidence"] for d in detections)
                          / len(detections), 3)
                    if detections else 0.0)

        result = {
            "file":          str(path),
            "page":          page_num,
            "image_size":    [W, H],
            "total_columns": len(detections),
            "detections":    detections,
            "stats":         {"by_shape": by_shape, "avg_confidence": avg_conf},
            "model":         self._model_name,
            "api_calls":     api_calls,
            "timestamp":     datetime.now().isoformat(),
            "_page_image":   img,
        }

        if self._save_to_memory:
            try:
                _save_to_db(result)
            except Exception as exc:
                result.setdefault("warnings", []).append(
                    f"Memory persistence failed: {exc}"
                )
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI(api_key=_load_api_key())

    def _run_vision(self, img: Image.Image) -> tuple[list[dict], int]:
        """Tile the image, call the model per tile, return (raw detections
        in FULL-PAGE pixel coordinates, api_call_count)."""
        self._ensure_client()
        W, H = img.size
        step = _TILE_SIZE - _TILE_OVERLAP

        tiles: list[tuple[int, int, Image.Image]] = []
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                x1 = min(x0 + _TILE_SIZE, W)
                y1 = min(y0 + _TILE_SIZE, H)
                xa = max(0, x1 - _TILE_SIZE)
                ya = max(0, y1 - _TILE_SIZE)
                tiles.append((xa, ya, img.crop((xa, ya, x1, y1))))

        if self._max_tiles is not None:
            tiles = tiles[: self._max_tiles]

        def _work(item: tuple[int, int, Image.Image]) -> list[dict]:
            xa, ya, tile = item
            results = []
            for c in self._call_model(tile):
                results.append({
                    "type":       c.get("shape", "square"),
                    "bbox":       [c["x1"] + xa, c["y1"] + ya,
                                   c["x2"] + xa, c["y2"] + ya],
                    "confidence": float(c.get("confidence", 0.5)),
                })
            return results

        out: list[dict] = []
        workers = min(_MAX_WORKERS, max(1, len(tiles)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for tile_out in pool.map(_work, tiles):
                out.extend(tile_out)
        return out, len(tiles)

    def _call_model(self, tile: Image.Image) -> list[dict]:
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        resp = self._client.chat.completions.create(
            model=self._model_name,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/png;base64,{b64}",
                         "detail": "high",
                     }},
                ],
            }],
        )
        txt = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            return []
        cols = data.get("columns") or []
        cleaned: list[dict] = []
        for c in cols:
            try:
                cleaned.append({
                    "x1": int(c["x1"]), "y1": int(c["y1"]),
                    "x2": int(c["x2"]), "y2": int(c["y2"]),
                    "shape": str(c.get("shape", "square")).lower(),
                    "confidence": float(c.get("confidence", 0.5)),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return cleaned

    def _postprocess(self, raw: list[dict], page_num: int) -> list[dict]:
        if not raw:
            return []
        boxes  = np.array([r["bbox"]       for r in raw], dtype=np.float32)
        confs  = np.array([r["confidence"] for r in raw], dtype=np.float32)
        keep   = _nms(boxes, confs, _NMS_IOU)

        dets: list[dict] = []
        for i in keep:
            if confs[i] < self._conf_threshold:
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            raw_shape = raw[i]["type"]
            shape = raw_shape if raw_shape in {"square", "round",
                                               "rectangle"} else "square"
            dets.append({
                "bbox_page":   [x1, y1, x2, y2],
                "shape":       shape,
                "confidence":  round(float(confs[i]), 4),
                "notes":       f"openai:{raw_shape}",
                "tile_index":  None,
                "page_num":    page_num,
                "is_circular": shape == "round",
                "width_mm":    None,
                "depth_mm":    None,
                "diameter_mm": None,
                "type_mark":   None,
            })
        return dets


# ══════════════════════════════════════════════════════════════════════════════
# Memory persistence (schema unchanged from the YOLO agent)
# ══════════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL, page_num INTEGER NOT NULL,
            image_w INTEGER, image_h INTEGER, total_cols INTEGER,
            model TEXT, tiles INTEGER
        );
        CREATE TABLE IF NOT EXISTS columns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            col_id INTEGER, shape TEXT, confidence REAL, notes TEXT,
            bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
            tile_index INTEGER, page_num INTEGER
        );
    """)
    return con


def _save_to_db(result: dict) -> str:
    run_id = str(uuid.uuid4())
    wh     = result.get("image_size", [None, None])
    con    = _db()
    with con:
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, result.get("timestamp"), result.get("file"),
             result.get("page", 0), wh[0], wh[1],
             result.get("total_columns", 0), result.get("model"), None),
        )
        for det in result.get("detections", []):
            bb = det.get("bbox_page", [None] * 4)
            con.execute(
                "INSERT INTO columns "
                "(run_id,col_id,shape,confidence,notes,"
                " bbox_x1,bbox_y1,bbox_x2,bbox_y2,tile_index,page_num)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, det.get("id"), det.get("shape"), det.get("confidence"),
                 det.get("notes"), bb[0], bb[1], bb[2], bb[3],
                 det.get("tile_index"), det.get("page_num")),
            )
    con.close()
    return run_id
