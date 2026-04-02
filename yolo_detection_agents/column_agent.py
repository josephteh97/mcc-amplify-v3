"""
column_agent.py — YOLOv11 column detection agent for mcc-amplify-v3.

Uses column-detect.pt (trained on MCC structural floor plans) to detect
structural columns and replaces the Ollama-vision-based pdf_detection_agent.

Model weights location (copy from Linux before first use):
    yolo_detection_agents/weights/column-detect.pt

    Linux source:
        ~/Document/generate-yolo-training-datasest-columns/column-detect.pt

Human feedback compatibility:
    The detection memory (detections.db + memory.json) schema is preserved
    from the old pdf_detection_agent so that feedback_app.py continues to work
    without modification.

Usage:
    from yolo_detection_agents.column_agent import YOLOColumnAgent

    agent  = YOLOColumnAgent()
    result = agent.detect("floor_plan.pdf", page_num=0)
    # result["detections"] — list of canonical column dicts
    # result["total_columns"] — int
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .base_yolo_agent import BaseYOLOAgent


# ── Default weights path (relative to this file's directory) ──────────────────
_WEIGHTS_DEFAULT = Path(__file__).parent / "weights" / "column-detect.pt"

# ── Memory paths (same directory as old pdf_detection_agent to keep parity) ───
_LEGACY_AGENT_DIR = Path(__file__).parent.parent / "pdf_detection_agent"
DB_PATH           = _LEGACY_AGENT_DIR / "detections.db"
MEMORY_JSON_PATH  = _LEGACY_AGENT_DIR / "memory.json"


# ── YOLO class name → canonical shape mapping ─────────────────────────────────
# Adjust if column-detect.pt uses different class names.
# Unknown classes are kept as detections with shape="unknown".
_CLASS_TO_SHAPE: dict[str, str] = {
    "column":           "square",
    "column_square":    "square",
    "column_rect":      "rectangle",
    "column_rectangle": "rectangle",
    "column_round":     "round",
    "column_circular":  "round",
    "column_i":         "i_beam",
    "column_i_beam":    "i_beam",
    # ducts / other elements from training set — skip these
    "duct":             "_skip",
    "duct_round":       "_skip",
    "duct_square":      "_skip",
}

# Aspect-ratio fallback when class name doesn't encode shape
_ROUND_RATIO_MAX  = 0.15   # |w-h|/(w+h) below this → round
_RECT_RATIO_MIN   = 0.25   # |w-h|/(w+h) above this → rectangle


# ══════════════════════════════════════════════════════════════════════════════
# YOLOColumnAgent
# ══════════════════════════════════════════════════════════════════════════════

class YOLOColumnAgent(BaseYOLOAgent):
    """
    Per-element YOLO agent for structural column detection.

    Args:
        weights_path: Path to column-detect.pt (defaults to
                      yolo_detection_agents/weights/column-detect.pt).
        conf_threshold: Minimum YOLO confidence to keep a detection (default 0.35).
        render_dpi: DPI for PDF rasterisation (default 150).
        save_to_memory: Whether to persist results to detections.db (default True).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.35,
        render_dpi: int = 150,
        save_to_memory: bool = True,
    ) -> None:
        super().__init__(
            weights_path   = weights_path or _WEIGHTS_DEFAULT,
            conf_threshold = conf_threshold,
            render_dpi     = render_dpi,
        )
        self._save_to_memory = save_to_memory

    # ── Override detect() to add memory persistence ───────────────────────────

    def detect(self, path: str | Path, page_num: int = 0) -> dict:
        result = super().detect(path, page_num=page_num)
        if "error" not in result and self._save_to_memory:
            try:
                _save_to_db(result)
            except Exception as exc:
                # Memory failure must not break the pipeline
                result.setdefault("warnings", []).append(
                    f"Memory persistence failed: {exc}"
                )
        return result

    # ── _postprocess implementation ───────────────────────────────────────────

    def _postprocess(
        self,
        raw_detections: list[dict],
        page_num: int,
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        """
        Convert raw YOLO boxes to canonical column dicts, then run NMS.

        Skips non-column classes (ducts etc.).
        Infers shape from class name or bbox aspect ratio as fallback.
        """
        canonical: list[dict] = []
        for raw in raw_detections:
            class_name = raw["type"].lower()
            shape = _CLASS_TO_SHAPE.get(class_name)

            if shape is None:
                # Unknown class: keep only if it looks like a column keyword
                if "col" in class_name:
                    shape = _infer_shape_from_bbox(raw["bbox"])
                else:
                    continue  # skip non-column classes

            if shape == "_skip":
                continue

            bbox = raw["bbox"]   # [x1, y1, x2, y2] in pixels
            canonical.append({
                "bbox_page":   bbox,
                "shape":       shape,
                "confidence":  round(raw["confidence"], 4),
                "notes":       f"yolo:{class_name}",
                "tile_index":  None,
                "page_num":    page_num,
                "is_circular": shape == "round",
                "width_mm":    None,
                "depth_mm":    None,
                "diameter_mm": None,
                "type_mark":   None,
            })

        return self._nms(canonical)


# ══════════════════════════════════════════════════════════════════════════════
# Shape inference from bounding-box geometry
# ══════════════════════════════════════════════════════════════════════════════

def _infer_shape_from_bbox(bbox: list[float]) -> str:
    """Guess column shape from bbox aspect ratio when class name is ambiguous."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    ratio = abs(w - h) / (w + h)
    if ratio < _ROUND_RATIO_MAX:
        return "round"
    if ratio > _RECT_RATIO_MIN:
        return "rectangle"
    return "square"


# ══════════════════════════════════════════════════════════════════════════════
# Memory persistence (same schema as pdf_detection_agent/agent.py)
# Kept here so the rest of the agent is self-contained.
# ══════════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id     TEXT PRIMARY KEY,
            timestamp  TEXT NOT NULL,
            file_path  TEXT NOT NULL,
            page_num   INTEGER NOT NULL,
            image_w    INTEGER,
            image_h    INTEGER,
            total_cols INTEGER,
            model      TEXT,
            tiles      INTEGER
        );
        CREATE TABLE IF NOT EXISTS columns (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id     TEXT NOT NULL REFERENCES runs(run_id),
            col_id     INTEGER,
            shape      TEXT,
            confidence REAL,
            notes      TEXT,
            bbox_x1    REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
            tile_index INTEGER,
            page_num   INTEGER
        );
        CREATE TABLE IF NOT EXISTS corrections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_hash  TEXT NOT NULL,
            file_path  TEXT,
            page_num   INTEGER,
            tile_index INTEGER,
            x_offset   INTEGER,
            y_offset   INTEGER,
            action     TEXT NOT NULL,
            shape      TEXT,
            bbox_x1    REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
            notes      TEXT,
            timestamp  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tile_notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_hash   TEXT NOT NULL,
            file_path   TEXT,
            page_num    INTEGER,
            tile_index  INTEGER,
            description TEXT,
            timestamp   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_col_run    ON columns(run_id);
        CREATE INDEX IF NOT EXISTS idx_col_shape  ON columns(shape);
        CREATE INDEX IF NOT EXISTS idx_col_conf   ON columns(confidence);
        CREATE INDEX IF NOT EXISTS idx_run_file   ON runs(file_path);
        CREATE INDEX IF NOT EXISTS idx_corr_hash  ON corrections(tile_hash);
        CREATE INDEX IF NOT EXISTS idx_tnotes_hash ON tile_notes(tile_hash);
    """)
    return con


def _save_to_db(result: dict) -> str:
    """Persist a detect() result to detections.db; returns the new run_id."""
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
