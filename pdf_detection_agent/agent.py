"""
agent.py — PDF Column Detection Agent (OpenAI Vision)
=====================================================
Single-file agent: PDF tiling, OpenAI GPT-4o vision detection, SQLite memory, agentic loop.

Improved harness over the Ollama-based version:
  - OpenAI GPT-4o vision for vastly better spatial reasoning
  - Structured JSON output (response_format) for reliable parsing
  - Reference images as few-shot examples alongside each tile
  - Tile content pre-filter to skip empty tiles (saves API costs)
  - Higher resolution tiles (1024x1024 coordinate space)
  - Simplified codebase (no model-specific branches)

    from agent import AgentSession, detect_file, memory_search

    result = detect_file("floor_plan.pdf", page_num=0)
    rows   = memory_search(shape="round", min_confidence=0.8)
    answer = AgentSession().ask("How many columns on page 0 of /path/to/plan.pdf?")
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

from PIL import Image, ImageDraw
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Load OpenAI API key from env var or openai_key.txt in agent directory."""
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    key_file = Path(__file__).parent / "openai_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    raise RuntimeError(
        "No OpenAI API key found. "
        "Set OPENAI_API_KEY env var or create openai_key.txt in the agent directory."
    )

OPENAI_API_KEY   = _load_api_key()
DEFAULT_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
TILE_SIZE        = 1280
TILE_STEP        = 1080          # 200 px overlap
RENDER_DPI       = 150
VISION_SIZE      = 1024          # tiles resized to this square before sending to API
MAX_TOOL_LOOPS   = 8
SKILLS_DIR       = Path(__file__).parent / "skills"
DB_PATH          = Path(__file__).parent / "detections.db"
GT_DIR           = Path(__file__).parent / "ground_truth" / "columns"
MEMORY_JSON_PATH = Path(__file__).parent / "memory.json"
VALID_SHAPES     = {"column"}
# Minimum dark-pixel ratio for a tile to be worth sending to the API
TILE_CONTENT_THRESHOLD = 0.01

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-initialise and return the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# SQLite memory
# ══════════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
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
        CREATE INDEX IF NOT EXISTS idx_col_run   ON columns(run_id);
        CREATE INDEX IF NOT EXISTS idx_col_shape ON columns(shape);
        CREATE INDEX IF NOT EXISTS idx_col_conf  ON columns(confidence);
        CREATE INDEX IF NOT EXISTS idx_run_file  ON runs(file_path);
        CREATE INDEX IF NOT EXISTS idx_corr_hash ON corrections(tile_hash);
        CREATE INDEX IF NOT EXISTS idx_tnotes_hash ON tile_notes(tile_hash);
    """)
    return con


def _save_to_db(result: dict) -> str:
    """Persist a detect_file() result; returns the new run_id."""
    run_id = str(uuid.uuid4())
    wh     = result.get("image_size", [None, None])
    con    = _db()
    with con:
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, result.get("timestamp"), result.get("file"),
             result.get("page", 0), wh[0], wh[1],
             result.get("total_columns", 0), result.get("model"),
             result.get("stats", {}).get("tiles")),
        )
        for det in result.get("detections", []):
            bb = det.get("bbox_page", [None] * 4)
            con.execute(
                "INSERT INTO columns "
                "(run_id,col_id,shape,confidence,notes,bbox_x1,bbox_y1,bbox_x2,bbox_y2,tile_index,page_num)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, det.get("id"), det.get("shape"), det.get("confidence"),
                 det.get("notes"), bb[0], bb[1], bb[2], bb[3],
                 det.get("tile_index"), det.get("page_num")),
            )
    con.close()
    return run_id


def memory_search(file: str | None = None, shape: str | None = None,
                  min_confidence: float = 0.0, limit: int = 50) -> list[dict]:
    """Query stored detections. All filters are optional."""
    clauses, params = ["c.confidence >= ?"], [min_confidence]
    if file:
        clauses.append("r.file_path LIKE ?");  params.append(f"%{file}%")
    if shape:
        clauses.append("c.shape = ?");         params.append(shape)
    con  = _db()
    rows = [dict(r) for r in con.execute(f"""
        SELECT r.run_id, r.timestamp, r.file_path, r.page_num, r.model,
               c.col_id, c.shape, c.confidence, c.notes,
               c.bbox_x1, c.bbox_y1, c.bbox_x2, c.bbox_y2
        FROM   columns c JOIN runs r ON c.run_id = r.run_id
        WHERE  {" AND ".join(clauses)}
        ORDER  BY c.confidence DESC LIMIT ?
    """, params + [limit]).fetchall()]
    con.close()
    return rows


def memory_runs(limit: int = 20) -> list[dict]:
    """Return the most recent detection runs."""
    con  = _db()
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()]
    con.close()
    return rows


def memory_stats() -> dict:
    """Aggregate statistics across all stored detections."""
    con = _db()
    total_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_cols = con.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
    by_shape   = {r["shape"]: r["cnt"] for r in con.execute(
        "SELECT shape, COUNT(*) cnt FROM columns GROUP BY shape").fetchall()}
    avg_conf   = con.execute("SELECT AVG(confidence) FROM columns").fetchone()[0]
    con.close()
    return {"total_runs": total_runs, "total_columns": total_cols,
            "by_shape": by_shape, "avg_confidence": round(avg_conf or 0.0, 3)}


def memory_clear(run_id: str | None = None) -> dict:
    """Delete one run (pass run_id) or wipe everything (run_id=None)."""
    con = _db()
    with con:
        if run_id:
            c = con.execute("DELETE FROM columns WHERE run_id=?", (run_id,)).rowcount
            r = con.execute("DELETE FROM runs    WHERE run_id=?", (run_id,)).rowcount
        else:
            c = con.execute("DELETE FROM columns").rowcount
            r = con.execute("DELETE FROM runs").rowcount
    con.close()
    return {"runs_deleted": r, "columns_deleted": c}


# ── memory.json (fast per-tile correction cache) ──────────────────────────────

def _mjson_load() -> dict:
    """Load memory.json; return empty structure if missing."""
    if MEMORY_JSON_PATH.exists():
        try:
            return json.loads(MEMORY_JSON_PATH.read_text())
        except Exception:
            pass
    return {"version": 1, "corrections": {}}


def _mjson_save(data: dict) -> None:
    MEMORY_JSON_PATH.write_text(json.dumps(data, indent=2))


# ── Tile hashing ──────────────────────────────────────────────────────────────

def _tile_hash(img: Image.Image) -> str:
    """Stable 16-char hash of a tile (resize to 64x64 first for robustness)."""
    buf = io.BytesIO()
    img.resize((64, 64), Image.LANCZOS).save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:16]


# ── Human corrections / feedback ─────────────────────────────────────────────

def add_correction(
    tile_hash: str,
    action: str,                  # 'confirm' | 'reject' | 'add'
    shape: str | None = None,
    bbox: list[float] | None = None,  # tile-space [x1,y1,x2,y2]
    notes: str = "",
    file_path: str = "",
    page_num: int = 0,
    tile_index: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
) -> dict:
    """
    Record a human correction for a tile.

    action='confirm' -- an existing detection is correct
    action='reject'  -- an existing detection is a false positive
    action='add'     -- a missed column; supply bbox + shape

    Persists to both SQLite (corrections table) and memory.json.
    """
    if action not in ("confirm", "reject", "add"):
        return {"error": "action must be confirm | reject | add"}
    ts = datetime.now().isoformat()
    bb = bbox or [0.0, 0.0, 0.0, 0.0]

    # SQLite
    con = _db()
    with con:
        con.execute(
            "INSERT INTO corrections "
            "(tile_hash,file_path,page_num,tile_index,x_offset,y_offset,"
            " action,shape,bbox_x1,bbox_y1,bbox_x2,bbox_y2,notes,timestamp)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tile_hash, file_path, page_num, tile_index,
             x_offset, y_offset, action, shape,
             bb[0], bb[1], bb[2], bb[3], notes, ts),
        )
    con.close()

    # memory.json cache
    data = _mjson_load()
    data["corrections"].setdefault(tile_hash, []).append(
        {"action": action, "shape": shape, "bbox": bb, "notes": notes, "timestamp": ts}
    )
    _mjson_save(data)
    return {"ok": True, "tile_hash": tile_hash, "action": action}


def add_corrections_batch(
    tile_hash: str,
    corrections: list[dict],   # each: {action, shape, bbox, notes}
    file_path: str = "",
    page_num: int = 0,
    tile_index: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
) -> dict:
    """
    Record multiple corrections for one tile in a single memory.json write.
    Use instead of looping add_correction() to avoid N reads + N writes.
    """
    if not corrections:
        return {"ok": True, "tile_hash": tile_hash, "count": 0}
    _VALID = ("confirm", "reject", "add")
    for c in corrections:
        if c.get("action") not in _VALID:
            return {"error": f"action must be one of {_VALID}"}
    ts   = datetime.now().isoformat()
    _NB  = [0.0, 0.0, 0.0, 0.0]
    con  = _db()
    with con:
        con.executemany(
            "INSERT INTO corrections "
            "(tile_hash,file_path,page_num,tile_index,x_offset,y_offset,"
            " action,shape,bbox_x1,bbox_y1,bbox_x2,bbox_y2,notes,timestamp)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(tile_hash, file_path, page_num, tile_index, x_offset, y_offset,
              c["action"], c.get("shape"), *(c.get("bbox") or _NB), c.get("notes",""), ts)
             for c in corrections],
        )
    con.close()

    data = _mjson_load()
    slot = data["corrections"].setdefault(tile_hash, [])
    for c in corrections:
        slot.append({"action": c["action"], "shape": c.get("shape"),
                     "bbox": c.get("bbox") or _NB,
                     "notes": c.get("notes",""), "timestamp": ts})
    _mjson_save(data)
    return {"ok": True, "tile_hash": tile_hash, "count": len(corrections)}


def undo_last_correction(tile_hash: str) -> dict:
    """
    Remove the most recent correction for a tile from memory.json.
    SQLite history is intentionally preserved (non-destructive audit trail).
    """
    data = _mjson_load()
    corrections = data["corrections"].get(tile_hash, [])
    if not corrections:
        return {"ok": False, "msg": "no corrections to undo"}
    removed = corrections.pop()
    if not corrections:
        del data["corrections"][tile_hash]
    _mjson_save(data)
    return {"ok": True, "removed": removed}


def clear_all_corrections() -> dict:
    """
    Wipe all human corrections from memory.json and the SQLite corrections table.
    Detection run history (runs/columns tables) is preserved.
    """
    _mjson_save({"version": 1, "corrections": {}})
    con = _db()
    with con:
        con.execute("DELETE FROM corrections")
    con.close()
    return {"ok": True}


def list_corrections(tile_hash: str | None = None, file_path: str | None = None,
                     limit: int = 100) -> list[dict]:
    """List recorded human corrections, optionally filtered by tile or file."""
    clauses, params = [], []
    if tile_hash:
        clauses.append("tile_hash = ?"); params.append(tile_hash)
    if file_path:
        clauses.append("file_path LIKE ?"); params.append(f"%{file_path}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    con  = _db()
    rows = [dict(r) for r in con.execute(
        f"SELECT * FROM corrections {where} ORDER BY timestamp DESC LIMIT ?",
        params + [limit]
    ).fetchall()]
    con.close()
    return rows


def search_similar_tiles(description: str, limit: int = 5) -> list[dict]:
    """Simple keyword search over stored tile descriptions (lightweight vector search)."""
    words   = [w.lower() for w in re.findall(r"\w{4,}", description)]
    if not words:
        return []
    clauses = [f"description LIKE ?" for _ in words]
    params  = [f"%{w}%" for w in words]
    con  = _db()
    rows = [dict(r) for r in con.execute(
        f"SELECT DISTINCT tile_hash, file_path, page_num, tile_index, description "
        f"FROM tile_notes WHERE {' OR '.join(clauses)} LIMIT ?",
        params + [limit]
    ).fetchall()]
    con.close()
    return rows


def _get_tile_context(tile_hash: str, _data: dict | None = None) -> str:
    """Build few-shot correction context to inject into the detection prompt."""
    data = _data if _data is not None else _mjson_load()
    corrections = data["corrections"].get(tile_hash, [])
    if not corrections:
        return ""
    # Deduplicate by bbox
    seen: dict[tuple, dict] = {}
    for c in corrections:
        key = tuple(round(v, 2) for v in c["bbox"][:4])
        seen[key] = c
    confirmed = [c for c in seen.values() if c["action"] in ("confirm", "add")]
    rejected  = [c for c in seen.values() if c["action"] == "reject"]
    yes = "; ".join(f"{c['shape']}@{[round(v,2) for v in c['bbox'][:4]]}" for c in confirmed) or "none"
    no  = "; ".join(f"{c.get('shape','?')}@{[round(v,2) for v in c['bbox'][:4]]}" for c in rejected) or "none"
    return f"\nPrior human corrections -- REAL columns: {yes} | FALSE positives to skip: {no}\n"


# ══════════════════════════════════════════════════════════════════════════════
# PDF / image utilities
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TileInfo:
    index: int; page_num: int
    x_offset: int; y_offset: int
    width: int; height: int        # actual content (may be < TILE_SIZE at edges)
    page_width: int; page_height: int


def _pdf_to_pil(path: Path, page_num: int, dpi: int) -> tuple[Image.Image, int]:
    """Returns (rendered PIL image, total page count) in a single fitz open."""
    import fitz
    doc   = fitz.open(str(path))
    n     = doc.page_count
    pix   = doc[page_num].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), n


def _pdf_page_count(path: Path) -> int:
    import fitz
    doc = fitz.open(str(path))
    n   = doc.page_count
    doc.close()
    return n


def _tiles(img: Image.Image, page_num: int) -> Generator[tuple[Image.Image, _TileInfo], None, None]:
    """Sliding-window tiling. Edge tiles are white-padded to TILE_SIZE x TILE_SIZE."""
    W, H = img.size
    idx  = 0
    y = 0
    while y < H:
        x = 0
        while x < W:
            crop = img.crop((x, y, min(x + TILE_SIZE, W), min(y + TILE_SIZE, H)))
            cw, ch = crop.size
            if cw < TILE_SIZE or ch < TILE_SIZE:
                tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
                tile.paste(crop, (0, 0))
            else:
                tile = crop
            yield tile, _TileInfo(idx, page_num, x, y, cw, ch, W, H)
            idx += 1; x += TILE_STEP
        y += TILE_STEP


def _to_page_bbox(bbox: list[float], t: _TileInfo) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [min(x1, t.width)  + t.x_offset, min(y1, t.height) + t.y_offset,
            min(x2, t.width)  + t.x_offset, min(y2, t.height) + t.y_offset]


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Return annotated copy of img with colour-coded bounding boxes."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        bbox = det.get("bbox_page") or det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        conf   = det.get("confidence", 0.5)
        colour = (0, 200, 80) if conf >= 0.80 else (220, 180, 0) if conf >= 0.50 else (220, 60, 60)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{(det.get('shape') or '?')[:1].upper()} {int(conf*100)}%", fill=colour)
    return out


def _tile_has_content(tile: Image.Image, info: _TileInfo) -> bool:
    """Quick pre-filter: skip tiles with almost no dark features (saves API cost)."""
    content = tile.crop((0, 0, info.width, info.height))
    gray = content.convert("L")
    pixels = list(gray.getdata())
    dark = sum(1 for p in pixels if p < 200)
    return (dark / len(pixels)) >= TILE_CONTENT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth reference images (few-shot visual examples)
# ══════════════════════════════════════════════════════════════════════════════

_KW_SQUARE = {"square", "sqaure", "squa", "sqau", "squ"}
_KW_ROUND  = {"round", "circle"}
_KW_I_BEAM = {"i_beam", "ibeam", "i_col", "i_sqau", "i_squa"}
_KW_RECT   = {"rect"}

_REFERENCES: list[tuple[str, str]] | None = None


def _infer_shape(filename: str) -> str | None:
    """Infer column shape from ground-truth filename, including combined shapes."""
    name = filename.lower()
    has_square = any(kw in name for kw in _KW_SQUARE)
    has_round  = any(kw in name for kw in _KW_ROUND)
    has_i_beam = any(kw in name for kw in _KW_I_BEAM)
    has_rect   = any(kw in name for kw in _KW_RECT)
    if has_i_beam and has_square:
        return "i_square"
    if has_round and has_square:
        return "square_round"
    if has_square:
        return "square"
    if has_rect:
        return "rectangle"
    if has_round:
        return "round"
    if has_i_beam:
        return "i_beam"
    return None


def _load_references() -> list[tuple[str, str]]:
    """
    Scan GT_DIR for images, infer shape from filename, keep the largest per shape.
    Cached after first call -- use references_reload() to refresh.
    """
    global _REFERENCES
    if _REFERENCES is not None:
        return _REFERENCES

    best: dict[str, tuple[int, Image.Image]] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff"):
        for p in GT_DIR.glob(ext):
            shape = _infer_shape(p.name)
            if shape is None:
                continue
            try:
                img  = Image.open(p).convert("RGB")
                area = img.width * img.height
            except Exception:
                continue
            if shape not in best or area > best[shape][0]:
                best[shape] = (area, img)

    _REFERENCES = [(shape, _pil_to_b64(img)) for shape, (_, img) in sorted(best.items())]
    return _REFERENCES


def references_reload() -> list[str]:
    """Rescan GT_DIR and reload references. Call after adding/removing files."""
    global _REFERENCES
    _REFERENCES = None
    return [shape for shape, _ in _load_references()]


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI vision detection
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_DETECT = f"""\
You are an expert structural engineer reading architectural floor plan drawings.
You detect structural columns -- the load-bearing concrete pillars supporting every floor above.

## How columns appear in floor plans
Columns sit at or near beam/grid intersections as small geometric shapes:
- **Solid filled** black/dark squares, rectangles, or circles
- **Outlined hollow** shapes with bold/thick border (very common in CAD drawings)
- **Cross-hatched** rectangles or squares
- **I-beam / H-section** profile symbols
They are often labelled nearby with IDs like C1, C2, H-C1, H-C9.

## Grid lines & balloons
Grid lines are thin dash-dot centrelines. **Grid balloons** are small open circles at
grid line ends carrying axis labels (A, B, 1, 2...). Do NOT confuse them with columns.

## What is NOT a column (critical -- avoid false positives)
- Grid balloons: open/hollow circles at grid line ends with axis labels
- Dimension lines, arrowheads, wall lines, door arcs
- North arrows, scale bars, title block content
- Room/slab labels (SB1, SB2, RCB, NSP...), furniture symbols, text annotations

## Output rules
- The image tile is {VISION_SIZE}x{VISION_SIZE} pixels
- Report bounding boxes as [x1, y1, x2, y2] pixel coordinates in the tile
- confidence: float 0.0-1.0
- shape: one of "square", "rectangle", "round", "i_beam", "square_round", "i_square"
- Detect ALL columns -- a tile may contain many. Do not stop after finding one.
- Respond ONLY with valid JSON.
"""

_USER_DETECT = f"""\
Examine this floor plan tile and detect ALL structural columns.

Columns sit at beam or grid intersections. They may be solid filled, outlined/hollow \
with a thick border, or cross-hatched -- all are valid. Column ID labels nearby \
(C1, C2, H-C1, H-C9, etc.) are a strong indicator.

For every column found output:
  "bbox": [x1, y1, x2, y2]  -- pixel coordinates within this {VISION_SIZE}x{VISION_SIZE} tile
  "confidence": float 0.0-1.0
  "shape": "square" | "rectangle" | "round" | "i_beam" | "square_round" | "i_square"
  "notes": one-line description (e.g. "solid square at grid C-3", "hollow circle near label H-C2")

Respond with this JSON structure:
{{
  "columns": [
    {{"bbox": [x1, y1, x2, y2], "confidence": 0.92, "shape": "square", "notes": "solid filled square at grid intersection"}}
  ],
  "tile_notes": "brief description of what this tile shows"
}}

If no columns are found: {{"columns": [], "tile_notes": "no columns detected"}}
"""


def _openai_chat(messages: list[dict], model: str = DEFAULT_MODEL,
                 max_tokens: int = 2048, temperature: float = 0.1,
                 json_mode: bool = True) -> str:
    """Call OpenAI Chat Completions API. Returns the assistant message content."""
    client = _get_client()
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _parse_json(text: str) -> dict:
    """Robust JSON parser with multiple fallback strategies."""
    text = text.strip()
    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {"columns": parsed}
        return parsed
    except json.JSONDecodeError:
        # Try to extract a complete JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Try to extract a complete JSON array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return {"columns": json.loads(m.group())}
            except json.JSONDecodeError:
                pass
        # Last resort: extract all complete {...} objects
        items = []
        for m in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
            try:
                items.append(json.loads(m.group()))
            except json.JSONDecodeError:
                pass
        if items:
            return {"columns": items}
    return {"columns": [], "tile_notes": f"parse_error: {text[:200]}"}


def _save_tile_note(tile_hash: str, description: str, info: _TileInfo, file_path: str) -> None:
    """Persist tile description for similarity search."""
    con = _db()
    with con:
        con.execute(
            "INSERT INTO tile_notes (tile_hash,file_path,page_num,tile_index,description,timestamp)"
            " VALUES (?,?,?,?,?,?)",
            (tile_hash, file_path, info.page_num, info.index,
             description, datetime.now().isoformat()),
        )
    con.close()


def _cell_dedup(dets: list[dict], w: int, h: int, bbox_key: str,
                grid_cols: int, grid_rows: int) -> list[dict]:
    """Keep highest-confidence detection per grid cell. Works for any bbox key."""
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    best: dict[tuple, dict] = {}
    for det in sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True):
        bbox = det.get(bbox_key, [])
        if len(bbox) != 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        cell = (int(cx / cell_w), int(cy / cell_h))
        if cell not in best:
            best[cell] = det
    return list(best.values())


def _detect_tile(tile: Image.Image, info: _TileInfo, model: str,
                 debug: bool = False, file_path: str = "",
                 _memory: dict | None = None) -> list[dict]:
    """
    Detect columns in a single tile using OpenAI vision API.

    Pipeline:
      1. Crop tile to actual content (strip white padding on edges)
      2. Resize to VISION_SIZE x VISION_SIZE for consistent coordinate space
      3. Build multimodal prompt with reference images + corrections
      4. Call OpenAI API with structured JSON output
      5. Parse response, scale coordinates back to tile space
      6. NMS + cell dedup
    """
    # Crop to actual content then resize to VISION_SIZE
    content   = tile.crop((0, 0, info.width, info.height))
    send_tile = content.resize((VISION_SIZE, VISION_SIZE), Image.LANCZOS)
    tile_hash = _tile_hash(send_tile)
    correction = _get_tile_context(tile_hash, _memory)

    # Build multimodal user message
    user_content: list[dict] = []

    # Few-shot: reference images (sent at low detail to save tokens)
    refs = _load_references()
    if refs:
        ref_text = "Reference examples of structural columns from floor plans:\n"
        for i, (shape, _) in enumerate(refs, 1):
            ref_text += f"  Image {i}: {shape} column\n"
        ref_text += f"\nThe LAST image (Image {len(refs) + 1}) is the floor plan tile to analyse.\n"
        user_content.append({"type": "text", "text": ref_text})
        for _, b64 in refs:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })

    # Correction context + detection prompt
    prompt = _USER_DETECT
    if correction:
        prompt = correction + "\n" + prompt
    user_content.append({"type": "text", "text": prompt})

    # The tile image (high detail for precise localization)
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(send_tile)}", "detail": "high"}
    })

    raw = _openai_chat(
        messages=[
            {"role": "system", "content": _SYSTEM_DETECT},
            {"role": "user", "content": user_content}
        ],
        model=model,
        max_tokens=2048,
        temperature=0.1,
        json_mode=True,
    )

    if debug:
        print(f"\n[DEBUG tile {info.index}] raw response:\n{raw}\n")

    parsed = _parse_json(raw)

    # Persist tile description for similarity search
    tile_note = parsed.get("tile_notes", "")
    if tile_note and tile_note not in ("no columns detected", "no columns visible"):
        _save_tile_note(tile_hash, tile_note, info, file_path)

    dets = []
    for col in parsed.get("columns", []):
        bbox = col.get("bbox", [])
        if len(bbox) != 4:
            continue
        numeric = all(isinstance(v, (int, float)) for v in bbox)
        if not numeric:
            continue
        # Handle normalized coordinates (0-1 range) -- scale up to VISION_SIZE
        if all(0.0 <= v <= 1.0 for v in bbox):
            bbox = [v * VISION_SIZE for v in bbox]
        # Scale from VISION_SIZE space back to actual content dimensions
        sx   = info.width  / VISION_SIZE
        sy   = info.height / VISION_SIZE
        bbox = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]

        shape = col.get("shape", "column")
        _VALID_SHAPES = ("square", "rectangle", "round", "i_beam", "square_round", "i_square")
        if shape not in _VALID_SHAPES:
            shape = "column"

        try:
            conf = float(col.get("confidence", 0.5))
            conf = conf / 100.0 if conf > 1.0 else conf
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.5

        dets.append({"bbox_tile": bbox, "bbox_page": _to_page_bbox(bbox, info),
                     "shape": shape, "confidence": conf,
                     "notes": col.get("notes", ""),
                     "tile_hash": tile_hash,
                     "tile_index": info.index, "page_num": info.page_num})

    dets = _nms(dets)
    dets = _cell_dedup(dets, info.width, info.height, "bbox_tile", 10, 10)
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# Detection pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _nms(detections: list[dict], threshold: float = 0.1) -> list[dict]:
    """IoU-based NMS. Threshold lowered to 0.1 -- structural columns never overlap."""
    kept = []
    for det in sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True):
        bbox = det.get("bbox_page", [])
        if len(bbox) == 4 and not any(_iou(bbox, k["bbox_page"]) > threshold for k in kept):
            kept.append(det)
    return kept


def detect_file(
    path:       str | Path,
    page_num:   int  = 0,
    model:      str  = DEFAULT_MODEL,
    dpi:        int  = RENDER_DPI,
    save_image: str | Path | None = None,
    verbose:    bool = True,
) -> dict:
    """
    Detect structural columns in a PDF page or image file.

    Improved harness pipeline:
      1. Render PDF page to high-res image
      2. Generate sliding-window tiles (1280x1280, step 1080)
      3. Pre-filter: skip tiles with insufficient content (saves API cost)
      4. For each tile: resize to 1024x1024, send to OpenAI with references + corrections
      5. Parse structured JSON response, scale coords to page space
      6. Global NMS + cell dedup across all tiles
      7. Persist results to SQLite

    Results are automatically saved to detections.db.
    Returns a dict with: file, page, total_columns, detections, stats, run_id.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    if path.suffix.lower() == ".pdf":
        img, n_pages = _pdf_to_pil(path, page_num, dpi)
        if verbose:
            print(f"  PDF pages: {n_pages}  ->  page {page_num}")
    else:
        img = Image.open(path).convert("RGB")

    W, H  = img.size
    tiles = list(_tiles(img, page_num))
    if verbose:
        print(f"  Image: {W}x{H} px  |  Tiles: {len(tiles)}")

    all_dets: list[dict] = []
    _memory = _mjson_load()
    skipped = 0
    for i, (tile_img, tile_info) in enumerate(tiles):
        if verbose:
            print(f"  Tile {i+1}/{len(tiles)}  offset=({tile_info.x_offset},{tile_info.y_offset})", end="")

        # Pre-filter: skip nearly empty tiles
        if not _tile_has_content(tile_img, tile_info):
            skipped += 1
            if verbose:
                print("  [SKIP: empty]")
            continue
        if verbose:
            print()

        try:
            all_dets.extend(_detect_tile(tile_img, tile_info, model,
                                         debug=verbose, file_path=str(path),
                                         _memory=_memory))
        except Exception as e:
            if verbose:
                print(f"\n  [TILE ERROR — skipped] {e}")

    if verbose:
        print()

    final = _nms(all_dets)
    final = _cell_dedup(final, W, H, "bbox_page", 12, 12)
    for idx, det in enumerate(final, 1):
        det["id"] = idx

    by_shape = dict(Counter(d["shape"] for d in final))
    avg_conf = round(sum(d["confidence"] for d in final) / len(final), 3) if final else 0.0

    result = {
        "file": str(path), "page": page_num, "image_size": [W, H],
        "total_columns": len(final), "detections": final,
        "stats": {"by_shape": by_shape, "avg_confidence": avg_conf,
                  "tiles": len(tiles), "tiles_skipped": skipped},
        "model": model, "timestamp": datetime.now().isoformat(),
    }

    if save_image:
        draw_detections(img, final).save(str(save_image))
        result["annotated_image"] = str(save_image)
        if verbose:
            print(f"  Saved: {save_image}")

    result["run_id"] = _save_to_db(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Utility / status
# ══════════════════════════════════════════════════════════════════════════════

def get_status() -> dict:
    """Check that the OpenAI API key is valid and the model is accessible."""
    try:
        client = _get_client()
        models = client.models.list()
        model_ids = [m.id for m in models.data[:50]]
        found = DEFAULT_MODEL in model_ids or any(DEFAULT_MODEL in m for m in model_ids)
        return {"openai": "ok" if found else "model_check_ok",
                "model": DEFAULT_MODEL, "api_key_set": True}
    except Exception as e:
        return {"openai": "unavailable", "error": str(e), "model": DEFAULT_MODEL}


def get_ground_truth_images() -> list[str]:
    return sorted(str(f) for f in GT_DIR.glob("*.png")) if GT_DIR.exists() else []


def detect_ground_truth(model: str = DEFAULT_MODEL) -> dict[str, dict]:
    return {Path(p).name: detect_file(p, model=model, verbose=False)
            for p in get_ground_truth_images()}


# ══════════════════════════════════════════════════════════════════════════════
# Agentic loop
# ══════════════════════════════════════════════════════════════════════════════

def _load_skills(*names: str) -> str:
    parts = [p.read_text() for name in names
             if (p := SKILLS_DIR / f"{name}.md").exists()]
    return "\n\n---\n\n".join(parts)


_AGENT_SYSTEM = f"""\
You are an expert architectural analysis agent that detects structural columns
in floor plan PDFs and images using the OpenAI GPT-4o vision model.

Available tools -- call them with a <tool_call> block:

  detect_file(path, page_num=0, dpi=150, save_image="", model="{DEFAULT_MODEL}")
  detect_ground_truth()
  memory_search(file="", shape="", min_confidence=0.0, limit=50)
  memory_stats()
  memory_clear(run_id="")
  memory_runs(limit=20)
  references_reload()
  get_status()
  get_ground_truth_images()
  add_correction(tile_hash, action, shape="", bbox=[x1,y1,x2,y2], notes="", file_path="", page_num=0, tile_index=0)
  list_corrections(tile_hash="", file_path="", limit=100)
  search_similar_tiles(description="keyword", limit=5)

add_correction actions: confirm=detection is correct, reject=false positive, add=missed column

TOOL CALL FORMAT:
<tool_call>
{{"name": "detect_file", "arguments": {{"path": "/path/to/file.pdf", "page_num": 0}}}}
</tool_call>

Column shapes: square | rectangle | round | i_beam | square_round | i_square
Confidence: float 0.0-1.0
"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _dispatch(name: str, args: dict) -> str:
    match name:
        case "detect_file":
            r = detect_file(path=args.get("path", ""), page_num=args.get("page_num", 0),
                            model=args.get("model", DEFAULT_MODEL), dpi=args.get("dpi", RENDER_DPI),
                            save_image=args.get("save_image") or None, verbose=True)
            return json.dumps({k: r.get(k) for k in
                               ("file", "page", "total_columns", "stats", "run_id",
                                "annotated_image", "error")}, indent=2)
        case "detect_ground_truth":
            return json.dumps(detect_ground_truth(model=args.get("model", DEFAULT_MODEL)),
                              indent=2, default=str)
        case "memory_search":
            return json.dumps(memory_search(**{k: args[k] for k in args
                                               if k in ("file", "shape", "min_confidence", "limit")}),
                              indent=2)
        case "memory_stats":    return json.dumps(memory_stats(), indent=2)
        case "memory_clear":    return json.dumps(memory_clear(run_id=args.get("run_id") or None), indent=2)
        case "memory_runs":     return json.dumps(memory_runs(args.get("limit", 20)), indent=2)
        case "references_reload": return json.dumps({"loaded_shapes": references_reload()}, indent=2)
        case "get_status":      return json.dumps(get_status(), indent=2)
        case "get_ground_truth_images": return json.dumps(get_ground_truth_images(), indent=2)
        case "add_correction":  return json.dumps(add_correction(**{
                                    k: args[k] for k in args
                                    if k in ("tile_hash","action","shape","bbox","notes",
                                             "file_path","page_num","tile_index","x_offset","y_offset")}), indent=2)
        case "list_corrections": return json.dumps(list_corrections(**{
                                    k: args[k] for k in args
                                    if k in ("tile_hash","file_path","limit")}), indent=2)
        case "search_similar_tiles": return json.dumps(search_similar_tiles(
                                    args.get("description",""), args.get("limit",5)), indent=2)
        case _:                 return json.dumps({"error": f"Unknown tool: {name}"})


class AgentSession:
    """
    Stateful agent session -- maintains conversation history across calls.

        session = AgentSession()
        print(session.ask("Detect columns in /tmp/plan.pdf"))
        print(session.ask("Show me all circle columns with confidence > 80%"))
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        skills     = _load_skills("detect_columns")
        system     = _AGENT_SYSTEM + (f"\n\n# Skills\n\n{skills}" if skills else "")
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        for _ in range(MAX_TOOL_LOOPS):
            raw = _openai_chat(
                messages=self.messages,
                model=self.model,
                max_tokens=2048,
                temperature=0.3,
                json_mode=False,  # Agent needs free-form text, not forced JSON
            )
            self.messages.append({"role": "assistant", "content": raw})
            match = _TOOL_CALL_RE.search(raw)
            if not match:
                return raw.strip()
            call = {}
            try:
                call   = json.loads(match.group(1))
                result = _dispatch(call.get("name", ""), call.get("arguments", {}))
            except json.JSONDecodeError as e:
                result = json.dumps({"error": f"bad JSON: {e}"})
            print(f"  [tool] {call.get('name', '?')}  ->  {result[:80]}...")
            self.messages.append({"role": "user", "content": f"Tool result:\n{result}"})
        return "[max tool loops reached]"

    def reset(self) -> None:
        self.messages = [self.messages[0]]


def run_agent(question: str, model: str = DEFAULT_MODEL) -> str:
    """One-shot helper: create a session, ask one question, return the answer."""
    return AgentSession(model=model).ask(question)
