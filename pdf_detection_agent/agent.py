"""
agent.py — PDF Column Detection Agent
======================================
Single-file agent: PDF tiling, Ollama vision detection, SQLite memory, agentic loop.

    from agent import AgentSession, detect_file, memory_search

    result = detect_file("floor_plan.pdf", page_num=0)
    rows   = memory_search(shape="circle", min_confidence=0.8)
    answer = AgentSession().ask("How many columns on page 0 of /path/to/plan.pdf?")
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sqlite3
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL",   "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
TILE_SIZE       = 1280
TILE_STEP       = 1080          # 200 px overlap
RENDER_DPI      = 150
MAX_TOOL_LOOPS  = 8
SKILLS_DIR      = Path(__file__).parent / "skills"
DB_PATH         = Path(__file__).parent / "detections.db"
GT_DIR          = Path(__file__).parent / "ground_truth" / "columns"
VALID_SHAPES    = {"square", "rectangle", "circle", "i_beam"}


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
        CREATE INDEX IF NOT EXISTS idx_col_run   ON columns(run_id);
        CREATE INDEX IF NOT EXISTS idx_col_shape ON columns(shape);
        CREATE INDEX IF NOT EXISTS idx_col_conf  ON columns(confidence);
        CREATE INDEX IF NOT EXISTS idx_run_file  ON runs(file_path);
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


# ══════════════════════════════════════════════════════════════════════════════
# PDF / image utilities
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TileInfo:
    index: int; page_num: int
    x_offset: int; y_offset: int
    width: int; height: int        # actual content (may be < TILE_SIZE at edges)
    page_width: int; page_height: int


def _pdf_to_pil(path: Path, page_num: int, dpi: int) -> Image.Image:
    import fitz
    doc = fitz.open(str(path))
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _pdf_page_count(path: Path) -> int:
    import fitz
    doc = fitz.open(str(path))
    n   = doc.page_count
    doc.close()
    return n


def _tiles(img: Image.Image, page_num: int) -> Generator[tuple[Image.Image, _TileInfo], None, None]:
    """Sliding-window tiling. Edge tiles are white-padded to TILE_SIZE × TILE_SIZE."""
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
    img.save(buf, format="PNG")
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
        draw.text((x1, max(0, y1 - 12)), f"{det.get('shape','?')[0].upper()} {int(conf*100)}%", fill=colour)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth reference images (few-shot visual examples)
# ══════════════════════════════════════════════════════════════════════════════

# Filename keywords → shape (first match wins; order matters)
_SHAPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("square",    ["square", "sqaure"]),   # include common typo
    ("rectangle", ["rect"]),
    ("circle",    ["round", "circle"]),
    ("i_beam",    ["i_beam", "ibeam", "i_col"]),
]

_REFERENCES: list[tuple[str, str]] | None = None   # [(shape, b64), ...] — cached


def _infer_shape(filename: str) -> str | None:
    name = filename.lower()
    for shape, keywords in _SHAPE_KEYWORDS:
        if any(kw in name for kw in keywords):
            return shape
    return None


def _load_references() -> list[tuple[str, str]]:
    """
    Scan GT_DIR for images, infer shape from filename, keep the largest per shape.
    Cached after first call — use references_reload() to refresh.
    """
    global _REFERENCES
    if _REFERENCES is not None:
        return _REFERENCES

    best: dict[str, tuple[int, Image.Image]] = {}  # shape → (area, img)
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
# Ollama vision detection
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_DETECT = """\
You are an expert architectural floor plan analyst specialised in detecting structural columns.

## How to read a floor plan
- **Grid lines** are thin dash-dot CENTRELINES running horizontally and vertically.
- **Grid balloons** are small OPEN circles at the ends of grid lines containing alphanumeric
  labels (A, B, C... / 1, 2, 3...). Do NOT confuse them with columns.
- **Structural columns** appear at or near the INTERSECTIONS of grid lines as small,
  solid, filled geometric shapes.

## Column shapes
| Shape     | Description                                              |
|-----------|----------------------------------------------------------|
| square    | Small filled black/dark square at a grid intersection    |
| rectangle | Filled dark rectangle (non-square) at a grid point       |
| circle    | Filled solid circle (round column)                       |
| i_beam    | I-beam or H-section profile symbol                       |

## What is NOT a column
- Open/hollow circles (grid balloons)
- Dimension lines, arrowheads, wall segments, door arcs
- North arrows, scale bars, title block content
"""

_USER_DETECT = """\
Examine this floor plan tile carefully and detect ALL structural columns.

Structural columns are small solid filled geometric shapes. They are commonly found at or near \
grid line intersections but may also appear along walls or independently. Do not require \
visible grid lines — focus on identifying the filled shapes themselves.

Look for:
- Small filled black/dark squares or rectangles
- Small solid filled circles (round columns)
- I-beam or H-section profile symbols
- Any small solid geometric shape that represents a load-bearing column

For every column found output:
  "bbox": [x1, y1, x2, y2]  — pixel bounding box within THIS tile
  "shape": "square" | "rectangle" | "circle" | "i_beam"
  "confidence": float 0.0–1.0
  "notes": one-line observation

Respond ONLY with valid JSON — no markdown fences, no extra text:
{
  "columns": [
    { "bbox": [x1, y1, x2, y2], "shape": "square", "confidence": 0.95, "notes": "..." }
  ],
  "tile_notes": "brief description"
}

If no columns are found: {"columns": [], "tile_notes": "no columns detected"}
"""


def _ollama(payload: dict, timeout: int = 180) -> str:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"]
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama unreachable at {OLLAMA_BASE_URL}. Run: ollama serve\n{e}")


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"columns": [], "tile_notes": f"parse_error: {text[:200]}"}


def _detect_tile(tile: Image.Image, info: _TileInfo, model: str, debug: bool = False) -> list[dict]:
    refs     = _load_references()
    preamble = "Reference examples of real structural columns from floor plans:\n"
    for i, (shape, _) in enumerate(refs, 1):
        preamble += f"  Image {i}: {shape} column\n"
    preamble += f"\nImage {len(refs) + 1} is the floor plan tile to analyse.\n\n"

    # /no_think disables qwen3's chain-of-thought mode, which otherwise consumes the
    # entire token budget inside <think>…</think> and leaves no JSON output.
    no_think = "/no_think\n" if "qwen3" in model.lower() else ""
    raw    = _ollama({"model": model, "stream": False,
                      "options": {"temperature": 0.2, "num_predict": 4096},
                      "messages": [{"role": "system", "content": _SYSTEM_DETECT},
                                   {"role": "user", "content": no_think + preamble + _USER_DETECT,
                                    "images": [b64 for _, b64 in refs] + [_pil_to_b64(tile)]}]})
    if debug:
        print(f"\n[DEBUG tile {info.index}] raw response:\n{raw}\n")
    dets = []
    for col in _parse_json(raw).get("columns", []):
        bbox = col.get("bbox", [])
        if len(bbox) != 4:
            continue
        shape = col.get("shape", "square")
        if shape not in VALID_SHAPES:
            shape = "square"
        try:
            conf = float(col.get("confidence", 0.5))
            conf = conf / 100.0 if conf > 1.0 else conf
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.5
        dets.append({"bbox_tile": bbox, "bbox_page": _to_page_bbox(bbox, info),
                     "shape": shape, "confidence": conf, "notes": col.get("notes", ""),
                     "tile_index": info.index, "page_num": info.page_num})
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


def _nms(detections: list[dict], threshold: float = 0.3) -> list[dict]:
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
    Results are automatically saved to detections.db.
    Returns a dict with: file, page, total_columns, detections, stats, run_id.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    if path.suffix.lower() == ".pdf":
        if verbose:
            print(f"  PDF pages: {_pdf_page_count(path)}  →  page {page_num}")
        img = _pdf_to_pil(path, page_num, dpi)
    else:
        img = Image.open(path).convert("RGB")

    W, H  = img.size
    tiles = list(_tiles(img, page_num))
    if verbose:
        print(f"  Image: {W}×{H} px  |  Tiles: {len(tiles)}")

    all_dets: list[dict] = []
    for i, (tile_img, tile_info) in enumerate(tiles):
        if verbose:
            print(f"  Tile {i+1}/{len(tiles)}  offset=({tile_info.x_offset},{tile_info.y_offset})", end="\r")
        try:
            all_dets.extend(_detect_tile(tile_img, tile_info, model, debug=verbose))
        except ConnectionError as e:
            if verbose:
                print(f"\n  [ERROR] {e}")
            return {"error": str(e)}

    if verbose:
        print()

    final = _nms(all_dets)
    for idx, det in enumerate(final, 1):
        det["id"] = idx

    by_shape: dict[str, int] = {}
    for d in final:
        by_shape[d["shape"]] = by_shape.get(d["shape"], 0) + 1
    avg_conf = round(sum(d["confidence"] for d in final) / len(final), 3) if final else 0.0

    result = {
        "file": str(path), "page": page_num, "image_size": [W, H],
        "total_columns": len(final), "detections": final,
        "stats": {"by_shape": by_shape, "avg_confidence": avg_conf, "tiles": len(tiles)},
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
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            data  = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            base  = DEFAULT_MODEL.split(":")[0]
            found = any(n == DEFAULT_MODEL or n.split(":")[0] == base for n in names)
            return {"ollama": "ok" if found else "model_not_pulled",
                    "model": DEFAULT_MODEL, "available": names}
    except Exception as e:
        return {"ollama": "unavailable", "error": str(e)}


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
in floor plan PDFs and images using a local Ollama vision model.

Available tools — call them with a <tool_call> block:

  detect_file(path, page_num=0, dpi=150, save_image="")
  detect_ground_truth()
  memory_search(file="", shape="", min_confidence=0.0, limit=50)
  memory_stats()
  memory_clear(run_id="")
  memory_runs(limit=20)
  references_reload()
  get_status()
  get_ground_truth_images()

TOOL CALL FORMAT:
<tool_call>
{{"name": "detect_file", "arguments": {{"path": "/path/to/file.pdf", "page_num": 0}}}}
</tool_call>

Column shapes: square | rectangle | circle | i_beam
Confidence: float 0.0–1.0
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
        case _:                 return json.dumps({"error": f"Unknown tool: {name}"})


class AgentSession:
    """
    Stateful agent session — maintains conversation history across calls.

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
            raw = _ollama({"model": self.model, "messages": self.messages,
                           "stream": False, "options": {"temperature": 0.3, "num_predict": 2048}})
            self.messages.append({"role": "assistant", "content": raw})
            match = _TOOL_CALL_RE.search(raw)
            if not match:
                return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            call = {}
            try:
                call   = json.loads(match.group(1))
                result = _dispatch(call.get("name", ""), call.get("arguments", {}))
            except json.JSONDecodeError as e:
                result = json.dumps({"error": f"bad JSON: {e}"})
            print(f"  [tool] {call.get('name', '?')}  →  {result[:80]}...")
            self.messages.append({"role": "tool", "content": result})
        return "[max tool loops reached]"

    def reset(self) -> None:
        self.messages = [self.messages[0]]


def run_agent(question: str, model: str = DEFAULT_MODEL) -> str:
    """One-shot helper: create a session, ask one question, return the answer."""
    return AgentSession(model=model).ask(question)
