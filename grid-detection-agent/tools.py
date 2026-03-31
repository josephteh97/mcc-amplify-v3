"""
All agent tool functions + SQLite / memory.json correction memory.

SQLite (grid_memory.db) — authoritative audit trail:
    runs        — one row per agent.run() call
    grid_results — labels detected in each run
    corrections — human-supplied corrections to a run

memory.json — fast per-PDF lookup cache:
    { "<pdf_hash>": { "vertical": [...], "horizontal": [...],
                      "notes": "...", "timestamp": "..." } }

The agent queries memory before Step 2 so past human corrections feed
back into the next detection as a strong context hint.
"""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image

import ollama_client
import pdf_renderer

# ── PATHS ─────────────────────────────────────────────────────────────────────

_DIR = Path(__file__).parent
DB_PATH         = _DIR / "grid_memory.db"
MEMORY_JSON_PATH = _DIR / "memory.json"
REFERENCE_SCRIPT = "/tmp/reference_approach.py"

# ── CONFIGURATION (shared with agent.py) ──────────────────────────────────────

MARGIN_FRACTION = 0.12
_reference_hint_cache: str | None = None
_schema_initialized = False


# ── DATABASE ──────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    global _schema_initialized
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if not _schema_initialized:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            pdf_path    TEXT NOT NULL,
            pdf_hash    TEXT NOT NULL,
            dpi         INTEGER,
            confidence  REAL,
            used_hint   INTEGER DEFAULT 0,
            steps       INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS grid_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            total_grid_lines  INTEGER,
            vertical_labels   TEXT,
            horizontal_labels TEXT,
            confidence        REAL,
            notes             TEXT
        );
        CREATE TABLE IF NOT EXISTS corrections (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_hash          TEXT NOT NULL,
            pdf_path          TEXT,
            vertical_labels   TEXT NOT NULL,
            horizontal_labels TEXT NOT NULL,
            notes             TEXT,
            timestamp         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_hash  ON runs(pdf_hash);
        CREATE INDEX IF NOT EXISTS idx_corr_hash  ON corrections(pdf_hash);
    """)
        _schema_initialized = True
    return con


# ── PDF HASH ──────────────────────────────────────────────────────────────────

def pdf_hash(pdf_path: str) -> str:
    """Stable 16-char hash from the first 64 KB of the PDF file."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read(65536)).hexdigest()[:16]


# ── memory.json ───────────────────────────────────────────────────────────────

def _mjson_load() -> dict:
    if MEMORY_JSON_PATH.exists():
        try:
            return json.loads(MEMORY_JSON_PATH.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return {"version": 1, "corrections": {}}


def _mjson_save(data: dict) -> None:
    MEMORY_JSON_PATH.write_text(json.dumps(data, indent=2))


def _row_to_correction(row) -> dict:
    """Convert a SQLite corrections row to the standard correction dict."""
    return {
        "vertical_labels":   json.loads(row["vertical_labels"]),
        "horizontal_labels": json.loads(row["horizontal_labels"]),
        "notes":     row["notes"] or "",
        "timestamp": row["timestamp"],
    }


# ── RENDERING TOOL ────────────────────────────────────────────────────────────

def tool_render_pdf(pdf_path: str, dpi: int = 200) -> dict:
    """Convert PDF to images. Returns image paths and page count."""
    images = pdf_renderer.pdf_to_images(pdf_path, dpi=dpi)
    return {"images": images, "page_count": len(images)}


# ── DETECTION TOOLS ───────────────────────────────────────────────────────────

def tool_detect_grid(image_path: str, prompt: str) -> dict:
    """Send a floor plan image to SEA-LION with the given prompt."""
    raw = ollama_client.query_vision(image_path, prompt)
    return _parse_json(raw)


def tool_verify(image_path: str, previous_result: dict, verification_prompt_tpl: str) -> dict:
    """Second-pass verification with previous result embedded in prompt."""
    prompt = verification_prompt_tpl.format(
        previous_result=json.dumps(previous_result, indent=2)
    )
    raw = ollama_client.query_vision(image_path, prompt)
    return _parse_json(raw)


def tool_zoom_margin(image_path: str, side: str, zoomed_prompt_tpl: str) -> dict:
    """
    Crop a margin band and query SEA-LION for labels in that band.

    Args:
        image_path: Path to the full floor plan PNG.
        side: One of "top", "bottom", "left", "right".
        zoomed_prompt_tpl: ZOOMED_PROMPT template string with {side} placeholder.

    Returns:
        {"labels": list[str]}
    """
    img = Image.open(image_path)
    w, h = img.size
    band = int(MARGIN_FRACTION * min(w, h))

    boxes = {
        "top":    (0, 0, w, band),
        "bottom": (0, h - band, w, h),
        "left":   (0, 0, band, h),
        "right":  (w - band, 0, w, h),
    }
    if side not in boxes:
        raise ValueError(f"side must be one of {list(boxes.keys())}, got: {side!r}")

    crop = img.crop(boxes[side])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"margin_{side}_")
    tmp_path = tmp.name
    crop.save(tmp_path, "PNG")
    tmp.close()

    try:
        raw = ollama_client.query_vision(tmp_path, zoomed_prompt_tpl.format(side=side))
    finally:
        os.unlink(tmp_path)

    return {"labels": _parse_json(raw, array=True)}


def tool_reconcile(full_result: dict, margin_results: dict, reconcile_prompt_tpl: str) -> dict:
    """Ask SEA-LION to reconcile full-image detection with all four margin scans."""
    prompt = reconcile_prompt_tpl.format(
        full_result=json.dumps(full_result, indent=2),
        top=margin_results.get("top", []),
        bottom=margin_results.get("bottom", []),
        left=margin_results.get("left", []),
        right=margin_results.get("right", []),
    )
    raw = ollama_client.query_text(prompt)
    return _parse_json(raw)


# ── REFERENCE TOOL ────────────────────────────────────────────────────────────

def tool_read_reference() -> str:
    """
    LAST RESORT — summarise final_success.ipynb logic as a hint for SEA-LION.
    Result is cached after first call.
    """
    global _reference_hint_cache
    if _reference_hint_cache is not None:
        return _reference_hint_cache

    if not os.path.isfile(REFERENCE_SCRIPT):
        return (
            "Reference script not available. "
            "Generate it with: jupyter nbconvert --to script "
            "~/Documents/test-grid-detector/final_success.ipynb --stdout "
            f"> {REFERENCE_SCRIPT}"
        )

    with open(REFERENCE_SCRIPT) as f:
        code = f.read()

    skip_re = re.compile("|".join([
        r"^\s*import ", r"^\s*from \w+ import", r"^\s*get_ipython\(",
        r"^\s*#\s*In\[", r"^\s*display\(", r"^\s*plt\.", r"^\s*fig\.",
        r"^\s*ax\.", r"^\s*print\(", r"^\s*IPython", r"^\s*%",
    ]))
    cleaned = re.sub(
        r"\n{3,}", "\n\n",
        "\n".join(line for line in code.splitlines() if not skip_re.match(line)),
    ).strip()

    summary_prompt = (
        "The following is Python code from a working grid line detector for "
        "construction floor plans. Summarise in 3-5 bullet points the KEY STEPS "
        "and TECHNIQUES used to identify grid line labels — focus on what a vision "
        "model should look for, not on implementation details.\n\n"
        f"```python\n{cleaned[:4000]}\n```"
    )
    _reference_hint_cache = ollama_client.query_text(summary_prompt)
    return _reference_hint_cache


# ── MEMORY TOOLS ──────────────────────────────────────────────────────────────

def tool_memory_lookup(pdf_path: str) -> dict | None:
    """
    Look up the most recent human correction for this PDF.

    Returns a dict with keys vertical_labels, horizontal_labels, notes
    if a correction exists — or None if no prior corrections.

    The agent uses this to inject a correction hint before Step 2.
    """
    phash = pdf_hash(pdf_path)
    data = _mjson_load()
    cached = data["corrections"].get(phash)
    if cached:
        return cached

    # Fall back to SQLite if memory.json was cleared or is stale
    con = _db()
    row = con.execute(
        "SELECT vertical_labels, horizontal_labels, notes, timestamp "
        "FROM corrections WHERE pdf_hash = ? ORDER BY id DESC LIMIT 1",
        (phash,)
    ).fetchone()
    con.close()
    if row:
        return _row_to_correction(row)
    return None


def tool_memory_save_run(pdf_path: str, result: dict, dpi: int, run_id: str) -> None:
    """
    Persist a completed detection run to SQLite.

    Args:
        pdf_path: Path to the source PDF.
        result:   Final result dict from agent.run().
        dpi:      DPI used for the final render.
        run_id:   UUID string for this run.
    """
    phash = pdf_hash(pdf_path)
    ts    = datetime.now().isoformat()
    con   = _db()
    with con:
        con.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, timestamp, pdf_path, pdf_hash, dpi, confidence, used_hint, steps) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, ts, pdf_path, phash, dpi,
             result.get("confidence", 0),
             1 if result.get("used_reference_hint") else 0,
             len(result.get("steps_taken", []))),
        )
        con.execute(
            "INSERT INTO grid_results "
            "(run_id, total_grid_lines, vertical_labels, horizontal_labels, confidence, notes) "
            "VALUES (?,?,?,?,?,?)",
            (run_id,
             result.get("total_grid_lines", 0),
             json.dumps(result.get("vertical_labels", [])),
             json.dumps(result.get("horizontal_labels", [])),
             result.get("confidence", 0),
             result.get("notes", "")),
        )
    con.close()


def tool_memory_add_correction(
    pdf_path: str,
    vertical_labels: list[str],
    horizontal_labels: list[str],
    notes: str = "",
) -> dict:
    """
    Record a human correction for a PDF and update both SQLite and memory.json.

    The correction is picked up by tool_memory_lookup() on the next agent.run()
    call for the same PDF, feeding it back as a strong context hint.

    Args:
        pdf_path:          Path to the PDF being corrected.
        vertical_labels:   Correct vertical (column) grid labels.
        horizontal_labels: Correct horizontal (row) grid labels.
        notes:             Optional explanation of what was wrong.

    Returns:
        {"ok": True, "pdf_hash": ..., "total_corrections": ...}
    """
    phash = pdf_hash(pdf_path)
    ts    = datetime.now().isoformat()
    v_json = json.dumps(vertical_labels)
    h_json = json.dumps(horizontal_labels)

    con = _db()
    with con:
        con.execute(
            "INSERT INTO corrections "
            "(pdf_hash, pdf_path, vertical_labels, horizontal_labels, notes, timestamp) "
            "VALUES (?,?,?,?,?,?)",
            (phash, pdf_path, v_json, h_json, notes, ts),
        )
        count = con.execute(
            "SELECT COUNT(*) FROM corrections WHERE pdf_hash = ?", (phash,)
        ).fetchone()[0]
    con.close()

    # Update memory.json cache — always reflect the latest correction
    data = _mjson_load()
    data["corrections"][phash] = {
        "vertical_labels":   vertical_labels,
        "horizontal_labels": horizontal_labels,
        "notes":    notes,
        "timestamp": ts,
    }
    _mjson_save(data)

    return {"ok": True, "pdf_hash": phash, "total_corrections": count}


def tool_memory_undo_correction(pdf_path: str) -> dict:
    """
    Remove the most recent correction for a PDF from memory.json.
    SQLite history is preserved as an audit trail.
    """
    phash = pdf_hash(pdf_path)
    data  = _mjson_load()
    if phash not in data["corrections"]:
        return {"ok": False, "msg": "no correction found for this PDF"}

    del data["corrections"][phash]

    # Restore the previous correction from SQLite if one exists
    con = _db()
    row = con.execute(
        "SELECT vertical_labels, horizontal_labels, notes, timestamp "
        "FROM corrections WHERE pdf_hash = ? ORDER BY id DESC LIMIT 1 OFFSET 1",
        (phash,)
    ).fetchone()
    con.close()
    if row:
        data["corrections"][phash] = _row_to_correction(row)

    _mjson_save(data)  # single write regardless of restore path
    return {"ok": True, "pdf_hash": phash}


def tool_memory_search(pdf_path: str | None = None, limit: int = 20) -> list[dict]:
    """
    Query stored detection runs, optionally filtered by PDF path substring.

    Returns a list of dicts with run metadata and detected labels.
    """
    where  = "WHERE r.pdf_path LIKE ? " if pdf_path else ""
    params = ([f"%{pdf_path}%"] if pdf_path else []) + [limit]
    con    = _db()
    rows   = con.execute(
        f"""SELECT r.run_id, r.timestamp, r.pdf_path, r.dpi, r.confidence,
                   g.vertical_labels, g.horizontal_labels, g.total_grid_lines
            FROM runs r JOIN grid_results g ON g.run_id = r.run_id
            {where}ORDER BY r.timestamp DESC LIMIT ?""",
        params
    ).fetchall()
    con.close()

    result = []
    for row in rows:
        d = dict(row)
        d["vertical_labels"]   = json.loads(d["vertical_labels"] or "[]")
        d["horizontal_labels"] = json.loads(d["horizontal_labels"] or "[]")
        result.append(d)
    return result


def tool_memory_corrections(pdf_path: str) -> list[dict]:
    """Return all corrections recorded for a specific PDF, newest first."""
    phash = pdf_hash(pdf_path)
    con   = _db()
    rows  = con.execute(
        "SELECT vertical_labels, horizontal_labels, notes, timestamp "
        "FROM corrections WHERE pdf_hash = ? ORDER BY id DESC",
        (phash,)
    ).fetchall()
    con.close()
    return [_row_to_correction(r) for r in rows]


def tool_memory_stats() -> dict:
    """Aggregate statistics across all stored runs and corrections."""
    con = _db()
    row = con.execute("""
        SELECT
            (SELECT COUNT(*)          FROM runs)              AS total_runs,
            (SELECT COUNT(*)          FROM corrections)        AS total_corr,
            (SELECT AVG(confidence)   FROM runs)              AS avg_conf,
            (SELECT SUM(total_grid_lines) FROM grid_results)  AS total_lines
    """).fetchone()
    con.close()
    return {
        "total_runs":           row["total_runs"]  or 0,
        "total_corrections":    row["total_corr"]  or 0,
        "avg_confidence":       round(row["avg_conf"] or 0.0, 3),
        "total_lines_detected": row["total_lines"] or 0,
    }


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _parse_json(raw: str, array: bool = False):
    """
    Extract and parse the first JSON object or array from a model response string.

    Args:
        raw:   Raw model output, possibly wrapped in markdown fences.
        array: If True, search for a JSON array; otherwise a JSON object.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    if array:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    else:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {
                "total_grid_lines": 0,
                "vertical_labels": [],
                "horizontal_labels": [],
                "confidence": 0.0,
                "notes": f"Could not parse JSON from response: {raw[:200]}",
            }
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            return {
                "total_grid_lines": 0,
                "vertical_labels": [],
                "horizontal_labels": [],
                "confidence": 0.0,
                "notes": f"JSON parse error: {exc}. Raw: {raw[:200]}",
            }
