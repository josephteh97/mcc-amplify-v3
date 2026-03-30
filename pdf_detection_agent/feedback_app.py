#!/usr/bin/env python3
"""
feedback_app.py — Mouse-draw feedback UI for column detection.

Two-click drawing:
  P1 click → orange dot appears.  P2 click →
    • Row selected  → EDIT MODE:  bbox saved to memory immediately (confirmed)
    • No row        → ADD MODE:   fills Add New sliders → ➕ Save New Detection

Sliders give live preview before saving.  Every save → memory.json + SQLite.

Usage:
    python feedback_app.py
    python feedback_app.py --pdf /path/to/plan.pdf --page 0 --model moondream:latest
"""
import argparse, json, sys, urllib.request
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import agent

# ── Constants ─────────────────────────────────────────────────────────────────
_C = {
    "pending":   (255, 200, 0),
    "confirmed": (0, 200, 80),
    "rejected":  (220, 60, 60),
    "selected":  (80, 180, 255),
    "edit":      (255, 140, 0),
    "new":       (200, 80, 255),
    "p1":        (255, 60, 0),
}
_SHAPES   = [s for s in ("square", "rectangle", "round", "i_beam", "square_round", "i_square")
              if s in agent.VALID_SHAPES]
_MAX_DISP = 680

# ── Ollama model list ──────────────────────────────────────────────────────────

def _get_ollama_models() -> list[str]:
    """Query Ollama for available models; return sorted list of names."""
    try:
        req = urllib.request.Request(f"{agent.OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        names = sorted(m["name"] for m in data.get("models", []))
        return names or [agent.DEFAULT_MODEL]
    except Exception:
        return [agent.DEFAULT_MODEL]

# ── State ─────────────────────────────────────────────────────────────────────

def _empty() -> dict:
    return {
        "file_path": "", "page_num": 0, "model": agent.DEFAULT_MODEL,
        "tiles": [], "tile_idx": 0,
        "dets": [], "reviews": {}, "selected": -1,
        "tile_hash": "", "disp_scale": 1.0,
        "p1": None,
        "all_tile_dets": {},   # tile_idx → list[dict] cached by detect_all
    }

# ── Image helpers ─────────────────────────────────────────────────────────────

def _disp(tile_img: Image.Image) -> tuple[Image.Image, float]:
    w, h  = tile_img.size
    scale = min(1.0, _MAX_DISP / max(w, h))
    out   = tile_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS) if scale < 1 else tile_img.copy()
    return out, scale


def _draw_detections(img: Image.Image, dets, reviews, selected, scale,
                     p1=None, preview=None, preview_col=None) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(dets):
        bbox = det.get("bbox_tile", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v * scale) for v in bbox]
        is_sel = (i == selected)
        col    = _C["selected"] if is_sel else _C.get(reviews.get(i, "pending"), _C["pending"])
        lw     = 4 if is_sel else 2
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        if reviews.get(i) == "rejected":
            draw.line([x1, y1, x2, y2], fill=col, width=2)
            draw.line([x2, y1, x1, y2], fill=col, width=2)
        lbl = f"[{i+1}] {det.get('shape','?')[:3]} {int(det.get('confidence',0)*100)}%"
        tw  = len(lbl)*7+4
        draw.rectangle([x1, max(0,y1-15), x1+tw, y1], fill=col)
        draw.text((x1+2, max(0,y1-15)), lbl, fill=(0,0,0))
    if p1:
        px, py = p1
        draw.ellipse([px-7,py-7,px+7,py+7], fill=_C["p1"], outline=(255,255,0), width=2)
        draw.text((px+10, py-9), "P1 — click P2", fill=_C["p1"])
    if preview and len(preview) == 4:
        bx1, by1, bx2, by2 = [int(v) for v in preview]
        draw.rectangle([bx1, by1, bx2, by2], outline=(preview_col or _C["new"]), width=3)
    return img


def _render(state, preview=None, preview_col=None) -> Image.Image | None:
    if not state.get("tiles"):
        return None
    tile_img, _ = state["tiles"][state["tile_idx"]]
    out, scale  = _disp(tile_img)
    return _draw_detections(out, state["dets"], state["reviews"],
                            state["selected"], scale,
                            p1=state.get("p1"), preview=preview, preview_col=preview_col)


def _det_rows(dets, reviews):
    icons = {"pending": "⏳", "confirmed": "✅", "rejected": "✗"}
    return [[i+1, d.get("shape","?"), f"{d.get('confidence',0):.0%}",
             str([int(v) for v in d.get("bbox_tile", [])]),
             icons.get(reviews.get(i, "pending"), "⏳")]
            for i, d in enumerate(dets)]


def _tile_lbl(state):
    if not state.get("tiles"):
        return "*no file loaded*"
    _, ti = state["tiles"][state["tile_idx"]]
    return (f"**Tile {state['tile_idx']+1}/{len(state['tiles'])}** — "
            f"offset ({ti.x_offset},{ti.y_offset}) — {ti.width}×{ti.height} px")


def _mem_md():
    mem   = agent._mjson_load()
    stats = agent.memory_stats()
    n     = sum(len(v) for v in mem["corrections"].values())
    return (f"**memory.json** — {len(mem['corrections'])} tile(s), {n} correction(s)  \n"
            f"**SQLite** — {stats['total_runs']} run(s), {stats['total_columns']} detection(s)")


def _bbox_tile_to_norm(bbox, tile_info):
    """Convert tile-space bbox to normalized 0–1 coords."""
    return [bbox[0]/tile_info.width,  bbox[1]/tile_info.height,
            bbox[2]/tile_info.width,  bbox[3]/tile_info.height]


def _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info):
    """Convert display-px coords to tile-space, page-space, and normalized 0–1."""
    bbox_tile = [x1/scale, y1/scale, x2/scale, y2/scale]
    bbox_page = agent._to_page_bbox(bbox_tile, tile_info)
    bbox_norm = _bbox_tile_to_norm(bbox_tile, tile_info)
    return bbox_tile, bbox_page, bbox_norm


def _mode_badge(state) -> str:
    """Prominent mode indicator so the operator always knows the current draw mode."""
    sel  = state.get("selected", -1)
    p1   = state.get("p1")
    dets = state.get("dets", [])
    if sel >= 0:
        shape = dets[sel].get("shape", "?") if sel < len(dets) else "?"
        if p1:
            return f"🟠 **EDIT MODE — det [{sel+1}] ({shape})** — click P2 to save new bbox"
        return f"🔵 **EDIT MODE — det [{sel+1}] ({shape}) selected** — click image to redraw · or drag sliders"
    if p1:
        return "🟣 **ADD MODE — P1 set** — click the opposite corner to complete the box"
    return "🟣 **ADD MODE** — no row selected · click image twice (P1 → P2) or fill sliders below"


def _pending_count(state) -> int:
    dets    = state.get("dets", [])
    reviews = state.get("reviews", {})
    return sum(1 for i in range(len(dets)) if reviews.get(i, "pending") == "pending")

# ── Actions ───────────────────────────────────────────────────────────────────

def load_file(path, page, model, state):
    path = Path((path or "").strip())
    if not path.name or not path.exists():
        return state, f"❌ Not found: {path}", None, _tile_lbl(state), _det_rows([], {})
    s = _empty()
    s["file_path"] = str(path)
    s["page_num"]  = int(page)
    s["model"]     = model.strip() or agent.DEFAULT_MODEL
    if path.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        pix = doc[int(page)].get_pixmap(
            matrix=fitz.Matrix(agent.RENDER_DPI/72, agent.RENDER_DPI/72),
            colorspace=fitz.csRGB)
        doc.close()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path).convert("RGB")
    s["tiles"] = list(agent._tiles(img, int(page)))
    disp, _    = _disp(s["tiles"][0][0])
    return s, f"✅ {len(s['tiles'])} tile(s)  |  {img.size[0]}×{img.size[1]} px", \
           disp, _tile_lbl(s), _det_rows([], {})


def refresh_models():
    """Re-query Ollama and update the model dropdown choices."""
    models = _get_ollama_models()
    return gr.update(choices=models, value=models[0] if models else agent.DEFAULT_MODEL)


def detect_tile(state):
    if not state.get("tiles"):
        return state, None, "⚠️ Load a file first.", _det_rows([], {}), "", _mode_badge(state)
    tile_img, tile_info = state["tiles"][state["tile_idx"]]
    model   = state.get("model", agent.DEFAULT_MODEL)
    _memory = agent._mjson_load()
    dets    = agent._detect_tile(tile_img, tile_info, model, debug=False,
                                 file_path=state["file_path"], _memory=_memory)
    state["tile_hash"]  = dets[0]["tile_hash"] if dets else agent._tile_hash(
        tile_img.crop((0, 0, tile_info.width, tile_info.height)).resize((640, 640), Image.LANCZOS))
    state["dets"]       = dets
    state["reviews"]    = {}
    state["selected"]   = -1
    state["p1"]         = None
    _, scale = _disp(tile_img)
    state["disp_scale"] = scale
    state["all_tile_dets"][state["tile_idx"]] = dets
    return (state, _render(state),
            f"🔍 {len(dets)} detection(s)  |  hash: {state['tile_hash']}",
            _det_rows(dets, {}), "", _mode_badge(state))


def detect_all_tiles(state):
    """Generator: detect all tiles sequentially, streaming progress to the UI."""
    tiles = state.get("tiles", [])
    if not tiles:
        yield state, _render(state), "⚠️ Load a file first.", _det_rows([], {}), "", _mode_badge(state)
        return
    model   = state.get("model", agent.DEFAULT_MODEL)
    _memory = agent._mjson_load()
    total_dets = 0
    for i, (tile_img, tile_info) in enumerate(tiles):
        yield (state, _render(state),
               f"🔄 Detecting tile {i+1}/{len(tiles)}…",
               _det_rows(state.get("dets", []), state.get("reviews", {})),
               "", _mode_badge(state))
        try:
            dets = agent._detect_tile(tile_img, tile_info, model, debug=False,
                                      file_path=state["file_path"], _memory=_memory)
        except Exception:
            dets = []
        state["all_tile_dets"][i] = dets
        total_dets += len(dets)

    cur = state["tile_idx"]
    state["dets"]     = state["all_tile_dets"].get(cur, [])
    state["reviews"]  = {}
    state["selected"] = -1
    state["p1"]       = None
    if state["dets"]:
        state["tile_hash"] = state["dets"][0]["tile_hash"]
    tile_img, _ = tiles[cur]
    _, scale = _disp(tile_img)
    state["disp_scale"] = scale
    yield (state, _render(state),
           f"✅ All {len(tiles)} tiles detected — {total_dets} detection(s) total",
           _det_rows(state["dets"], {}), "", _mode_badge(state))


def select_row(state, evt: gr.SelectData):
    dets = state.get("dets", [])
    idx  = evt.index[0] if evt.index else -1
    _u   = gr.update
    if not (0 <= idx < len(dets)):
        return (state, _render(state), "", 0.5,
                _u(), _u(), _u(), _u(), "square", "", _mode_badge(state))
    det  = dets[idx]
    state["selected"] = idx
    state["p1"]       = None
    tile_img_sel      = state["tiles"][state["tile_idx"]][0]
    _, scale          = _disp(tile_img_sel)
    state["disp_scale"] = scale
    bbox = det.get("bbox_tile", [0, 0, 100, 100])
    db   = [int(v * scale) for v in bbox]
    conf = det.get("confidence", 0.5)
    info = (f"**Detection [{idx+1}]** — {det.get('shape','?')} | "
            f"conf {conf:.0%} | tile px {[int(v) for v in bbox]}")
    return (state, _render(state), info, round(conf, 2),
            _u(value=db[0]), _u(value=db[1]), _u(value=db[2]), _u(value=db[3]),
            det.get("shape", "square"), det.get("notes", ""), _mode_badge(state))


def on_image_click(state, evt: gr.SelectData):
    """
    Two-click rectangle drawing.
    Click 1 → stores P1 (shows orange dot).
    Click 2 in edit mode → auto-saves corrected bbox to memory (confirmed).
    Click 2 in add mode  → fills Add New sliders for review before saving.
    """
    if not state.get("tiles"):
        return (state, None, "Load a file first.",
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                _mode_badge(state), gr.update(), gr.update(), gr.update())

    x, y = int(evt.index[0]), int(evt.index[1])
    _u   = gr.update

    if state.get("p1") is None:
        state["p1"] = [x, y]
        return (state, _render(state), f"📍 P1 = ({x},{y}) — now click the opposite corner",
                _u(), _u(), _u(), _u(),
                _u(), _u(), _u(), _u(),
                _mode_badge(state), _u(), _u(), _u())

    p1    = state["p1"]
    x1, y1 = min(p1[0], x), min(p1[1], y)
    x2, y2 = max(p1[0], x), max(p1[1], y)
    state["p1"] = None
    sel   = state.get("selected", -1)

    if sel >= 0:
        dets   = state.get("dets", [])
        det    = dets[sel]
        scale  = state.get("disp_scale", 1.0)
        shape  = det.get("shape", "square")
        conf   = det.get("confidence", 1.0)
        _, tile_info = state["tiles"][state["tile_idx"]]
        bbox_tile, bbox_page, bbox_norm = _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info)

        agent.add_correction(
            tile_hash=state["tile_hash"], action="confirm", shape=shape,
            bbox=bbox_norm, notes=f"user-drawn bbox|conf={conf:.2f}",
            file_path=state["file_path"], page_num=state.get("page_num", 0),
            tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
        )
        dets[sel]["bbox_tile"] = bbox_tile
        dets[sel]["bbox_page"] = bbox_page
        state["reviews"][sel]  = "confirmed"

        save_msg_txt = f"✅ [{sel+1}] bbox saved to memory — {shape}"
        return (state, _render(state), save_msg_txt,
                _u(value=x1), _u(value=y1), _u(value=x2), _u(value=y2),
                _u(), _u(), _u(), _u(),
                _mode_badge(state),
                _u(value=save_msg_txt), _det_rows(dets, state["reviews"]), _mem_md())
    else:
        img = _render(state, preview=[x1, y1, x2, y2], preview_col=_C["new"])
        msg = f"✏️ Box [{x1},{y1},{x2},{y2}] — set shape below and click ➕ Save New."
        return (state, img, msg,
                _u(), _u(), _u(), _u(),
                _u(value=x1), _u(value=y1), _u(value=x2), _u(value=y2),
                _mode_badge(state), _u(), _u(), _u())


def live_preview(x1, y1, x2, y2, state, col=None):
    if col is None:
        col = _C["edit"] if state.get("selected", -1) >= 0 else _C["new"]
    if x1 < x2 and y1 < y2:
        return _render(state, preview=[x1, y1, x2, y2], preview_col=col)
    return _render(state)


def save_correction(judgement, conf, x1, y1, x2, y2, shape, notes, state):
    dets = state.get("dets", [])
    idx  = state.get("selected", -1)
    if idx < 0 or not dets:
        return state, _render(state), "⚠️ Select a detection row first.", _det_rows(dets, state.get("reviews", {})), _mem_md()
    if x1 >= x2 or y1 >= y2:
        return state, _render(state), "⚠️ x1 must be < x2 and y1 < y2.", _det_rows(dets, state.get("reviews", {})), _mem_md()

    _, tile_info = state["tiles"][state["tile_idx"]]
    scale        = state.get("disp_scale", 1.0)
    bbox_tile, bbox_page, bbox_norm = _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info)
    action = "confirm" if judgement == "correct" else "reject"

    agent.add_correction(
        tile_hash=state["tile_hash"], action=action, shape=shape,
        bbox=bbox_norm, notes=notes or f"{action}|conf={conf:.2f}",
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    dets[idx]["confidence"] = conf
    dets[idx]["bbox_tile"]  = bbox_tile
    dets[idx]["bbox_page"]  = bbox_page
    dets[idx]["shape"]      = shape
    state["reviews"][idx]   = "confirmed" if action == "confirm" else "rejected"

    icon = "✅" if action == "confirm" else "✗"
    return (state, _render(state),
            f"{icon} [{idx+1}] saved — {action} | {shape} | conf {conf:.0%}",
            _det_rows(dets, state["reviews"]), _mem_md())


def reject_quick(state):
    """One-click reject of the selected detection using its existing bbox."""
    dets = state.get("dets", [])
    idx  = state.get("selected", -1)
    if idx < 0 or not dets:
        return state, _render(state), "⚠️ Select a detection row first.", _det_rows(dets, state.get("reviews", {})), _mem_md()
    det          = dets[idx]
    _, tile_info = state["tiles"][state["tile_idx"]]
    bbox_norm    = _bbox_tile_to_norm(det.get("bbox_tile", [0,0,0,0]), tile_info)
    agent.add_correction(
        tile_hash=state["tile_hash"], action="reject", shape=det.get("shape", "square"),
        bbox=bbox_norm, notes="quick-reject",
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    state["reviews"][idx] = "rejected"
    return (state, _render(state),
            f"✗ [{idx+1}] rejected",
            _det_rows(dets, state["reviews"]), _mem_md())


def confirm_100(state):
    """Confirm selected detection with confidence forced to 1.0."""
    dets = state.get("dets", [])
    idx  = state.get("selected", -1)
    if idx < 0 or not dets:
        return state, _render(state), "⚠️ Select a detection row first.", _det_rows(dets, state.get("reviews", {})), _mem_md()
    det          = dets[idx]
    _, tile_info = state["tiles"][state["tile_idx"]]
    bbox_norm    = _bbox_tile_to_norm(det.get("bbox_tile", [0,0,0,0]), tile_info)
    agent.add_correction(
        tile_hash=state["tile_hash"], action="confirm", shape=det.get("shape", "square"),
        bbox=bbox_norm, notes="confirmed-100pct",
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    dets[idx]["confidence"] = 1.0
    state["reviews"][idx]   = "confirmed"
    return (state, _render(state),
            f"✅ [{idx+1}] confirmed at 100% confidence",
            _det_rows(dets, state["reviews"]), _mem_md())


def clear_memory_ui(state):
    """Wipe corrections table + memory.json. Detection run history (runs/columns) preserved."""
    agent.clear_all_corrections()
    state["reviews"] = {}
    return (state, _render(state),
            "🗑️ Corrections cleared — memory.json + SQLite corrections table wiped.",
            _det_rows(state.get("dets", []), {}), _mem_md())


def clear_detection_history_ui(state):
    """Wipe the runs and columns tables (detection history). Corrections preserved."""
    result = agent.memory_clear()
    state["dets"]    = []
    state["reviews"] = {}
    return (state, _render(state),
            f"🗑️ Detection history cleared — {result['runs_deleted']} run(s), {result['columns_deleted']} detection(s) removed.",
            _det_rows([], {}), _mem_md())


def confirm_all(state):
    """Confirm all pending detections for the current tile in one click."""
    dets = state.get("dets", [])
    if not dets:
        return state, _render(state), "⚠️ No detections to confirm.", _det_rows([], {}), _mem_md()
    _, tile_info = state["tiles"][state["tile_idx"]]
    pending = [(i, det) for i, det in enumerate(dets)
               if state["reviews"].get(i, "pending") == "pending"]
    if not pending:
        return state, _render(state), "✅ All detections already reviewed.", _det_rows(dets, state["reviews"]), _mem_md()
    agent.add_corrections_batch(
        tile_hash=state["tile_hash"],
        corrections=[{"action": "confirm", "shape": det.get("shape","square"),
                      "bbox": _bbox_tile_to_norm(det.get("bbox_tile",[0,0,0,0]), tile_info),
                      "notes": "bulk-confirm"} for _, det in pending],
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    for i, _ in pending:
        state["reviews"][i] = "confirmed"
    return (state, _render(state),
            f"✅ {len(pending)} detection(s) confirmed",
            _det_rows(dets, state["reviews"]), _mem_md())


def reject_all(state):
    """Reject all pending detections for the current tile in one click."""
    dets = state.get("dets", [])
    if not dets:
        return state, _render(state), "⚠️ No detections to reject.", _det_rows([], {}), _mem_md()
    _, tile_info = state["tiles"][state["tile_idx"]]
    pending = [(i, det) for i, det in enumerate(dets)
               if state["reviews"].get(i, "pending") == "pending"]
    if not pending:
        return state, _render(state), "✗ All detections already reviewed.", _det_rows(dets, state["reviews"]), _mem_md()
    agent.add_corrections_batch(
        tile_hash=state["tile_hash"],
        corrections=[{"action": "reject", "shape": det.get("shape","square"),
                      "bbox": _bbox_tile_to_norm(det.get("bbox_tile",[0,0,0,0]), tile_info),
                      "notes": "bulk-reject"} for _, det in pending],
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    for i, _ in pending:
        state["reviews"][i] = "rejected"
    return (state, _render(state),
            f"✗ {len(pending)} detection(s) rejected",
            _det_rows(dets, state["reviews"]), _mem_md())


def undo_last(state):
    """Remove the most recent correction for this tile from memory.json."""
    tile_hash = state.get("tile_hash", "")
    if not tile_hash:
        return state, _render(state), "⚠️ No tile detected yet — nothing to undo.", _mem_md()
    result = agent.undo_last_correction(tile_hash)
    if not result.get("ok"):
        return state, _render(state), "⚠️ No corrections to undo for this tile.", _mem_md()
    removed = result.get("removed", {})
    return (state, _render(state),
            f"↩ Undone: {removed.get('action','?')} | {removed.get('shape','?')}",
            _mem_md())


def save_new(x1, y1, x2, y2, shape, conf, notes, state):
    if not state.get("tile_hash"):
        return state, None, "⚠️ Detect a tile first.", _det_rows(state.get("dets", []), {}), _mem_md()
    if x1 >= x2 or y1 >= y2:
        return state, _render(state), "⚠️ x1 must be < x2 and y1 < y2.", _det_rows(state.get("dets", []), state.get("reviews", {})), _mem_md()

    _, tile_info = state["tiles"][state["tile_idx"]]
    scale        = state.get("disp_scale", 1.0)
    bbox_tile, bbox_page, bbox_norm = _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info)

    agent.add_correction(
        tile_hash=state["tile_hash"], action="add", shape=shape,
        bbox=bbox_norm, notes=notes or f"user-added {shape}|conf={conf:.2f}",
        file_path=state["file_path"], page_num=state.get("page_num", 0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    new_det = {"bbox_tile": bbox_tile, "bbox_page": bbox_page,
               "shape": shape, "confidence": conf, "notes": notes or f"user-added {shape}"}
    state["dets"].append(new_det)
    new_idx = len(state["dets"]) - 1
    state["reviews"][new_idx] = "confirmed"
    state["selected"] = new_idx

    return (state, _render(state),
            f"➕ New {shape} added — saved to memory. LLM uses this next run.",
            _det_rows(state["dets"], state["reviews"]), _mem_md())


def nav(direction, state):
    tiles = state.get("tiles", [])
    if not tiles:
        return state, None, _tile_lbl(state), _det_rows([], {}), _mode_badge(state)

    new_idx = max(0, min(len(tiles)-1, state["tile_idx"] + direction))
    if new_idx == state["tile_idx"]:
        return state, _render(state), _tile_lbl(state), _det_rows(state.get("dets", []), state.get("reviews", {})), _mode_badge(state)

    pending = _pending_count(state)
    state["tile_idx"]  = new_idx
    state["reviews"]   = {}
    state["selected"]  = -1
    state["tile_hash"] = ""
    state["p1"]        = None

    # Restore pre-computed dets from detect_all if available
    cached = state["all_tile_dets"].get(new_idx)
    if cached is not None:
        state["dets"] = cached
        if cached:
            state["tile_hash"] = cached[0]["tile_hash"]
    else:
        state["dets"] = []

    tile_img, _ = tiles[new_idx]
    disp_img, scale = _disp(tile_img)
    state["disp_scale"] = scale

    lbl = _tile_lbl(state)
    if pending > 0:
        lbl = f"⚠️ {pending} unreviewed left behind  |  {lbl}"
    if cached is not None and cached:
        lbl += f"  |  📋 {len(cached)} cached det(s)"

    return state, disp_img, lbl, _det_rows(state["dets"], {}), _mode_badge(state)

# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui(default_pdf="", default_page=0, default_model=agent.DEFAULT_MODEL):
    _models = _get_ollama_models()
    _default_model = default_model if default_model in _models else (_models[0] if _models else default_model)

    with gr.Blocks(title="Column Detection Feedback") as app:
        state = gr.State(_empty())

        gr.Markdown(
            "# 🏗️ Column Detection Feedback\n"
            "**Click image twice** (P1 → P2) to draw a box. "
            "Row selected = **EDIT** (auto-saves); no row = **ADD NEW** (fills sliders). "
            "Sliders give live preview. Every save → memory immediately."
        )

        # ── File bar ──────────────────────────────────────────────────────────
        with gr.Row():
            pdf_in   = gr.Textbox(value=default_pdf, label="PDF / image path", scale=5)
            page_in  = gr.Number(value=default_page, label="Page", minimum=0, maximum=99, precision=0, scale=1)
            model_dd = gr.Dropdown(choices=_models, value=_default_model, label="Model",
                                   allow_custom_value=True, scale=3)
            refresh_models_btn = gr.Button("↺ Models", scale=1, variant="secondary")
            load_btn = gr.Button("📂 Load", scale=1, variant="primary")
        load_status = gr.Markdown("*No file loaded*")
        gr.Markdown("---")

        with gr.Row(equal_height=False):

            # ── Left: image + table ───────────────────────────────────────────
            with gr.Column(scale=3):
                mode_badge = gr.Markdown(_mode_badge(_empty()))
                tile_out   = gr.Image(label="Tile — hover for coords · click to draw",
                                      type="pil", interactive=False, height=_MAX_DISP)
                tile_lbl   = gr.Markdown("*load a file*")

                with gr.Row():
                    prev_btn   = gr.Button("◀ Prev",           scale=1)
                    detect_btn = gr.Button("🔍 Detect Tile",    scale=2, variant="primary")
                    next_btn   = gr.Button("Next ▶",           scale=1)
                detect_all_btn = gr.Button("🔄 Detect All Tiles", variant="secondary")
                detect_msg = gr.Markdown("")

                with gr.Row():
                    confirm_all_btn = gr.Button("✅ Confirm All", variant="secondary", scale=1)
                    reject_all_btn  = gr.Button("✗ Reject All",  variant="stop",      scale=1)

                gr.Markdown("### Detections — **click a row to select it → edit in right panel**")
                det_table = gr.Dataframe(
                    headers=["#", "Shape", "Conf", "Bbox (tile px)", "Status"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False, wrap=True, row_count=(6, "dynamic"),
                )

            # ── Right: edit + add + memory panels ────────────────────────────
            with gr.Column(scale=2):

                # EDIT panel
                gr.Markdown("### ✏️ Edit selected detection")
                sel_info  = gr.Markdown("*click a row in the table*")
                judgement = gr.Radio(["correct", "wrong"], value="correct",
                                     label="Judgement", info="Is this detection correct?")
                conf_sl   = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Confidence")
                gr.Markdown("**Bbox** (display px) — drag sliders or click image for live preview")
                with gr.Row():
                    x1_sl = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="x1")
                    y1_sl = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="y1")
                    x2_sl = gr.Slider(0, _MAX_DISP, value=100, step=1, label="x2")
                    y2_sl = gr.Slider(0, _MAX_DISP, value=100, step=1, label="y2")
                shape_dd  = gr.Dropdown(_SHAPES, value="square", label="Shape")
                notes_in  = gr.Textbox(label="Notes", lines=1)
                with gr.Row():
                    save_btn      = gr.Button("💾 Save Correction",  variant="primary", scale=2)
                    confirm100_btn= gr.Button("💯 Confirm 100%",     variant="primary", scale=1)
                    reject_btn    = gr.Button("✗ Reject",            variant="stop",    scale=1)
                save_msg   = gr.Markdown("")
                undo_btn   = gr.Button("↩ Undo Last Correction", variant="secondary")

                gr.Markdown("---")

                # ADD NEW panel
                gr.Markdown("### ➕ Add missed detection")
                gr.Markdown("Click image (no row selected) to fill coords, or drag sliders:")
                with gr.Row():
                    nx1 = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="x1")
                    ny1 = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="y1")
                    nx2 = gr.Slider(0, _MAX_DISP, value=100, step=1, label="x2")
                    ny2 = gr.Slider(0, _MAX_DISP, value=100, step=1, label="y2")
                with gr.Row():
                    n_shape = gr.Dropdown(_SHAPES, value="square", label="Shape", scale=2)
                    n_conf  = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Conf", scale=3)
                n_notes  = gr.Textbox(label="Notes", lines=1)
                add_btn  = gr.Button("➕ Save New Detection", variant="primary")

                gr.Markdown("---")

                # Memory panel
                gr.Markdown("### 💾 Memory")
                gr.Markdown(
                    "*Corrections* — what the LLM learns from.  "
                    "*Detection history* — past `detect_file` runs stored in SQLite."
                )
                mem_out     = gr.Markdown(_mem_md())
                with gr.Row():
                    refresh_btn      = gr.Button("↺ Sync",                    variant="secondary", scale=1)
                    clear_mem_btn    = gr.Button("🗑️ Clear Corrections",       variant="stop",      scale=2)
                    clear_hist_btn   = gr.Button("🗑️ Clear Detection History", variant="stop",      scale=2)

        # ── Wiring ────────────────────────────────────────────────────────────

        refresh_models_btn.click(refresh_models, outputs=[model_dd])

        load_btn.click(load_file, [pdf_in, page_in, model_dd, state],
                       [state, load_status, tile_out, tile_lbl, det_table])

        detect_btn.click(detect_tile, [state],
                         [state, tile_out, detect_msg, det_table, sel_info, mode_badge])

        detect_all_btn.click(detect_all_tiles, [state],
                             [state, tile_out, detect_msg, det_table, sel_info, mode_badge])

        det_table.select(select_row, [state],
                         [state, tile_out, sel_info, conf_sl,
                          x1_sl, y1_sl, x2_sl, y2_sl, shape_dd, notes_in, mode_badge])

        tile_out.select(on_image_click, [state],
                        [state, tile_out, detect_msg,
                         x1_sl, y1_sl, x2_sl, y2_sl,
                         nx1,   ny1,   nx2,   ny2,
                         mode_badge,
                         save_msg, det_table, mem_out])

        # Live preview as sliders move
        for sl in [x1_sl, y1_sl, x2_sl, y2_sl]:
            sl.input(live_preview, [x1_sl, y1_sl, x2_sl, y2_sl, state], [tile_out])
        for sl in [nx1, ny1, nx2, ny2]:
            sl.input(lambda x1, y1, x2, y2, s: live_preview(x1, y1, x2, y2, s, col=_C["new"]),
                     [nx1, ny1, nx2, ny2, state], [tile_out])

        save_btn.click(save_correction,
                       [judgement, conf_sl, x1_sl, y1_sl, x2_sl, y2_sl, shape_dd, notes_in, state],
                       [state, tile_out, save_msg, det_table, mem_out])

        reject_btn.click(reject_quick, [state],
                         [state, tile_out, save_msg, det_table, mem_out])

        confirm100_btn.click(confirm_100, [state],
                             [state, tile_out, save_msg, det_table, mem_out])

        confirm_all_btn.click(confirm_all, [state],
                              [state, tile_out, detect_msg, det_table, mem_out])

        reject_all_btn.click(reject_all, [state],
                             [state, tile_out, detect_msg, det_table, mem_out])

        undo_btn.click(undo_last, [state],
                       [state, tile_out, save_msg, mem_out])

        add_btn.click(save_new,
                      [nx1, ny1, nx2, ny2, n_shape, n_conf, n_notes, state],
                      [state, tile_out, detect_msg, det_table, mem_out])

        prev_btn.click(lambda s: nav(-1, s), [state],
                       [state, tile_out, tile_lbl, det_table, mode_badge])
        next_btn.click(lambda s: nav(+1, s), [state],
                       [state, tile_out, tile_lbl, det_table, mode_badge])

        refresh_btn.click(lambda: _mem_md(), outputs=[mem_out])

        clear_mem_btn.click(clear_memory_ui, [state],
                            [state, tile_out, detect_msg, det_table, mem_out])

        clear_hist_btn.click(clear_detection_history_ui, [state],
                             [state, tile_out, detect_msg, det_table, mem_out])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",   default="")
    parser.add_argument("--page",  type=int, default=0)
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()
    build_ui(args.pdf, args.page, args.model).launch(
        server_name="0.0.0.0", server_port=args.port, share=False)
