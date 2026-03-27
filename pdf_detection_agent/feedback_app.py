#!/usr/bin/env python3
"""
feedback_app.py — Mouse-draw feedback UI for column detection.

Drawing a rectangle:
  • Click anywhere on the image → sets P1 (orange dot)
  • Click again               → sets P2, box appears instantly
  • If a detection is selected in the table → box goes to the EDIT panel
  • If nothing is selected               → box goes to the ADD NEW panel
  No mode buttons needed.

Editing via sliders:
  • Select a detection → sliders pre-fill with its bbox
  • Drag any slider → image preview updates live
  • Click Save Correction

Every save → memory.json + SQLite immediately. LLM uses corrections next run.

Usage:
    python feedback_app.py
    python feedback_app.py --pdf /path/to/plan.pdf --page 0
"""
import argparse, sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import agent

# ── Colours ───────────────────────────────────────────────────────────────────
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

# ── State ─────────────────────────────────────────────────────────────────────

def _empty() -> dict:
    return {
        "file_path": "", "page_num": 0, "model": agent.DEFAULT_MODEL,
        "tiles": [], "tile_idx": 0,
        "dets": [], "reviews": {}, "selected": -1,
        "tile_hash": "", "disp_scale": 1.0,
        "p1": None,   # [x,y] first click in display coords; None = waiting for P1
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
    if preview and len(preview)==4:
        bx1,by1,bx2,by2 = [int(v) for v in preview]
        col = preview_col or _C["new"]
        draw.rectangle([bx1,by1,bx2,by2], outline=col, width=3)
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
    icons = {"pending":"⏳","confirmed":"✅","rejected":"✗"}
    return [[i+1, d.get("shape","?"), f"{d.get('confidence',0):.0%}",
             str([int(v) for v in d.get("bbox_tile",[])]),
             icons.get(reviews.get(i,"pending"),"⏳")]
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


def _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info):
    """Convert display-px coords to tile-space, page-space, and normalized 0–1."""
    bbox_tile = [x1/scale, y1/scale, x2/scale, y2/scale]
    bbox_page = agent._to_page_bbox(bbox_tile, tile_info)
    bbox_norm = [bbox_tile[0]/tile_info.width,  bbox_tile[1]/tile_info.height,
                 bbox_tile[2]/tile_info.width,  bbox_tile[3]/tile_info.height]
    return bbox_tile, bbox_page, bbox_norm


def _click_hint(state):
    if state.get("p1"):
        return "🔴 **P1 set** — click the opposite corner to complete the rectangle"
    sel = state.get("selected", -1)
    if sel >= 0:
        return f"🖱️ Click image to redraw bbox for detection [{sel+1}], or drag sliders below"
    return "🖱️ Click image twice to draw a new detection (P1 → P2)"

# ── Actions ───────────────────────────────────────────────────────────────────

def load_file(path, page, model, state):
    path = Path(path.strip())
    if not path.exists():
        return state, f"❌ Not found: {path}", None, _tile_lbl(state), _det_rows([],{})
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
           disp, _tile_lbl(s), _det_rows([],{})


def detect_tile(state):
    if not state.get("tiles"):
        return state, None, "⚠️ Load a file first.", _det_rows([],{}), "", _click_hint(state)
    tile_img, tile_info = state["tiles"][state["tile_idx"]]
    model = state.get("model", agent.DEFAULT_MODEL)
    dets  = agent._detect_tile(tile_img, tile_info, model, debug=False,
                               file_path=state["file_path"])
    state["tile_hash"]  = dets[0]["tile_hash"] if dets else agent._tile_hash(
        tile_img.crop((0, 0, tile_info.width, tile_info.height)).resize((640, 640), Image.LANCZOS))
    state["dets"]       = dets
    state["reviews"]    = {}
    state["selected"]   = -1
    state["p1"]         = None
    w, h = tile_img.size
    state["disp_scale"] = min(1.0, _MAX_DISP / max(w, h))
    return (state, _render(state),
            f"🔍 {len(dets)} detection(s)  |  hash: {state['tile_hash']}",
            _det_rows(dets, {}), "", _click_hint(state))


def select_row(state, evt: gr.SelectData):
    dets = state.get("dets", [])
    idx  = evt.index[0] if evt.index else -1
    _u   = gr.update
    if not (0 <= idx < len(dets)):
        return (state, _render(state), "", 0.5,
                _u(), _u(), _u(), _u(), "square", "", _click_hint(state))
    det  = dets[idx]
    state["selected"] = idx
    state["p1"]       = None
    tile_img_sel      = state["tiles"][state["tile_idx"]][0]
    _, scale          = _disp(tile_img_sel)
    state["disp_scale"] = scale
    bbox = det.get("bbox_tile", [0,0,100,100])
    db   = [int(v * scale) for v in bbox]
    conf = det.get("confidence", 0.5)
    info = (f"**Detection [{idx+1}]** — {det.get('shape','?')} | "
            f"conf {conf:.0%} | tile px {[int(v) for v in bbox]}")
    return (state, _render(state), info, round(conf,2),
            _u(value=db[0]), _u(value=db[1]), _u(value=db[2]), _u(value=db[3]),
            det.get("shape","square"), det.get("notes",""), _click_hint(state))


def on_image_click(state, evt: gr.SelectData):
    """
    Two-click rectangle drawing.
    Click 1 → stores P1 (shows orange dot).
    Click 2 in edit mode  → auto-saves bbox to memory immediately (confirmed).
    Click 2 in add mode   → fills the Add New sliders; press Save New to commit.
    No mode buttons required.
    """
    if not state.get("tiles"):
        return (state, None, "Load a file first.",
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                _click_hint(state),
                gr.update(), gr.update(), gr.update())

    x, y = int(evt.index[0]), int(evt.index[1])
    _u   = gr.update

    if state.get("p1") is None:
        state["p1"] = [x, y]
        return (state, _render(state), f"📍 P1 = ({x},{y}) — now click the opposite corner",
                _u(), _u(), _u(), _u(),
                _u(), _u(), _u(), _u(),
                _click_hint(state),
                _u(), _u(), _u())

    p1    = state["p1"]
    x1,y1 = min(p1[0],x), min(p1[1],y)
    x2,y2 = max(p1[0],x), max(p1[1],y)
    state["p1"] = None
    sel   = state.get("selected", -1)

    if sel >= 0:
        dets   = state.get("dets", [])
        det    = dets[sel]
        scale  = state.get("disp_scale", 1.0)
        shape  = det.get("shape", "square")
        conf   = det.get("confidence", 1.0)
        tile_img, tile_info = state["tiles"][state["tile_idx"]]
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
        coords = (_u(value=x1), _u(value=y1), _u(value=x2), _u(value=y2))
        return (state, _render(state), save_msg_txt,
                *coords, _u(), _u(), _u(), _u(),
                _click_hint(state),
                _u(value=save_msg_txt), _det_rows(dets, state["reviews"]), _mem_md())
    else:
        img  = _render(state, preview=[x1,y1,x2,y2], preview_col=_C["new"])
        msg  = f"✏️ Box [{x1},{y1},{x2},{y2}] → new detection. Set shape and Save New."
        coords = (_u(value=x1), _u(value=y1), _u(value=x2), _u(value=y2))
        return (state, img, msg,
                _u(), _u(), _u(), _u(), *coords,
                _click_hint(state),
                _u(), _u(), _u())


def live_preview(x1, y1, x2, y2, state, col=None):
    """Redraw image as user drags sliders — gives instant visual feedback."""
    if col is None:
        col = _C["edit"] if state.get("selected", -1) >= 0 else _C["new"]
    if x1 < x2 and y1 < y2:
        return _render(state, preview=[x1,y1,x2,y2], preview_col=col)
    return _render(state)


def save_correction(judgement, conf, x1, y1, x2, y2, shape, notes, state):
    dets = state.get("dets", [])
    idx  = state.get("selected", -1)
    if idx < 0 or not dets:
        return state, _render(state), "⚠️ Select a detection row first.", _det_rows(dets, state.get("reviews",{})), _mem_md()
    if x1 >= x2 or y1 >= y2:
        return state, _render(state), "⚠️ x1 must be < x2 and y1 < y2.", _det_rows(dets, state.get("reviews",{})), _mem_md()

    tile_img, tile_info = state["tiles"][state["tile_idx"]]
    scale               = state.get("disp_scale", 1.0)
    bbox_tile, bbox_page, bbox_norm = _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info)
    action    = "confirm" if judgement == "correct" else "reject"

    agent.add_correction(
        tile_hash=state["tile_hash"], action=action, shape=shape,
        bbox=bbox_norm, notes=notes or f"{action}|conf={conf:.2f}",
        file_path=state["file_path"], page_num=state.get("page_num",0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    dets[idx]["confidence"] = conf
    dets[idx]["bbox_tile"]  = bbox_tile
    dets[idx]["bbox_page"]  = bbox_page
    dets[idx]["shape"]      = shape
    state["reviews"][idx]   = "confirmed" if action == "confirm" else "rejected"

    icon = "✅" if action=="confirm" else "✗"
    return (state, _render(state),
            f"{icon} [{idx+1}] saved — {action} | {shape} | conf {conf:.0%}",
            _det_rows(dets, state["reviews"]), _mem_md())


def save_new(x1, y1, x2, y2, shape, conf, notes, state):
    if not state.get("tile_hash"):
        return state, None, "⚠️ Detect a tile first.", _det_rows(state.get("dets",[]),{}), _mem_md()
    if x1 >= x2 or y1 >= y2:
        return state, _render(state), "⚠️ x1 must be < x2 and y1 < y2.", _det_rows(state.get("dets",[]),state.get("reviews",{})), _mem_md()

    tile_img, tile_info = state["tiles"][state["tile_idx"]]
    scale               = state.get("disp_scale", 1.0)
    bbox_tile, bbox_page, bbox_norm = _disp_to_bboxes(x1, y1, x2, y2, scale, tile_info)

    agent.add_correction(
        tile_hash=state["tile_hash"], action="add", shape=shape,
        bbox=bbox_norm, notes=notes or f"user-added {shape}|conf={conf:.2f}",
        file_path=state["file_path"], page_num=state.get("page_num",0),
        tile_index=tile_info.index, x_offset=tile_info.x_offset, y_offset=tile_info.y_offset,
    )
    new_det = {"bbox_tile": bbox_tile, "bbox_page": bbox_page,
               "shape": shape, "confidence": conf, "notes": notes or f"user-added {shape}"}
    state["dets"].append(new_det)
    new_idx = len(state["dets"])-1
    state["reviews"][new_idx] = "confirmed"
    state["selected"] = new_idx

    return (state, _render(state),
            f"➕ New {shape} added — saved to memory. LLM uses this next run.",
            _det_rows(state["dets"], state["reviews"]), _mem_md())


def nav(direction, state):
    tiles = state.get("tiles",[])
    if not tiles:
        return state, None, _tile_lbl(state), _det_rows([],{})
    state["tile_idx"]  = max(0, min(len(tiles)-1, state["tile_idx"]+direction))
    state["dets"]      = []
    state["reviews"]   = {}
    state["selected"]  = -1
    state["tile_hash"] = ""
    state["p1"]        = None
    tile_img, _        = tiles[state["tile_idx"]]
    disp_img, scale    = _disp(tile_img)
    state["disp_scale"] = scale
    return state, disp_img, _tile_lbl(state), _det_rows([],{})

# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui(default_pdf="", default_page=0, default_model=agent.DEFAULT_MODEL):
    with gr.Blocks(title="Column Detection Feedback") as app:
        state = gr.State(_empty())

        gr.Markdown(
            "# 🏗️ Column Detection Feedback\n"
            "**Click image twice** to draw a rectangle (P1 → P2). "
            "If a detection row is selected the box edits it; otherwise it adds a new one. "
            "Sliders give live preview. Every save → memory immediately."
        )

        # ── File bar ──────────────────────────────────────────────────────────
        with gr.Row():
            pdf_in   = gr.Textbox(value=default_pdf, label="PDF / image path", scale=5)
            page_in  = gr.Number(value=default_page, label="Page", minimum=0, maximum=99, precision=0, scale=1)
            model_in = gr.Textbox(value=default_model, label="Model", scale=2)
            load_btn = gr.Button("📂 Load", scale=1, variant="secondary")
        load_status = gr.Markdown("*No file loaded*")
        gr.Markdown("---")

        with gr.Row(equal_height=False):

            # ── Left: image + table ───────────────────────────────────────────
            with gr.Column(scale=3):
                click_hint = gr.Markdown(_click_hint(_empty()))
                tile_out   = gr.Image(label="Current tile — hover=coords, click=draw",
                                      type="pil", interactive=False, height=_MAX_DISP)
                tile_lbl   = gr.Markdown("*load a file*")
                with gr.Row():
                    prev_btn   = gr.Button("◀ Prev",        scale=1)
                    detect_btn = gr.Button("🔍 Detect Tile", scale=2, variant="primary")
                    next_btn   = gr.Button("Next ▶",        scale=1)
                detect_msg = gr.Markdown("")

                gr.Markdown("### Detections — **click row to select → edit right panel**")
                det_table = gr.Dataframe(
                    headers=["#","Shape","Conf","Bbox (tile px)","Status"],
                    datatype=["number","str","str","str","str"],
                    interactive=False, wrap=True, row_count=(6,"dynamic"),
                )

            # ── Right: edit + add panels ──────────────────────────────────────
            with gr.Column(scale=2):

                # EDIT panel
                gr.Markdown("### ✏️ Edit selected detection")
                sel_info  = gr.Markdown("*click a row in the table*")
                judgement = gr.Radio(["correct","wrong"], value="correct",
                                     label="Judgement", info="Is this detection correct?")
                conf_sl   = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Confidence")
                gr.Markdown("**Bbox** (display px) — drag sliders or click image for live preview")
                with gr.Row():
                    x1_sl = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="x1", scale=1)
                    y1_sl = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="y1", scale=1)
                    x2_sl = gr.Slider(0, _MAX_DISP, value=100, step=1, label="x2", scale=1)
                    y2_sl = gr.Slider(0, _MAX_DISP, value=100, step=1, label="y2", scale=1)
                shape_dd  = gr.Dropdown(_SHAPES, value="square", label="Shape")
                notes_in  = gr.Textbox(label="Notes", lines=1)
                save_btn  = gr.Button("💾 Save Correction", variant="primary")
                save_msg  = gr.Markdown("")

                gr.Markdown("---")

                # ADD NEW panel
                gr.Markdown("### ➕ Add missed detection")
                gr.Markdown("Click image (no row selected) or fill manually:")
                with gr.Row():
                    nx1 = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="x1", scale=1)
                    ny1 = gr.Slider(0, _MAX_DISP, value=0,   step=1, label="y1", scale=1)
                    nx2 = gr.Slider(0, _MAX_DISP, value=100, step=1, label="x2", scale=1)
                    ny2 = gr.Slider(0, _MAX_DISP, value=100, step=1, label="y2", scale=1)
                with gr.Row():
                    n_shape = gr.Dropdown(_SHAPES, value="square", label="Shape", scale=2)
                    n_conf  = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Conf", scale=3)
                n_notes  = gr.Textbox(label="Notes", lines=1)
                add_btn  = gr.Button("➕ Save New Detection", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Memory")
                mem_out     = gr.Markdown(_mem_md())
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

        # ── Wiring ────────────────────────────────────────────────────────────

        load_btn.click(load_file, [pdf_in, page_in, model_in, state],
                       [state, load_status, tile_out, tile_lbl, det_table])

        detect_btn.click(detect_tile, [state],
                         [state, tile_out, detect_msg, det_table, sel_info, click_hint])

        det_table.select(select_row, [state],
                         [state, tile_out, sel_info, conf_sl,
                          x1_sl, y1_sl, x2_sl, y2_sl, shape_dd, notes_in, click_hint])

        # Image click → two-click draw
        # Edit mode (row selected): second click auto-saves bbox to memory
        # Add mode (no row):        second click fills Add New sliders
        tile_out.select(on_image_click, [state],
                        [state, tile_out, detect_msg,
                         x1_sl, y1_sl, x2_sl, y2_sl,   # edit panel sliders
                         nx1,   ny1,   nx2,   ny2,      # add panel sliders
                         click_hint,
                         save_msg, det_table, mem_out]) # auto-save feedback

        # Live preview as sliders move
        for sl in [x1_sl, y1_sl, x2_sl, y2_sl]:
            sl.input(live_preview, [x1_sl, y1_sl, x2_sl, y2_sl, state], [tile_out])
        for sl in [nx1, ny1, nx2, ny2]:
            sl.input(lambda x1,y1,x2,y2,s: live_preview(x1,y1,x2,y2,s,col=_C["new"]),
                     [nx1, ny1, nx2, ny2, state], [tile_out])

        save_btn.click(save_correction,
                       [judgement, conf_sl, x1_sl, y1_sl, x2_sl, y2_sl, shape_dd, notes_in, state],
                       [state, tile_out, save_msg, det_table, mem_out])

        add_btn.click(save_new,
                      [nx1, ny1, nx2, ny2, n_shape, n_conf, n_notes, state],
                      [state, tile_out, save_msg, det_table, mem_out])

        prev_btn.click(lambda s: nav(-1, s), [state],
                       [state, tile_out, tile_lbl, det_table])
        next_btn.click(lambda s: nav(+1, s), [state],
                       [state, tile_out, tile_lbl, det_table])
        refresh_btn.click(lambda: _mem_md(), outputs=[mem_out])

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
