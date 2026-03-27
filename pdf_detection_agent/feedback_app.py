#!/usr/bin/env python3
"""
feedback_app.py — Interactive tile-by-tile feedback UI for the column detection agent.

Usage:
    python feedback_app.py
    python feedback_app.py --pdf /path/to/file.pdf --page 0

Opens a Gradio web interface at http://localhost:7860
"""
import argparse
import io
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import agent

# ── Colour scheme ─────────────────────────────────────────────────────────────
_COLOURS = {
    "pending":   (255, 200, 0),    # yellow  — not yet reviewed
    "confirmed": (0, 200, 80),     # green   — user confirmed correct
    "rejected":  (220, 60, 60),    # red     — user rejected
    "new":       (80, 160, 255),   # blue    — user-added correction
}
_SHAPE_OPTIONS = ["square", "rectangle", "round", "i_beam", "square_round", "i_square"]

# ── Session state (pure Python dict, passed via gr.State) ─────────────────────

def _empty_state() -> dict:
    return {
        "file_path": "",
        "page_num":  0,
        "model":     agent.DEFAULT_MODEL,
        "tiles":     [],          # list of (PIL_Image, TileInfo)
        "tile_idx":  0,
        "dets":      [],          # list of agent detection dicts for current tile
        "reviews":   {},          # det index → "confirmed" | "rejected"
        "tile_hash": "",
    }


# ── Image rendering ───────────────────────────────────────────────────────────

def _render_tile(tile_img: Image.Image, dets: list[dict], reviews: dict,
                 highlight_idx: int | None = None,
                 new_bbox: list[float] | None = None) -> Image.Image:
    """Draw detection bboxes on the tile image. Returns annotated PIL image."""
    out  = tile_img.copy()
    draw = ImageDraw.Draw(out)

    for i, det in enumerate(dets):
        bbox = det.get("bbox_tile", det.get("bbox_page", []))
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        state   = reviews.get(i, "pending")
        colour  = _COLOURS[state]
        width   = 4 if i == highlight_idx else 2
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)

        label = f"[{i+1}] {det.get('shape','?')[:3]} {int(det.get('confidence',0)*100)}%"
        draw.rectangle([x1, max(0, y1-14), x1+len(label)*7+4, y1], fill=colour)
        draw.text((x1+2, max(0, y1-14)), label, fill=(0, 0, 0))

        if state == "rejected":
            draw.line([x1, y1, x2, y2], fill=_COLOURS["rejected"], width=2)
            draw.line([x2, y1, x1, y2], fill=_COLOURS["rejected"], width=2)

    if new_bbox and len(new_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in new_bbox]
        draw.rectangle([x1, y1, x2, y2], outline=_COLOURS["new"], width=3)
        draw.text((x1+2, max(0, y1-14)), "NEW", fill=_COLOURS["new"])

    return out


def _tile_to_display(tile_img: Image.Image) -> Image.Image:
    """Scale tile for display — moondream sends 320×320 content-cropped tiles."""
    # Content was cropped to info.width × info.height, then resized to 320×320 for model.
    # For display we show the original tile (1280×1280 or smaller) at a sensible size.
    w, h = tile_img.size
    max_dim = 640
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        tile_img = tile_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return tile_img


# ── Core actions ──────────────────────────────────────────────────────────────

def load_file(file_path: str, page_num: int, model: str, state: dict):
    """Load PDF/image and build tile list. Does NOT run detection yet."""
    path = Path(file_path.strip())
    if not path.exists():
        return state, f"❌ File not found: {path}", None, "—", _det_table([])

    state = _empty_state()
    state["file_path"] = str(path)
    state["page_num"]  = int(page_num)
    state["model"]     = model.strip() or agent.DEFAULT_MODEL

    # Rasterise / open image
    if path.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        pix = doc[int(page_num)].get_pixmap(
            matrix=fitz.Matrix(agent.RENDER_DPI/72, agent.RENDER_DPI/72),
            colorspace=fitz.csRGB)
        doc.close()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path).convert("RGB")

    tiles = list(agent._tiles(img, int(page_num)))
    state["tiles"] = tiles
    info_msg = f"✅ Loaded {len(tiles)} tile(s)  |  {img.size[0]}×{img.size[1]} px"
    return state, info_msg, None, _tile_label(state), _det_table([])


def detect_current_tile(state: dict):
    """Run vision model on current tile and store detections."""
    tiles = state.get("tiles", [])
    idx   = state.get("tile_idx", 0)
    if not tiles:
        return state, None, "⚠️ Load a file first.", _det_table([])

    tile_img, tile_info = tiles[idx]
    model     = state.get("model", agent.DEFAULT_MODEL)
    file_path = state.get("file_path", "")

    dets = agent._detect_tile(tile_img, tile_info, model, debug=False, file_path=file_path)
    for d in dets:
        d.setdefault("bbox_tile", d.get("bbox_page", []))

    state["dets"]      = dets
    state["reviews"]   = {}
    state["tile_hash"] = agent._tile_hash(
        tile_img.crop((0, 0, tile_info.width, tile_info.height)).resize((320, 320), Image.LANCZOS)
        if "moondream" in model.lower()
        else tile_img.crop((0, 0, tile_info.width, tile_info.height)).resize((640, 640), Image.LANCZOS)
    )

    display_tile = _tile_to_display(tile_img)
    annotated    = _render_tile(display_tile, _scale_dets_for_display(dets, tile_img, display_tile),
                                state["reviews"])
    msg = f"🔍 Detected {len(dets)} column(s)  |  tile_hash: {state['tile_hash']}"
    return state, annotated, msg, _det_table(dets)


def _scale_dets_for_display(dets, orig_tile, display_tile):
    """Scale bbox_tile coords to display_tile resolution."""
    ow, oh = orig_tile.size
    dw, dh = display_tile.size
    scaled = []
    for d in dets:
        bbox = d.get("bbox_tile", [])
        if len(bbox) == 4:
            sx, sy = dw / ow, dh / oh
            d = dict(d)
            d["bbox_tile"] = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
        scaled.append(d)
    return scaled


def confirm_detection(det_idx_str: str, state: dict):
    """Mark a detection as confirmed (correct)."""
    dets = state.get("dets", [])
    if not dets:
        return state, None, "⚠️ No detections to confirm.", _det_table(dets)
    try:
        idx = int(det_idx_str) - 1
    except (ValueError, TypeError):
        return state, None, "⚠️ Select a detection first.", _det_table(dets)
    if not (0 <= idx < len(dets)):
        return state, None, f"⚠️ Detection {idx+1} not found.", _det_table(dets)

    det  = dets[idx]
    tile_img, tile_info = state["tiles"][state["tile_idx"]]

    agent.add_correction(
        tile_hash  = state["tile_hash"],
        action     = "confirm",
        shape      = det.get("shape"),
        bbox       = det.get("bbox_page", []),
        notes      = det.get("notes", ""),
        file_path  = state["file_path"],
        page_num   = state.get("page_num", 0),
        tile_index = tile_info.index,
        x_offset   = tile_info.x_offset,
        y_offset   = tile_info.y_offset,
    )
    state["reviews"][idx] = "confirmed"

    display_tile = _tile_to_display(tile_img)
    annotated = _render_tile(display_tile,
                             _scale_dets_for_display(dets, tile_img, display_tile),
                             state["reviews"], highlight_idx=idx)
    msg = f"✅ Detection [{idx+1}] confirmed and saved to memory."
    return state, annotated, msg, _det_table(dets, state["reviews"])


def reject_detection(det_idx_str: str, state: dict):
    """Mark a detection as rejected (false positive)."""
    dets = state.get("dets", [])
    if not dets:
        return state, None, "⚠️ No detections to reject.", _det_table(dets)
    try:
        idx = int(det_idx_str) - 1
    except (ValueError, TypeError):
        return state, None, "⚠️ Select a detection first.", _det_table(dets)
    if not (0 <= idx < len(dets)):
        return state, None, f"⚠️ Detection {idx+1} not found.", _det_table(dets)

    det  = dets[idx]
    tile_img, tile_info = state["tiles"][state["tile_idx"]]

    agent.add_correction(
        tile_hash  = state["tile_hash"],
        action     = "reject",
        shape      = det.get("shape"),
        bbox       = det.get("bbox_page", []),
        notes      = "rejected by user",
        file_path  = state["file_path"],
        page_num   = state.get("page_num", 0),
        tile_index = tile_info.index,
        x_offset   = tile_info.x_offset,
        y_offset   = tile_info.y_offset,
    )
    state["reviews"][idx] = "rejected"

    display_tile = _tile_to_display(tile_img)
    annotated = _render_tile(display_tile,
                             _scale_dets_for_display(dets, tile_img, display_tile),
                             state["reviews"], highlight_idx=idx)
    msg = f"✗ Detection [{idx+1}] rejected and saved to memory."
    return state, annotated, msg, _det_table(dets, state["reviews"])


def add_manual_correction(x1: float, y1: float, x2: float, y2: float,
                           shape: str, notes: str, state: dict):
    """Save a user-drawn bounding box as a new 'add' correction."""
    if not state.get("tile_hash"):
        return state, None, "⚠️ Detect a tile first.", _det_table(state.get("dets", []))
    if not (x1 < x2 and y1 < y2):
        return state, None, "⚠️ Invalid bbox: x1 < x2 and y1 < y2 required.", _det_table(state.get("dets", []))

    tile_img, tile_info = state["tiles"][state["tile_idx"]]
    bbox_page = agent._to_page_bbox([x1, y1, x2, y2], tile_info)

    agent.add_correction(
        tile_hash  = state["tile_hash"],
        action     = "add",
        shape      = shape,
        bbox       = bbox_page,
        notes      = notes or f"user-added {shape}",
        file_path  = state["file_path"],
        page_num   = state.get("page_num", 0),
        tile_index = tile_info.index,
        x_offset   = tile_info.x_offset,
        y_offset   = tile_info.y_offset,
    )

    # Add to dets list so it shows on the image
    new_det = {"bbox_tile": [x1, y1, x2, y2], "bbox_page": bbox_page,
               "shape": shape, "confidence": 1.0,
               "notes": notes or f"user-added {shape}", "tile_hash": state["tile_hash"]}
    state["dets"].append(new_det)
    new_idx = len(state["dets"]) - 1
    state["reviews"][new_idx] = "confirmed"

    dets = state["dets"]
    display_tile = _tile_to_display(tile_img)
    annotated = _render_tile(display_tile,
                             _scale_dets_for_display(dets, tile_img, display_tile),
                             state["reviews"])
    msg = f"➕ New {shape} column added at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}] — saved to memory."
    return state, annotated, msg, _det_table(dets, state["reviews"])


def nav_tile(direction: int, state: dict):
    """Navigate to prev/next tile."""
    tiles = state.get("tiles", [])
    if not tiles:
        return state, None, "⚠️ Load a file first.", _det_table([])
    state["tile_idx"] = max(0, min(len(tiles)-1, state["tile_idx"] + direction))
    state["dets"]     = []
    state["reviews"]  = {}
    state["tile_hash"] = ""
    tile_img, _ = tiles[state["tile_idx"]]
    return state, _tile_to_display(tile_img), _tile_label(state), _det_table([])


def preview_bbox(x1: float, y1: float, x2: float, y2: float, state: dict):
    """Show a blue preview box on the current tile image."""
    tiles = state.get("tiles", [])
    if not tiles:
        return None
    tile_img, _ = tiles[state["tile_idx"]]
    dets  = state.get("dets", [])
    display_tile = _tile_to_display(tile_img)
    new_bbox = [x1, y1, x2, y2] if (x1 < x2 and y1 < y2) else None
    return _render_tile(display_tile,
                        _scale_dets_for_display(dets, tile_img, display_tile),
                        state.get("reviews", {}), new_bbox=new_bbox)


def memory_summary() -> str:
    mem   = agent._mjson_load()
    stats = agent.memory_stats()
    n_corr = sum(len(v) for v in mem["corrections"].values())
    return (f"**memory.json:** {len(mem['corrections'])} tile(s)  |  {n_corr} correction(s)\n\n"
            f"**SQLite:** {stats['total_runs']} run(s)  |  {stats['total_columns']} column(s) detected")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tile_label(state: dict) -> str:
    tiles = state.get("tiles", [])
    if not tiles:
        return "No file loaded"
    idx = state["tile_idx"]
    _, info = tiles[idx]
    return (f"Tile {idx+1}/{len(tiles)}  |  offset ({info.x_offset}, {info.y_offset})"
            f"  |  content {info.width}×{info.height} px")


def _det_table(dets: list[dict], reviews: dict | None = None) -> list[list]:
    reviews = reviews or {}
    rows = []
    for i, d in enumerate(dets):
        status = reviews.get(i, "pending")
        icon   = {"pending": "⏳", "confirmed": "✅", "rejected": "✗"}.get(status, "⏳")
        bb = [int(v) for v in d.get("bbox_page", d.get("bbox_tile", []))]
        rows.append([i+1, d.get("shape","?"), f"{d.get('confidence',0):.0%}", str(bb), icon])
    return rows


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(default_pdf: str = "", default_page: int = 0,
             default_model: str = agent.DEFAULT_MODEL) -> gr.Blocks:

    with gr.Blocks(title="Column Detection Feedback") as app:
        state = gr.State(_empty_state())

        gr.Markdown("# 🏗️ Column Detection — Feedback Interface\n"
                    "Detect columns tile-by-tile, confirm/reject each box, draw corrections. "
                    "Every action saves immediately to `memory.json` and SQLite.")

        # ── Top row: file loading ─────────────────────────────────────────────
        with gr.Row():
            pdf_input   = gr.Textbox(value=default_pdf, label="PDF or image path",
                                     placeholder="/path/to/floor_plan.pdf", scale=4)
            page_input  = gr.Number(value=default_page, label="Page", minimum=0,
                                    maximum=99, precision=0, scale=1)
            model_input = gr.Textbox(value=default_model, label="Model", scale=2)
            load_btn    = gr.Button("📂 Load", scale=1, variant="secondary")

        load_status = gr.Markdown("*No file loaded*")

        gr.Markdown("---")

        # ── Main area ─────────────────────────────────────────────────────────
        with gr.Row():

            # Left column: tile display + navigation
            with gr.Column(scale=3):
                gr.Markdown("### Tile View")
                gr.Markdown("*Hover over the image to read pixel coordinates (shown bottom-left). "
                            "Use those values in the correction inputs on the right.*")
                tile_img_out = gr.Image(label="Current tile", type="pil",
                                        interactive=False, height=500)
                tile_label   = gr.Markdown("*Load a file to begin*")

                with gr.Row():
                    prev_btn    = gr.Button("◀ Prev tile",    scale=1)
                    detect_btn  = gr.Button("🔍 Detect tile",  scale=2, variant="primary")
                    next_btn    = gr.Button("Next tile ▶",    scale=1)

                detect_status = gr.Markdown("")

            # Right column: detections + feedback
            with gr.Column(scale=2):
                gr.Markdown("### Detections")
                det_table = gr.Dataframe(
                    headers=["#", "Shape", "Conf", "Bbox (page px)", "Status"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False, wrap=True,
                )

                with gr.Row():
                    det_selector = gr.Number(label="Detection #", value=1, minimum=1,
                                             precision=0, scale=1)
                    confirm_btn  = gr.Button("✅ Correct", variant="primary", scale=1)
                    reject_btn   = gr.Button("✗ Wrong",   variant="stop",    scale=1)

                gr.Markdown("---")
                gr.Markdown("### ➕ Add / correct a bounding box")
                gr.Markdown("Hover over the tile image to read coordinates, then enter them below.")

                with gr.Row():
                    x1_in = gr.Number(label="x1", value=0, precision=0, scale=1)
                    y1_in = gr.Number(label="y1", value=0, precision=0, scale=1)
                    x2_in = gr.Number(label="x2", value=100, precision=0, scale=1)
                    y2_in = gr.Number(label="y2", value=100, precision=0, scale=1)

                with gr.Row():
                    shape_drop  = gr.Dropdown(_SHAPE_OPTIONS, value="square",
                                              label="Shape", scale=2)
                    corr_notes  = gr.Textbox(label="Notes (optional)", scale=3)

                with gr.Row():
                    preview_btn = gr.Button("👁 Preview",       scale=1, variant="secondary")
                    add_btn     = gr.Button("💾 Save Correction", scale=2, variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Memory status")
                mem_status  = gr.Markdown(memory_summary())
                refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

        # ── Wire events ───────────────────────────────────────────────────────

        load_btn.click(
            load_file,
            inputs=[pdf_input, page_input, model_input, state],
            outputs=[state, load_status, tile_img_out, tile_label, det_table],
        )

        detect_btn.click(
            detect_current_tile,
            inputs=[state],
            outputs=[state, tile_img_out, detect_status, det_table],
        )

        confirm_btn.click(
            confirm_detection,
            inputs=[det_selector, state],
            outputs=[state, tile_img_out, detect_status, det_table],
        )

        reject_btn.click(
            reject_detection,
            inputs=[det_selector, state],
            outputs=[state, tile_img_out, detect_status, det_table],
        )

        add_btn.click(
            add_manual_correction,
            inputs=[x1_in, y1_in, x2_in, y2_in, shape_drop, corr_notes, state],
            outputs=[state, tile_img_out, detect_status, det_table],
        )

        preview_btn.click(
            preview_bbox,
            inputs=[x1_in, y1_in, x2_in, y2_in, state],
            outputs=[tile_img_out],
        )

        prev_btn.click(
            lambda s: nav_tile(-1, s),
            inputs=[state],
            outputs=[state, tile_img_out, tile_label, det_table],
        )

        next_btn.click(
            lambda s: nav_tile(+1, s),
            inputs=[state],
            outputs=[state, tile_img_out, tile_label, det_table],
        )

        refresh_btn.click(lambda: memory_summary(), outputs=[mem_status])

        # Update memory status after any save action
        for btn in [confirm_btn, reject_btn, add_btn]:
            btn.click(lambda: memory_summary(), outputs=[mem_status])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",   default="", help="PDF or image path to pre-load")
    parser.add_argument("--page",  type=int, default=0)
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()

    app = build_ui(default_pdf=args.pdf, default_page=args.page, default_model=args.model)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)
