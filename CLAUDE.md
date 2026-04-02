# MCC-Amplify-v3 — Claude Code Context

## Project Overview

PDF floor plan → BIM (.rvt + .glb) pipeline using a multi-agent architecture.

**Three-stage pipeline:**
```
PDF → [Stage 1: Detection] → [Stage 2: Validation] → [Stage 3: BIM Translation] → .rvt + .glb
```

**Repos:**
- This repo (v3): https://github.com/josephteh97/mcc-amplify-v3
- Reference repo (v1/AI): https://github.com/josephteh97/mcc-amplify-ai

---

## Current Migration Task — YOLO Column Detection

### What was done
`pdf_detection_agent/` (Ollama vision-based, slow, requires running LLM server) has been
**replaced** with a per-element YOLO agent at `yolo_detection_agents/`.

Files created/modified:
- `yolo_detection_agents/__init__.py` — package
- `yolo_detection_agents/base_yolo_agent.py` — abstract base: PDF render, CLAHE, YOLO inference, NMS
- `yolo_detection_agents/column_agent.py` — `YOLOColumnAgent` backed by `column-detect.pt`
- `backend/controller.py` — `_run_column_detection()` now calls `YOLOColumnAgent` directly
- `requirements.txt` — added `ultralytics`, `opencv-python-headless`, `pymupdf`

### What still needs to be done on this machine

#### 1. Copy the model weights (Linux → this repo)
```bash
# From a Linux terminal / WSL:
cp ~/Document/generate-yolo-training-datasest-columns/column-detect.pt \
   <path-to-repo>/yolo_detection_agents/weights/column-detect.pt
```
Or via WSL on Windows:
```powershell
cp \\wsl$\Ubuntu\home\<user>\Document\generate-yolo-training-datasest-columns\column-detect.pt `
   C:\MyDocuments\mcc-amplify-v3\yolo_detection_agents\weights\column-detect.pt
```

#### 2. Install dependencies
```bash
pip install ultralytics>=8.3.0 opencv-python-headless>=4.9.0 pymupdf>=1.24.0
# or just:
pip install -r requirements.txt
```

#### 3. Verify YOLO class names
The model `column-detect.pt` was trained on `columns-and-ducts-detection-1`.
Check what class names the model actually uses:
```python
from ultralytics import YOLO
m = YOLO("yolo_detection_agents/weights/column-detect.pt")
print(m.names)   # e.g. {0: 'column', 1: 'duct', ...}
```
Then update `_CLASS_TO_SHAPE` in `yolo_detection_agents/column_agent.py` to match.

#### 4. Smoke test
```python
from yolo_detection_agents.column_agent import YOLOColumnAgent
agent  = YOLOColumnAgent()
result = agent.detect("path/to/any/floor_plan.pdf", page_num=0)
print(result["total_columns"], result["detections"][:2])
```

---

## Key File Map

| File | Purpose |
|------|---------|
| `backend/controller.py` | Pipeline orchestrator (Stage 1→2→3) |
| `backend/server.py` | FastAPI HTTP server + WebSocket chat |
| `backend/base_agent.py` | Abstract BaseAgent (Validation, Translator inherit) |
| `yolo_detection_agents/base_yolo_agent.py` | Abstract YOLO agent: render, enhance, infer |
| `yolo_detection_agents/column_agent.py` | Column detection via column-detect.pt |
| `yolo_detection_agents/weights/column-detect.pt` | **Must copy from Linux** |
| `grid-detection-agent/agent.py` | Grid line detection (Ollama vision — unchanged) |
| `pdf_detection_agent/` | Old Ollama column agent — **no longer called by pipeline** |
| `pdf_detection_agent/detections.db` | Human correction memory (still written by column_agent.py) |
| `pdf_detection_agent/feedback_app.py` | Human feedback UI (still works, no changes needed) |
| `validation/agent.py` | DfMA geometry validation |
| `translator/agent.py` | Pixel→mm + Revit Transaction JSON |
| `translator/project_context.json` | Singapore DfMA defaults |

---

## Canonical Detection Format

`controller._parse_detections()` reads `column_result["detections"]` and expects each item:

```python
{
    "id":          int,
    "bbox_page":   [x1, y1, x2, y2],   # pixels, full-page image
    "shape":       "square"|"round"|"rectangle"|"i_beam"|"unknown",
    "confidence":  float,               # 0.0–1.0
    "notes":       str,
    "tile_index":  int | None,
    "page_num":    int,
    "is_circular": bool,
    "width_mm":    float | None,
    "depth_mm":    float | None,
    "diameter_mm": float | None,
    "type_mark":   str | None,
}
```

`column_agent.py` already produces this format. Do NOT change the keys.

---

## Per-Element YOLO Pattern (for future elements)

To add beam detection when `beam-detect.pt` is available:

```python
# yolo_detection_agents/beam_agent.py
from .base_yolo_agent import BaseYOLOAgent

_WEIGHTS = Path(__file__).parent / "weights" / "beam-detect.pt"

class YOLOBeamAgent(BaseYOLOAgent):
    def __init__(self, weights_path=None, conf_threshold=0.35, render_dpi=150):
        super().__init__(weights_path or _WEIGHTS, conf_threshold, render_dpi)

    def _postprocess(self, raw_detections, page_num, img_w, img_h):
        # map YOLO classes → canonical beam dicts, run self._nms(...)
        ...
```

Then expose via `__init__.py` and wire into `controller.py` parallel stage.

---

## Environment

- Platform: Windows 10 (dev), Linux (model training)
- Backend: Python 3.11+, FastAPI
- YOLO: Ultralytics YOLOv11
- PDF rendering: PyMuPDF (fitz) preferred, pdf2image fallback
- Vision LLM (grid detection only): Ollama at http://localhost:11434
- Revit: Windows .NET add-in via HTTP at localhost
