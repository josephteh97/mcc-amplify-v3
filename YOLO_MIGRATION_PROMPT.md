# Claude Prompt: Replace pdf_detection_agent with Per-Element YOLO Agents

## Context

You are working across two repositories:

- **Repo 1 (mcc-amplify-ai):** https://github.com/josephteh97/mcc-amplify-ai
  - A 7-stage hybrid AI pipeline (PDF → BIM). Uses a single YOLOv11 model (`yolov11_floorplan.pt`) to detect all elements (wall, door, window, column, room, fixture) in one pass via `backend/services/core/orchestrator.py:_run_yolo()`.
  - YOLO output format: `{"type": str, "bbox": [x1,y1,x2,y2], "confidence": float}` in pixel space.

- **Repo 2 (mcc-amplify-v3):** https://github.com/josephteh97/mcc-amplify-v3
  - A multi-agent PDF-to-BIM pipeline. Currently uses `pdf_detection_agent/` which detects columns using **Ollama vision models** (SEA-LION) via tile-based sliding window (1280×1280 px tiles, 200 px overlap). This is slow, requires Ollama running, and depends on a large vision-language model.
  - The pipeline is orchestrated by `backend/controller.py` which runs grid detection and column detection **in parallel**, then passes results to `validation/` and `translator/`.

## Current Architecture of mcc-amplify-v3 (Relevant to Migration)

```
PDF Upload
    ↓
backend/controller.py
    ├── _run_grid_detection()     → grid-detection-agent/agent.py (Ollama vision)
    └── _run_column_detection()  → pdf_detection_agent/agent.py  ← REPLACE THIS
         ↓
    Output: { detections: [{bbox, center_px, confidence, shape}, ...], total_columns, ... }
         ↓
validation/agent.py  (DfMA rule checks: C1-C3, G1-G2, D1, W1-W2)
         ↓
translator/agent.py  (pixel → mm, Revit Transaction JSON, Revit API)
         ↓
.rvt + .glb output
```

### What pdf_detection_agent currently outputs (canonical format expected by controller.py)
```python
{
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],   # pixels, absolute on full-page image
            "center_px": [cx, cy],
            "confidence": 0.87,
            "shape": "square" | "round" | "rectangle" | "i_beam" | "unknown"
        }
    ],
    "total_columns": int,
    "image_size": [width, height],
    "page": int,
    "model": str
}
```

## The Migration Plan

### Goal
Replace `pdf_detection_agent/` with a **per-element YOLO agent** architecture.

- Each construction element type gets its own dedicated YOLO model (its own `.pt` weights file and its own agent module).
- **Phase 1 (NOW):** Replace column detection with `column-detect.pt` (YOLOv11, trained on structural columns).
- Future phases: add `beam-detect.pt`, `wall-detect.pt`, `door-detect.pt`, etc.

### YOLO Model Available
- **File:** `~/Document/generate-yolo-training-datasest-columns/column-detect.pt` (Linux path)
- **Framework:** Ultralytics YOLOv11
- **Task:** Detect structural columns in floor plan images
- **Training data:** MCC construction structural drawings (Singapore DfMA context)
- **Reference training config:** `mcc_construction/yolo11_columns/args.yaml` (imgsz=640, task=detect)

### New Architecture to Build

```
pdf_detection_agent/          ← REPLACE ENTIRELY with:
yolo_detection_agents/
    __init__.py
    base_yolo_agent.py        ← Abstract base: load .pt, run inference, normalize output
    column_agent.py           ← Uses column-detect.pt, handles column-specific post-processing
    (future: beam_agent.py, wall_agent.py, ...)
    weights/
        column-detect.pt      ← Copy from ~/Document/generate-yolo-training-datasest-columns/
```

### How to Integrate into controller.py

Replace the call to `pdf_detection_agent` in `backend/controller.py:_run_column_detection()`:

```python
# OLD (remove):
from pdf_detection_agent.agent import detect_file
result = detect_file(pdf_path, page_num)

# NEW:
from yolo_detection_agents.column_agent import YOLOColumnAgent
agent = YOLOColumnAgent(weights_path="yolo_detection_agents/weights/column-detect.pt")
result = agent.detect(pdf_path, page_num=page_num)
```

The output contract from `YOLOColumnAgent.detect()` must match the **existing canonical format** above so that `validation/agent.py` receives the same data structure — no changes needed downstream.

### Key Implementation Details

#### base_yolo_agent.py
- Load weights via `ultralytics.YOLO(weights_path)`
- Render PDF page to image via `pdf2image` or `fitz` (PyMuPDF) at 150–200 DPI
- Apply CLAHE contrast enhancement (same as mcc-amplify-ai's `_enhance_for_yolo()`)
- Run inference: `results = model(image, verbose=False)`
- Handle tiling if image > 1280px (use `imgsz=1280` or tile with overlap like old agent)
- Return normalized detections in canonical format

#### column_agent.py
- Inherits from `BaseYOLOAgent`
- Post-processing: map YOLO class name → `shape` field (e.g., class `column_square` → `"square"`, `column_round` → `"round"`)
- Filter by confidence threshold (default 0.35)
- Merge overlapping boxes (NMS already done by YOLO, but deduplicate near-identical centers)
- Preserve `detections.db` SQLite memory for correction history (same schema as old agent)
- Preserve `memory.json` fast correction cache (human feedback loop must still work)

#### Preserve Human Feedback Loop
The old `pdf_detection_agent/feedback_app.py` and correction memory (`add_correction()`, `list_corrections()`) must continue to work. The new column agent should import and use the same `detections.db` schema.

### What NOT to Change
- `grid-detection-agent/` — keep as-is (still Ollama vision, grid lines are different from element detection)
- `validation/agent.py` and `validation/tools.py` — no changes (receives same canonical format)
- `translator/agent.py` — no changes
- `backend/server.py` API endpoints — no changes
- Frontend — no changes

### Reference: How mcc-amplify-ai Does YOLO (mcc-amplify-ai/backend/services/core/orchestrator.py)

```python
def _run_yolo(self, image_data: dict) -> list:
    enhanced = self._enhance_for_yolo(image_data["image"])
    results = self.yolo(enhanced, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "type":       r.names[int(box.cls)],
                "bbox":       box.xyxy[0].tolist(),
                "confidence": float(box.conf),
            })
    return detections

def _enhance_for_yolo(self, image) -> np.ndarray:
    # Convert PIL → numpy, apply CLAHE in LAB space, return numpy
    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

Adapt this pattern for the per-element agent.

### Acceptance Criteria
1. `YOLOColumnAgent.detect(pdf_path, page_num)` returns the canonical detection format.
2. `backend/controller.py` calls the new agent with no changes to the downstream pipeline.
3. Human correction workflow (`detections.db`, `memory.json`, `feedback_app.py`) still works.
4. No Ollama dependency for column detection — runs fully offline with local `.pt` weights.
5. Detection speed is faster than current tile-based Ollama approach (< 5 seconds per page on CPU).
6. Unit test: load `column-detect.pt`, run on a sample PDF, assert detections list is non-empty.

## File Paths Quick Reference

| Component | Path in mcc-amplify-v3 |
|-----------|------------------------|
| Pipeline orchestrator | `backend/controller.py` |
| OLD column detection agent | `pdf_detection_agent/agent.py` |
| OLD grid detection agent | `grid-detection-agent/agent.py` |
| Validation agent | `validation/agent.py` |
| BIM translator | `translator/agent.py` |
| Abstract base agent | `backend/base_agent.py` |
| NEW YOLO agent dir (to create) | `yolo_detection_agents/` |
| NEW column weights (to copy) | `yolo_detection_agents/weights/column-detect.pt` |
| Source weights (Linux) | `~/Document/generate-yolo-training-datasest-columns/column-detect.pt` |
| Reference YOLO impl (repo 1) | `mcc-amplify-ai/backend/services/core/orchestrator.py` |
