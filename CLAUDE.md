# MCC-Amplify-v3 — Claude Code Context

## Who & What

**User:** Joseph (josephteh97) — software engineer building an agentic BIM pipeline for a Singapore main contractor.

**Goal:** Fully automate the conversion of architectural PDF floor plans into Revit `.rvt` models that comply with Singapore DfMA standards (SS CP 65 / BCA DfMA Advisory 2021), eliminating the need for manual Revit modelling.

**Repos:**
- v3 (this repo): `https://github.com/josephteh97/mcc-amplify-v3`
- v1 reference: `https://github.com/josephteh97/mcc-amplify-ai` — the original hard-coded pipeline; used as ground truth and seed data

---

## Project Evolution

| Version | Approach | Location |
|---------|----------|----------|
| **v1** (`mcc-amplify-ai`) | Hard-coded algorithmic pipeline — brittle Revit API calls, no memory | `~/Documents/mcc-amplify-ai` |
| **v2** (`mcc-amplify-v2`) | Agentic rewrite — decentralized agents with private SQLite memory, SEA-LION column detection | `~/Documents/mcc-amplify-v2` |
| **v3** (`mcc-amplify-v3`) | Full stack — YOLO column detection, FastAPI backend, React frontend, 3D GLB viewer, WebSocket chat | `~/Documents/mcc-amplify-v3` |

---

## Pipeline Overview

```
PDF upload
  → FastAPI backend (localhost:8000)
    → [Stage 1: Detection — parallel]
        Grid Detection Agent  (SEA-LION vision via Ollama)
        YOLO Column Agent     (YOLOv11 · column-detect.pt)
    → detection_parser        (normalise → Raw Geometry JSON)
    → [Stage 2: Validation Agent]
        geometry_checker  (rules G1 G2 C1 C2 C3 D1 W1 W2)
        loop_closer       (snap open wall endpoints ≤10 px)
        → validation/memory.sqlite
    → [Stage 3: BIM-Translator Agent]
        coordinate_transformer  (pixel → world mm, grid snap)
        revit_schema_mapper     (Transaction JSON)
        revit_api_client        (POST localhost:5000/build-model, self-correct ×3)
        → translator/memory.sqlite
    → glTF Exporter            (Transaction JSON → .glb for 3D viewer)
  → .rvt saved on Windows at C:\RevitOutput\{job_id}.rvt
  → .glb saved on Linux  at data/models/gltf/{job_id}.glb
  → React frontend (localhost:5173) — 3D viewer, edit panel, download
```

**Key invariant:** `job_id` is generated once at upload (`server.py:84`) and flows unchanged through the entire pipeline. The same UUID appears in `_jobs` dict, the Linux `.rvt` path, and the Windows `C:\RevitOutput\` path.

---

## Architecture Rules (do not violate)

1. **No shared tools.py.** Each agent (`validation/`, `translator/`) has its own private `tools.py` and `memory.sqlite`. The controller (`backend/controller.py`) is the only courier between them.
2. **BaseAgent contract.** All agents inherit from `backend/base_agent.py`. The `@memory_first` decorator runs before `_process()`. The `_load_agent_tools(label, path)` method loads an agent's private tools under a unique `sys.modules` key to prevent namespace collisions.
3. **YOLO agent is a singleton.** `_YOLO_AGENT = YOLOColumnAgent()` is instantiated once at module load in `controller.py`. Do NOT instantiate it per-call — the model loads from disk on first `detect()` and must be reused.
4. **job_id flows from server.py.** `server.py` creates `job_id = str(uuid.uuid4())` and passes it to `controller.run_pipeline(job_id=job_id)`. The controller overrides the randomly generated one in `_parse_detections()`. Do not generate a new UUID in the controller or pipeline stages.
5. **Column detection is YOLO, not SEA-LION.** `pdf_detection_agent/` (Ollama vision) is archived and no longer called. The active agent is `yolo_detection_agents/column_agent.py`. Grid detection still uses SEA-LION via Ollama.

---

## Two-Machine Setup

| Machine | Role | Key Process |
|---------|------|-------------|
| **Linux (dev)** | Python backend, YOLO, Ollama, FastAPI, frontend | `uvicorn backend.server:app --port 8000` |
| **Windows (Revit server)** | Revit 2023 + C# Add-in HTTP server | `revit_server/csharp_service/build.bat` → port 5000 |

**Connectivity:** Linux calls `http://<windows-ip>:5000/build-model` (not `localhost` unless WSL on same machine). Set:
```bash
export WINDOWS_REVIT_SERVER=http://<windows-ip>:5000
export REVIT_SERVER_API_KEY=my-revit-key-2023   # must match config.json
```

**Windows Revit endpoint:** `POST /build-model`. Responds with raw `.rvt` bytes.

**Common issue:** `build.bat` polls port 49152 (raw socket) to confirm Revit started, but the HTTP API is on port 5000. `build.bat` reporting success does NOT mean port 5000 is ready. Verify with: `Invoke-WebRequest http://localhost:5000/health`.

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/server.py` | FastAPI — all REST endpoints + WebSocket chat |
| `backend/controller.py` | Pipeline orchestrator — runs all 3 stages |
| `backend/base_agent.py` | Abstract BaseAgent + `@memory_first` decorator |
| `backend/gltf_exporter.py` | Transaction JSON → GLB (trimesh) |
| `yolo_detection_agents/column_agent.py` | YOLO column detection (`YOLOColumnAgent`) |
| `yolo_detection_agents/base_yolo_agent.py` | Lazy load, PDF render (PyMuPDF), CLAHE, inference |
| `yolo_detection_agents/weights/column-detect.pt` | Trained weights — must exist before running |
| `grid-detection-agent/agent.py` | Grid label detection via SEA-LION (Ollama) |
| `validation/agent.py` | ValidationAgent — DfMA rule enforcement |
| `validation/tools.py` | geometry_checker, loop_closer, standard_thickness_lookup |
| `translator/agent.py` | BIMTranslatorAgent — self-correction loop ×3 |
| `translator/tools.py` | coordinate_transformer, revit_schema_mapper, revit_api_client |
| `translator/project_context.json` | Singapore DfMA defaults (bay widths, wall thickness, etc.) |
| `revit_server/RevitService/ApiServer.cs` | Windows HTTP server — `POST /build-model` |
| `revit_server/csharp_service/ModelBuilder.cs` | Builds Revit model from Transaction JSON |
| `revit_server/csharp_service/config.json` | Port (5000), API key, CORS |
| `revit_server/csharp_service/build.bat` | Build + deploy + launch Revit |
| `frontend/src/components/Layout.jsx` | Three-column workspace (1327 lines — main UI) |
| `frontend/src/components/UploadPanel.jsx` | Upload, status poll, downloads |
| `frontend/src/components/Viewer.jsx` | Three.js 3D GLB viewer |
| `frontend/src/components/EditPanel.jsx` | Element patch editor → triggers rebuild |
| `frontend/src/components/ChatPanel.jsx` | WebSocket AI chat (Ollama) |

---

## API Endpoints (FastAPI — localhost:8000)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/upload` | Receive PDF, return `job_id` |
| POST | `/api/process/{id}` | Start pipeline in background thread |
| GET | `/api/status/{id}` | Poll progress (0–100) and result |
| GET | `/api/download/rvt/{id}` | Serve .rvt file |
| GET | `/api/download/gltf/{id}` | Serve .glb file |
| GET/PUT | `/api/project_profile` | Load/save DfMA project context |
| GET | `/api/model/{id}/recipe` | Get validated geometry for edit panel |
| PATCH | `/api/model/{id}/recipe` | Apply single element edit |
| POST | `/api/rebuild/{id}` | Re-run translator with edited geometry |
| WS | `/ws/chat/{user_id}` | AI chat (Ollama fallback: Qwen3.5→Llama3.1→Qwen2.5) |

---

## DfMA Rules (Singapore SS CP 65)

| Code | Element | Check |
|------|---------|-------|
| G1 | Grid | Confidence ≥ 0.75 |
| G2 | Grid | Both V and H axes detected |
| C1 | Column | Shape is rectangular or circular |
| C2 | Column | Section 200–1500 mm |
| **C3** | Column | Centre coordinates present — **FATAL if missing** |
| D1 | Column | No duplicate locations |
| W1 | Wall | Interior 150–250 mm, exterior 200–350 mm |
| **W2** | Wall | Non-zero length — **FATAL if violated** |

C3 and W2 trigger a `refinement_request` and pipeline abort (no re-validation retry possible).

---

## Canonical Detection Formats

### Column (from `column_agent.py` → `controller._parse_detections()`)
```python
{
    "id":          int,
    "bbox_page":   [x1, y1, x2, y2],   # pixels on full-page image
    "shape":       "square"|"round"|"rectangle"|"i_beam"|"unknown",
    "confidence":  float,               # 0.0–1.0
    "notes":       str,                 # e.g. "yolo:column_square"
    "tile_index":  int | None,
    "page_num":    int,
    "is_circular": bool,
    "width_mm":    float | None,        # None until Revit lookup
    "depth_mm":    float | None,
    "diameter_mm": float | None,
    "type_mark":   str | None,
}
```

### Grid (from `grid-detection-agent/agent.py`)
```python
{
    "total_grid_lines":  int,
    "vertical_labels":   ["1","2","3",...],
    "horizontal_labels": ["A","B","C",...],
    "confidence":        float,
    "notes":             str,
}
```

Do NOT change these key names — they are consumed by `_parse_detections()` and the validation agent.

---

## Self-Correction Loop (Translator)

When `revit_api_client` receives a Revit error:
1. Log to `translator/memory.sqlite`
2. Query memory for past correction matching error substring
3. Fallback to hard-coded `_KNOWN_CORRECTIONS` dict
4. Apply fix to transaction JSON, retry (max 3 total)
5. If still failing after ×3 → emit `refinement_request` to controller
6. Controller runs new `ValidationAgent()` with error context injected → retry translation

Known auto-corrections:
- `"Wall cannot be created"` → remove zero-length walls
- `"extrusion error"` → clamp column to 200 mm minimum
- `"overlapping"` → deduplicate columns at same grid intersection
- `"HostObject is not valid"` → set `host_wall_id` to null
- `"Level not found"` → ensure levels list precedes elements

---

## YOLO Column Detection Setup

The model weights must be present at:
```
yolo_detection_agents/weights/column-detect.pt
```

Copy from training machine:
```bash
cp ~/Document/generate-yolo-training-datasest-columns/column-detect.pt \
   yolo_detection_agents/weights/column-detect.pt
```

Verify class names match `_CLASS_TO_SHAPE` in `column_agent.py`:
```python
from ultralytics import YOLO
m = YOLO("yolo_detection_agents/weights/column-detect.pt")
print(m.names)  # e.g. {0: 'column', 1: 'column_square', 2: 'duct', ...}
```

Smoke test:
```python
from yolo_detection_agents.column_agent import YOLOColumnAgent
agent  = YOLOColumnAgent()
result = agent.detect("path/to/floor_plan.pdf", page_num=0)
print(result["total_columns"], result["detections"][:2])
```

---

## Adding a New Element Type (e.g. Beams)

1. Train YOLO on beam annotations → `beam-detect.pt`
2. Create `yolo_detection_agents/beam_agent.py` subclassing `BaseYOLOAgent`
3. Add `_BEAM_AGENT = YOLOBeamAgent()` singleton in `controller.py`
4. Add `_run_beam_detection()` function following the same pattern as `_run_column_detection()`
5. Submit to `ThreadPoolExecutor` in Stage 1
6. Add beam field to `_parse_detections()` output dict
7. Add beam rules to `validation/tools.py` geometry_checker
8. Add beam schema to `translator/tools.py` revit_schema_mapper

---

## Running the Stack

### Linux (backend + frontend)
```bash
# Install deps
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Seed agent memories (first run only)
python seed_memory.py

# Start backend
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (separate terminal)
cd frontend && npm run dev   # → http://localhost:5173
```

### Windows (Revit server)
```bat
cd revit_server\csharp_service
build.bat
:: Wait for Revit to open, then verify:
:: http://localhost:5000/health → {"status":"healthy","revit_initialized":true}
```

### Environment variables (Linux)
```bash
export WINDOWS_REVIT_SERVER=http://<windows-lan-ip>:5000
export REVIT_SERVER_API_KEY=my-revit-key-2023
```

---

## Recent Significant Changes

- **YOLO replaces SEA-LION for columns** — `pdf_detection_agent/` is archived; `yolo_detection_agents/column_agent.py` is active. Singleton instantiated at controller module load.
- **job_id unified** — `server.py` now passes upload `job_id` to `controller.run_pipeline()`. Same UUID appears in `_jobs` dict, Linux `.rvt` path, and Windows `C:\RevitOutput\` path.
- **Revit endpoint** — Windows C# server listens on `POST /build-model` (`revit_server/RevitService/ApiServer.cs`).
- **glTF export** — Added `backend/gltf_exporter.py`; pipeline now produces `.glb` alongside `.rvt` for in-browser 3D preview.
- **Frontend edit loop** — Users can click elements in the Three.js viewer → edit geometry in EditPanel → PATCH recipe → POST rebuild without re-running detection/validation.
