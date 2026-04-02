"""
server.py — HTTP API for MCC-Amplify-v2
=========================================
Wraps controller.run_pipeline() behind a REST API so the React frontend
(frontend/) can upload a PDF, poll for progress, and download the .rvt.

Endpoints (all under /api/):
  POST   /api/upload                 — receive PDF, return job_id
  POST   /api/process/{job_id}       — kick off pipeline in background
  GET    /api/status/{job_id}        — poll progress / result
  GET    /api/download/rvt/{job_id}  — download the generated .rvt
  GET    /api/download/gltf/{job_id} — placeholder (glTF not yet generated)
  GET    /api/project_profile        — return default project context
  PUT    /api/project_profile        — save project context overrides
  GET    /api/model/{job_id}/recipe  — return validated geometry for EditPanel
  PATCH  /api/model/{job_id}/recipe  — apply element patch (human-in-the-loop)
  POST   /api/rebuild/{job_id}       — re-run translation after edits
  GET    /api/corrections/defaults/{type} — return memory hints for an element type
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

for _p in (str(_ROOT), str(Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import controller  # noqa: E402 — must come after sys.path setup

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MCC-Amplify-v2 API", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (keyed by job_id) ────────────────────────────────────
_jobs: dict[str, dict] = {}
_DATA_DIR = _ROOT / "data"
_UPLOAD_DIR = _DATA_DIR / "uploads"
_RVT_DIR    = _DATA_DIR / "models" / "rvt"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_RVT_DIR.mkdir(parents=True, exist_ok=True)

# Default project context — overridable via PUT /api/project_profile
_profile: dict = {}
_PROFILE_PATH = _ROOT / "translator" / "project_context.json"
if _PROFILE_PATH.exists():
    try:
        _profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Upload
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_floor_plan(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    job_id  = str(uuid.uuid4())
    save_path = _UPLOAD_DIR / f"{job_id}.pdf"
    save_path.write_bytes(await file.read())

    _jobs[job_id] = {
        "status":    "uploaded",
        "progress":  0,
        "message":   "File uploaded. Ready to process.",
        "filename":  file.filename,
        "pdf_path":  str(save_path),
        "result":    None,
        "error":     None,
        "created_at": datetime.now().isoformat(),
    }
    return {"job_id": job_id, "status": "uploaded"}


# ══════════════════════════════════════════════════════════════════════════════
# Process — runs pipeline in a background thread
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process/{job_id}")
async def process_floor_plan(job_id: str, body: dict = {}):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")
    if _jobs[job_id]["status"] == "processing":
        return {"job_id": job_id, "status": "already_processing"}

    _jobs[job_id]["status"]   = "processing"
    _jobs[job_id]["progress"] = 5
    _jobs[job_id]["message"]  = "Pipeline starting…"

    ctx = {**_profile, **body.get("project_context", {})}

    thread = threading.Thread(
        target=_run_pipeline_bg,
        args=(job_id, _jobs[job_id]["pdf_path"], ctx),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "processing"}


def _run_pipeline_bg(job_id: str, pdf_path: str, ctx: dict) -> None:
    def _prog(pct: int, msg: str):
        _jobs[job_id]["progress"] = pct
        _jobs[job_id]["message"]  = msg

    try:
        _prog(10, "Stage 1 — Detecting grid lines and columns…")
        result = controller.run_pipeline(
            pdf_path=pdf_path,
            project_context=ctx,
            page_num=0,
            verbose=False,
        )
        _prog(90, "Finalising…")

        rvt_path  = result.get("rvt_path")
        gltf_path = result.get("gltf_path")
        _jobs[job_id].update({
            "status":   "completed" if result.get("ok") else "failed",
            "progress": 100 if result.get("ok") else -1,
            "message":  "Done." if result.get("ok") else result.get("error_log", "Pipeline failed."),
            "error":    None if result.get("ok") else result.get("error_log"),
            "result": {
                "ok":      result.get("ok"),
                "files": {
                    "rvt":  rvt_path,
                    "gltf": gltf_path,
                },
                "stats": {
                    "grids":   result.get("element_counts", {}).get("grids", 0),
                    "columns": result.get("element_counts", {}).get("columns", 0),
                    "walls":   result.get("element_counts", {}).get("walls", 0),
                },
                "warnings":   result.get("warnings", []),
                "error_log":  result.get("error_log"),
                "timings_s":  result.get("timings_s", {}),
                "raw_geometry": result.get("raw_geometry"),
                "validated_payload": result.get("validated_payload"),
            },
        })
    except Exception as exc:
        _jobs[job_id].update({
            "status":   "failed",
            "progress": -1,
            "message":  str(exc),
            "error":    str(exc),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Status
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")
    return _jobs[job_id]


# ══════════════════════════════════════════════════════════════════════════════
# Downloads
# ══════════════════════════════════════════════════════════════════════════════

def _serve_job_file(job_id: str, file_key: str, extension: str, media_type: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    path = (job.get("result") or {}).get("files", {}).get(file_key)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"{extension.upper()} file not yet available.")
    filename = f"{Path(job.get('filename', 'model.pdf')).stem}_{job_id}.{extension}"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/download/rvt/{job_id}")
def download_rvt(job_id: str):
    return _serve_job_file(job_id, "rvt", "rvt", "application/octet-stream")


@app.get("/api/download/gltf/{job_id}")
def download_gltf(job_id: str):
    return _serve_job_file(job_id, "gltf", "glb", "model/gltf-binary")


# ── Stub endpoints so the frontend doesn't spam 404s ─────────────────────────

@app.get("/api/chat/models")
def chat_models():
    return {"models": []}


@app.websocket("/ws/chat/{user_id}")
async def ws_chat(websocket: WebSocket, user_id: str):
    # Chat agent not implemented in v2 — accept then close so browser stops retrying
    await websocket.accept()
    await websocket.close(code=1001)


# ══════════════════════════════════════════════════════════════════════════════
# Project profile (project_context.json)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/project_profile")
def get_profile():
    return _profile


@app.put("/api/project_profile")
async def save_profile(body: dict):
    global _profile
    _profile = body
    _PROFILE_PATH.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# Recipe — validated geometry for human-in-the-loop editing
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/model/{job_id}/recipe")
def get_recipe(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    result = job.get("result") or {}
    geometry = (result.get("validated_payload") or {}).get("geometry") or {}
    return {
        "grids":   geometry.get("grid", {}),
        "columns": geometry.get("columns", []),
        "walls":   geometry.get("walls", []),
        "doors":   geometry.get("doors", []),
        "windows": geometry.get("windows", []),
    }


@app.patch("/api/model/{job_id}/recipe")
async def patch_recipe(job_id: str, body: dict):
    """Apply a single element edit. Stored in memory for the next rebuild."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    element_type  = body.get("element_type")
    element_index = body.get("element_index")
    patch         = body.get("patch", {})

    result   = job.get("result") or {}
    geometry = (result.get("validated_payload") or {}).get("geometry") or {}
    elements = geometry.get(element_type, [])

    if not isinstance(elements, list) or element_index >= len(elements):
        raise HTTPException(400, f"Invalid element_type or index.")

    elements[element_index].update(patch)
    return {"ok": True, "updated": elements[element_index]}


@app.post("/api/rebuild/{job_id}")
async def rebuild(job_id: str):
    """Re-run BIM translation with the current (possibly patched) geometry."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    result   = job.get("result") or {}
    geometry = (result.get("validated_payload") or {}).get("geometry") or {}
    ctx      = {**_profile}

    job["status"]   = "processing"
    job["progress"] = 50
    job["message"]  = "Rebuilding with edits…"

    thread = threading.Thread(
        target=_run_pipeline_bg,
        args=(job_id, job["pdf_path"], ctx),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "status": "processing"}


# ══════════════════════════════════════════════════════════════════════════════
# Correction defaults (memory hints for EditPanel)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/corrections/defaults/{element_type}")
def correction_defaults(element_type: str):
    try:
        from translator.agent import BIMTranslatorAgent
        agent = BIMTranslatorAgent()
        patterns = agent._tools.memory_io.query_patterns(element_type, outcome="success")
        return {"element_type": element_type, "patterns": patterns[:5]}
    except Exception:
        return {"element_type": element_type, "patterns": []}


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "service": "mcc-amplify-v2"}


if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
