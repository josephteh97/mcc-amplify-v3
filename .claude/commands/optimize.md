---
description: Analyze code for performance bottlenecks and suggest optimizations
allowed-tools: Read, Grep, Glob, Bash
---

# Performance Optimization Analysis

Analyze the following for performance issues: $ARGUMENTS

## Step 1 — Identify Target Files

- If a file or path is given in `$ARGUMENTS`, analyze that directly.
- If no argument is given, run `git diff --name-only HEAD~1` to find the most recently changed files and analyze those.
- Read each relevant file **fully** before making any suggestions.

## Step 2 — Identify Bottlenecks

Check for issues in these categories, with project-specific patterns for this codebase:

### Algorithmic Complexity
- O(n²) or worse loops that could be flattened or replaced
- Redundant iterations over the same data (e.g. scanning `detections[]` twice)
- Missing early exits / short-circuits in rule checks (`geometry_checker`)

### Memory
- Large objects held in memory unnecessarily (e.g. full PDF pages in `base_yolo_agent.py`)
- Dict copies where in-place mutation or references would suffice
- Accumulating results without streaming (e.g. building large `transaction_json` in one shot)

### I/O & Network
- Repeated SQLite queries inside a loop — should be batched with `WHERE IN (...)`
- Synchronous Revit API calls that block the pipeline thread unnecessarily
- Redundant file reads (e.g. reading `memory.sqlite` multiple times per agent run)
- PDF rendered multiple times for the same page (check `base_yolo_agent.py` and `grid-detection-agent/tools.py`)

### Python-Specific
- `for` loops where list comprehensions or `map()` would be faster
- Missing `numpy` vectorization in coordinate transforms (`translator/tools.py`)
- SQLite: missing indexes on `feature_signature`, `element_type`, `success_count` columns
- `json.dumps` / `json.loads` round-trips that can be avoided
- Agent instantiated per-call instead of as a module-level singleton

### Concurrency
- Sequential operations that are independent and could run in `ThreadPoolExecutor`
- Blocking I/O (SQLite writes, file saves) on the FastAPI event loop thread
- The `_run_pipeline_bg()` thread in `server.py` — check if any sub-stages could be parallelized beyond the current Stage 1 grid+column parallel

### Frontend (React / Three.js)
- Missing `useMemo` / `useCallback` on expensive computations in `Layout.jsx`
- GLB reload on every status poll instead of only when the file changes
- WebSocket message handling — check for state updates that trigger unnecessary re-renders
- Three.js scene not disposed on component unmount (memory leak)

## Step 3 — Output Format

For each issue found, provide:

1. **File + line number** — exact reference (e.g. `translator/tools.py:142`)
2. **Problem** — what is slow and why (include complexity class if relevant)
3. **Fix** — concrete code snippet showing the improvement
4. **Impact** — `high` / `medium` / `low` with a one-line reason

---

## Step 4 — Priority List

Finish with a **Top 3 changes to make first**, ranked by impact-to-effort ratio. Focus on changes that improve end-to-end pipeline latency for a typical floor plan PDF, since that is the primary user-facing metric.

---

## Project Context (read before analyzing)

This is a PDF-to-BIM pipeline. The hot path per pipeline run is:

```
PDF upload → Stage 1 (parallel: grid detect + YOLO column detect)
           → Stage 2 (validation: geometry_checker + loop_closer)
           → Stage 3 (translation: coordinate_transformer + revit_schema_mapper + revit_api_client → Revit HTTP call)
           → glTF export
```

**Known slow points to check first:**
- `base_yolo_agent.py` — PDF renders at 150 DPI; YOLO inference on full-page image; CLAHE applied per run
- `grid-detection-agent/agent.py` — Ollama vision call (external HTTP, can be 5–30s)
- `revit_api_client` in `translator/tools.py` — synchronous HTTP POST to Windows Revit server, 300s timeout
- `validation/tools.py` `loop_closer` — iterates all wall pairs to find open endpoints; O(n²) if many walls
- `translator/tools.py` `coordinate_transformer` — grid snap iterates all grid intersections per column

**Singletons already in place** (do not flag as issues):
- `_YOLO_AGENT = YOLOColumnAgent()` — model loaded once at `controller.py` module init
- `ValidationAgent` and `BIMTranslatorAgent` — re-instantiated per pipeline run (intentional for memory isolation)
