# MCC-Amplify-v3 — Codebase Audit Report

**Date:** 2026-04-10  
**Scope:** Read-only audit of all pipeline stages, C# Revit service, and frontend viewer.  
**Constraint:** No agent logic, memory schemas, or Revit C# service modified during this session.

---

## Critical Gap 1 — Column dimension field name mismatch (all columns placed at 200 × 200 mm)

### The Gap

Every column placed into Revit is assigned the minimum fallback size (200 mm × 200 mm) regardless of what the YOLO detector or type resolver measured. No actual column dimensions ever reach the Revit schema mapper.

### Origin

**Writer — `type_resolution_agents/column_resolver.py` lines 99–108:**

```python
est_width_mm  = round(w_px * scale_mm_per_px, 0) if scale_mm_per_px else None
est_depth_mm  = round(h_px * scale_mm_per_px, 0) if scale_mm_per_px else None
est_diam_mm   = round(((w_px + h_px) / 2) * scale_mm_per_px, 0) if scale_mm_per_px else None

resolved[i].update({
    "est_width_mm":  est_width_mm,
    "est_depth_mm":  est_depth_mm,
    "est_diam_mm":   est_diam_mm,
    ...
})
```

The resolver writes keys prefixed with `est_`.

**Reader — `translator/tools.py` lines 420–424:**

```python
if col.get("is_circular") and col.get("diameter_mm"):   # key: diameter_mm
    w = d = float(col["diameter_mm"])
else:
    w = float(col.get("width_mm") or ctx["column_min_section"])   # key: width_mm
    d = float(col.get("depth_mm") or ctx["column_min_section"])   # key: depth_mm
```

The schema mapper reads keys *without* the `est_` prefix. Both lookups miss → `col.get("width_mm")` returns `None` → fallback `ctx["column_min_section"]` = 200 mm is used for every column.

### Impact

- Affects **all columns** in every processed job.
- The YOLO smoke test on `c9bfec04-ca62-4ae1-8545-3ea2aeb89615.pdf` confirmed 372 columns detected with `width_mm=None, depth_mm=None, diameter_mm=None` at the detection stage — the resolver adds `est_width_mm` etc., none of which the mapper reads.
- Revit receives uniform 200 × 200 mm columns even when the drawing shows 400 × 600 mm or circular 500 mm sections.
- DfMA rule C2 (section 200–1500 mm) is trivially satisfied by the fallback value and cannot detect over-slender columns from real drawings.

### Concrete Fix

In `type_resolution_agents/column_resolver.py`, rename the three output keys:

```python
# BEFORE (lines 99–108):
resolved[i].update({
    "est_width_mm":  est_width_mm,
    "est_depth_mm":  est_depth_mm,
    "est_diam_mm":   est_diam_mm,
    ...
})

# AFTER:
resolved[i].update({
    "width_mm":    est_width_mm,
    "depth_mm":    est_depth_mm,
    "diameter_mm": est_diam_mm,
    ...
})
```

No changes are required in `translator/tools.py` — its key names are already correct.

If backward compatibility with memory.sqlite is needed, also rename in any SQLite INSERT inside `column_resolver.py` that stores the resolved dict.

---

## Critical Gap 2 — Dual C# Revit server projects with incompatible path handlers

### The Gap

There are two separate C# projects in `revit_server/`. The one that `build.bat` builds and deploys is **not** the one the Python client is designed to call.

| | `revit_server/csharp_service/` | `revit_server/RevitService/` |
|---|---|---|
| Server type | Raw TCP socket | `System.Net.HttpListener` |
| Handles | `POST /build` | `POST /build-model` |
| Buffer size | **8 192 bytes** (hard limit) | Stream-based — no fixed limit |
| Path Python calls | `/build-model` | `/build-model` ✓ |
| Health response | `{"status":"OK"}` for unknown paths | Plain text "Revit service healthy" |
| `build.bat` deploys? | **Yes** | No |

When `csharp_service/Program.cs` is running and Python posts to `/build-model`, the raw TCP handler returns `{"status":"OK"}` (the catch-all branch). The Python `revit_api_client` receives that, reads `status != "QUEUED"`, and logs an "Add-in not initialised" error. The `.rvt` file is never written.

Additionally, the 8 192-byte receive buffer in `Program.cs:103` means any Transaction JSON larger than ~8 KB is silently truncated, causing JSON parse errors inside Revit.

### Fix

Either:

**Option A (recommended):** Update `build.bat` to build and deploy `revit_server/RevitService/` instead of `revit_server/csharp_service/`. The `RevitService` project is the one Python expects.

**Option B:** Add a `/build-model` handler to `csharp_service/Program.cs` that mirrors the `RevitService/ApiServer.cs` implementation, and increase the receive buffer from 8 192 to at least 131 072 bytes (128 KB).

---

## Critical Gap 3 — mm-space snapping (addressed in this session)

**Status: Fixed.**

`validation/grid_snap.py` previously snapped column centres in pixel space. A 32/42 grid-lines detection miss caused a 32 % scale error, displacing every column by one grid bay. Replaced with `snap_to_grid_mm()` which converts pixel centres to approximate mm first, then snaps with `tolerance_mm = assumed_bay_mm × 0.5` (default 4 200 mm). This is robust to scale errors up to ±50 % of a bay width.

`translator/tools.py` updated to import and call the new function. The old pixel-space infrastructure (`v_lines_px`, `h_lines_px`, `v_world`, `h_world`) removed from the column transform loop.

---

## Smoke Test Results

### YOLO Column Agent — `data/uploads/c9bfec04-ca62-4ae1-8545-3ea2aeb89615.pdf`

```
Model:           yolo_detection_agents/weights/column-detect.pt  (5 MB, YOLOv11)
Input page:      0  (page 1 of PDF)
Image size:      14 042 × 9 934 px
Total columns:   372
Avg confidence:  0.957
Shape breakdown: square × 372  (all — _CLASS_TO_SHAPE maps all classes to "square")

Sample detections (first 3):
  id=1  shape=square  conf=0.965  bbox=[8249.3, 6438.0, 8283.7, 6471.4]  center=[8266.5, 6454.7]
  id=2  shape=square  conf=0.932  bbox=[7904.1, 6109.4, 7938.2, 6143.6]  center=[7921.2, 6126.5]
  id=3  shape=square  conf=0.918  bbox=[8249.7, 6114.2, 8283.8, 6147.7]  center=[8266.8, 6130.9]

Width/depth at detection stage: all None  (expected — type resolver fills these)
After type resolver:            est_width_mm / est_depth_mm populated  (but wrong key names — see Gap 1)
After schema mapper:            all columns → 200 × 200 mm  (fallback — confirms Gap 1)
```

### Grid Detection Agent — PDF render step

```
PDF → PIL Image: OK  (1 page rendered to /tmp/grid_page_*.png)
Ollama vision call: NOT run (no live Ollama instance during audit)
Grid agent output fields confirmed: total_grid_lines, vertical_labels, horizontal_labels, confidence, notes
```

Note: The grid agent does **not** emit `_grid_lines_px`. As a result, `cross_element_validator/validator.py` grid-intersection check (`_grid_lines_px` is never populated) is permanently neutral — it never quarantines any column for being off-grid.

---

## Minor Issues

| # | Location | Issue |
|---|----------|-------|
| M1 | `validation/tools.py` D1 rule | Duplicate column detection uses `round(centre[0] / 50) * 50` (pixel space) after coordinate transform has moved centres to mm space — will generate false positives or miss duplicates |
| M2 | `backend/controller.py` | `sys.path.insert/remove` pattern in `run_pipeline()` is not thread-safe; concurrent PDF uploads will corrupt `sys.path` |
| M3 | `backend/server.py` | `_jobs` is an in-memory dict — all job status is lost on uvicorn restart |
| M4 | `revit_server/RevitService/ApiServer.cs` `/health` | Returns plain text; Python `revit_api_client` attempts `json.loads()` and fails silently |
| M5 | `DEPLOYMENT.md` | Several commands reference `~/Documents/mcc-amplify-v2/` (v2 path) instead of v3 |
| M6 | `type_resolution_agents/base_resolver.py` | `pytesseract` and `cv2` both commented out in `requirements.txt`; OCR and Hough circle detection are silently disabled in production — type resolver falls back to synthetic sizing |

---

## Priority Order for Next Session

1. **Fix Gap 1** — rename `est_width_mm`→`width_mm`, `est_depth_mm`→`depth_mm`, `est_diam_mm`→`diameter_mm` in `column_resolver.py` (~3 lines). Highest ROI: fixes column sizing for every future job.
2. **Fix Gap 2** — update `build.bat` to point at `revit_server/RevitService/` or add `/build-model` handler to `csharp_service/`. Required for end-to-end `.rvt` output.
3. **Fix M1** — D1 duplicate detection should hash rounded mm coordinates, not pixel coordinates.
4. **Fix M6** — uncomment `opencv-python-headless` and `pytesseract` in `requirements.txt` (or document explicitly that type resolution is synthetic-only and the fallback is intentional).
