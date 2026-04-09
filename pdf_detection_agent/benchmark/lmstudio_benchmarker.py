#!/usr/bin/env python3
"""
lmstudio_benchmarker.py — LM Studio Vision Model Benchmark
===========================================================
Tests whichever vision model is currently loaded in LM Studio against the
structural floor plan tile used in the Ollama benchmark.

Workflow (repeat for each of the 12 models):
  1. Open LM Studio → Local Server → Start Server
  2. Load a vision model in LM Studio
  3. conda run -n yolo python lmstudio_benchmarker.py
  4. Unload, load next model, repeat

Results accumulate in lmstudio_results.json (one entry per model, replaced on re-run).

Usage:
    conda run -n yolo python lmstudio_benchmarker.py
    conda run -n yolo python lmstudio_benchmarker.py --image /path/to/tile.png
    conda run -n yolo python lmstudio_benchmarker.py --model "specific-model-id"
    conda run -n yolo python lmstudio_benchmarker.py --results   # print summary only
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

from utils import TIMEOUT_SECONDS, count_elements, validate_json

# ── Config ────────────────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234/v1"
IMAGE_PATH = "/tmp/test_floorplan_tile.png"
RESULTS_JSON = Path(__file__).parent / "lmstudio_results.json"
LOG_FILE = Path(__file__).parent / "lmstudio_results.log"

DETECTION_PROMPT = """\
You are a structural engineer analysing a construction floor plan.
Detect ALL structural columns and grid lines visible in this image tile.

Return ONLY a valid JSON array of objects. Each object must have exactly these keys:
  "element_type": "column" or "grid_line"
  "coordinates": [x1, y1, x2, y2]  — pixel bounding box in this image
  "confidence": float 0.0-1.0
  "shape": for columns: "square"|"rectangle"|"round"|"i_beam"; for grid_line: "horizontal"|"vertical"

Example:
[
  {"element_type": "column", "coordinates": [120, 340, 155, 375], "confidence": 0.92, "shape": "square"},
  {"element_type": "grid_line", "coordinates": [0, 500, 1280, 500], "confidence": 0.85, "shape": "horizontal"}
]

If nothing found, return: []
Do NOT include any text outside the JSON array.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_loaded_models(client: openai.OpenAI) -> list[str]:
    try:
        return [m.id for m in client.models.list().data]
    except Exception as e:
        log(f"Cannot reach LM Studio at {LM_STUDIO_URL}: {e}")
        return []


# ── Core benchmark ────────────────────────────────────────────────────────────

def benchmark_model(client: openai.OpenAI, model_id: str, image_path: str) -> dict:
    result = {
        "model": model_id,
        "status": "pending",
        "inference_time_s": None,
        "json_valid": False,
        "raw_output": "",
        "elements": {},
        "error": "",
        "timestamp": datetime.now().isoformat(),
    }

    b64 = encode_image(image_path)
    log(f"  Inferring with {model_id}...")
    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": DETECTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=4096,
            temperature=0.1,
        )
    except openai.APITimeoutError:
        elapsed = round(time.time() - t0, 2)
        result.update(inference_time_s=elapsed, status="timeout",
                      error=f"exceeded {TIMEOUT_SECONDS}s")
        log(f"  TIMEOUT after {elapsed}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        result.update(inference_time_s=elapsed, status="error", error=str(e)[:500])
        log(f"  ERROR after {elapsed}s: {str(e)[:120]}")
        return result

    elapsed = round(time.time() - t0, 2)
    result["inference_time_s"] = elapsed

    raw = response.choices[0].message.content or ""
    result["raw_output"] = raw[:3000]

    is_valid, parsed, err = validate_json(raw)
    result["json_valid"] = is_valid
    if is_valid and parsed is not None:
        result["elements"] = count_elements(parsed)
        result["status"] = "success"
    else:
        result["error"] = err
        result["status"] = "invalid_json"

    log(f"  Done in {elapsed}s — JSON valid: {is_valid} — "
        f"elements: {result['elements'].get('total', 0)}")
    return result


# ── Persistence ───────────────────────────────────────────────────────────────

def save_result(result: dict) -> None:
    """Append/replace result in lmstudio_results.json (atomic write)."""
    existing: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            existing = json.loads(RESULTS_JSON.read_text())
        except Exception:
            pass
    existing = [r for r in existing if r.get("model") != result["model"]]
    existing.append(result)
    _tmp = RESULTS_JSON.with_suffix(".tmp")
    _tmp.write_text(json.dumps(existing, indent=2, default=str))
    _tmp.rename(RESULTS_JSON)


def load_results() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    try:
        return json.loads(RESULTS_JSON.read_text())
    except Exception:
        return []


# ── Display ───────────────────────────────────────────────────────────────────

def print_single(result: dict) -> None:
    e = result.get("elements", {})
    print(f"\n{'='*65}")
    print(f"  Model   : {result['model']}")
    print(f"  Status  : {result['status']}")
    print(f"  Time    : {result.get('inference_time_s')}s")
    print(f"  JSON    : {'VALID' if result['json_valid'] else 'INVALID'}")
    print(f"  Columns : {e.get('columns', 0)}")
    print(f"  Grid    : {e.get('grid_lines', 0)}")
    print(f"  Total   : {e.get('total', 0)}")
    if result.get("error"):
        print(f"  Error   : {result['error'][:120]}")
    raw = result.get("raw_output", "")
    if raw:
        print(f"  Output  : {raw[:300]}")
    print(f"{'='*65}\n")


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    print(f"\n{'='*85}")
    print(f"  LM STUDIO BENCHMARK — {len(results)} model(s) tested so far")
    print(f"{'='*85}")
    header = f"  {'Model':<44} {'Time':>7} {'JSON':>6} {'Cols':>5} {'Grid':>5} {'Total':>6}  {'Status'}"
    print(header)
    print(f"  {'-'*80}")
    for r in sorted(results, key=lambda x: -(x.get("elements", {}).get("total", 0))):
        model = r["model"][:43]
        t = f"{r['inference_time_s']:.1f}s" if r.get("inference_time_s") else "    -"
        jv = "VALID" if r.get("json_valid") else "   NO"
        e = r.get("elements", {})
        cols = str(e.get("columns", "-"))
        grids = str(e.get("grid_lines", "-"))
        total = str(e.get("total", "-"))
        status = r.get("status", "?")
        print(f"  {model:<44} {t:>7} {jv:>6} {cols:>5} {grids:>5} {total:>6}  {status}")
    print(f"{'='*85}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LM Studio vision benchmark")
    parser.add_argument("--image", default=IMAGE_PATH, help="Test image path")
    parser.add_argument("--model", default=None,
                        help="Model ID to test (default: auto-detect from LM Studio)")
    parser.add_argument("--url", default=LM_STUDIO_URL,
                        help=f"LM Studio server URL (default: {LM_STUDIO_URL})")
    parser.add_argument("--results", action="store_true",
                        help="Print accumulated results and exit")
    args = parser.parse_args()

    if args.results:
        print_summary(load_results())
        return

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"=== LM Studio Benchmark — {datetime.now().isoformat()} ===")

    client = openai.OpenAI(base_url=args.url, api_key="lm-studio", timeout=float(TIMEOUT_SECONDS))

    # Resolve model ID
    if args.model:
        model_id = args.model
    else:
        models = get_loaded_models(client)
        if not models:
            print("\nERROR: Cannot reach LM Studio or no model is loaded.")
            print("Steps:")
            print("  1. Open LM Studio desktop app")
            print("  2. Go to the 'Local Server' tab (left sidebar)")
            print("  3. Click 'Start Server'")
            print("  4. Load a vision model using the model picker at the top")
            print("  5. Re-run this script")
            sys.exit(1)
        if len(models) > 1:
            log(f"Multiple models loaded: {models} — using first. Pass --model to override.")
        model_id = models[0]

    log(f"Image : {args.image} ({Path(args.image).stat().st_size // 1024} KB)")
    log(f"Model : {model_id}")

    result = benchmark_model(client, model_id, args.image)
    save_result(result)
    print_single(result)
    print_summary(load_results())
    log(f"Saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
