#!/usr/bin/env python3
"""
batch_test_runner.py — Run a specific model list and append to test_vision_model_summary.txt
Usage: conda run -n yolo python batch_test_runner.py
"""
from __future__ import annotations
import concurrent.futures
import json
import time
from datetime import datetime
from pathlib import Path
import ollama
from utils import TIMEOUT_SECONDS, count_elements, ollama_chat_with_timeout, validate_json

IMAGE_PATH = "/tmp/test_floorplan_tile.png"
RESULTS_JSON = Path(__file__).parent / "batch_results.json"
SUMMARY_FILE = Path(__file__).parent / "test_vision_model_summary.txt"

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

# ── Model list ────────────────────────────────────────────────────────────────
# (tag, description, approx_gb, pull_needed)
MODELS_TO_TEST = [
    # Already installed — run immediately
    ("qwen3.5:0.8b",    "Qwen3.5 0.8B — tiny thinking (not VL)",          1.0,  False),
    ("qwen3.5:4b",      "Qwen3.5 4B — mid thinking (not VL)",              3.4,  False),
    ("qwen3-vl:2b",     "Qwen3-VL 2B — smallest Qwen vision",              1.9,  False),
    ("qwen3-vl:8b",     "Qwen3-VL 8B — largest tested Qwen vision",        6.1,  False),
    ("ministral-3:3b",  "Ministral 3B — Mistral edge (no vision encoder)", 3.0,  False),
    ("glm-ocr:latest",  "GLM-OCR — THUDM document OCR",                    2.2,  False),
    ("gemma3:12b",      "Gemma 3 12B — Google large vision",                8.1,  False),
    # Need to pull
    ("gemma4:e4b",      "Gemma 4 Effective 4B — Google edge vision",        4.0,  True),
    ("gemma4:latest",   "Gemma 4 latest — Google vision (alias)",           4.0,  True),
    ("qwen3-vl:latest", "Qwen3-VL latest — largest Qwen vision",            6.1,  True),
    ("ministral-3:latest", "Ministral 3 latest — Mistral edge",             3.0,  True),
    ("glm-ocr:98_o",    "GLM-OCR 98_o variant",                             2.2,  True),
]

# ── Skipped models (log only — do not pull/test) ──────────────────────────────
SKIPPED_MODELS = [
    ("qwen3.5:9b",             "already tested in summary ([15])"),
    ("gemma4:e2b",             "already tested in summary ([16])"),
    ("qwen3.5:2b",             "already tested in summary ([9])"),
    ("qwen3-vl:4b",            "already tested in summary ([7])"),
    ("qwen3.5:latest",         "likely qwen3.5:32b — too large for 8 GB VRAM"),
    ("translategemma:latest",  "text translation model — no vision encoder"),
    ("translategemma:4b",      "text translation model — no vision encoder"),
    ("translategemma:12b",     "text translation model — no vision encoder"),
    ("devstral-small-2:latest","code generation model — no vision encoder"),
    ("devstral-small-2:24b",   "code generation model + 24B too large"),
    ("ministral-3:8b",         "does not exist on Ollama"),
    ("ministral-3:14b",        "does not exist on Ollama"),
    ("glm-ocr:bf16",           "bf16 unquantized — likely >16 GB"),
    ("mistral-small3.2",       "text-only LLM — no vision encoder"),
    ("mistral-small3.1:latest","text-only LLM — no vision encoder"),
    ("minstral-small3.1:24b",  "typo + text-only + 24B too large"),
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pull_model(tag: str) -> bool:
    log(f"  Pulling {tag}...")
    try:
        ollama.pull(tag)
        log(f"  Pull OK: {tag}")
        return True
    except Exception as e:
        log(f"  Pull FAILED: {tag} — {e}")
        return False


def run_one(tag: str, image_path: str) -> dict:
    result = {
        "model": tag, "status": "pending",
        "inference_time_s": None, "json_valid": False,
        "raw_output": "", "elements": {}, "error": "",
    }
    log(f"  Inferring {tag}...")
    t0 = time.time()
    try:
        response = ollama_chat_with_timeout(
            TIMEOUT_SECONDS,
            model=tag,
            messages=[{"role": "user", "content": DETECTION_PROMPT, "images": [image_path]}],
            options={"num_predict": 4096, "temperature": 0.1},
            keep_alive=0,
        )
    except concurrent.futures.TimeoutError:
        elapsed = round(time.time() - t0, 2)
        result.update(inference_time_s=elapsed, status="timeout",
                      error=f"exceeded {TIMEOUT_SECONDS}s")
        log(f"  TIMEOUT after {elapsed}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        result.update(inference_time_s=elapsed, status="error", error=str(e)[:300])
        log(f"  ERROR {elapsed}s: {str(e)[:100]}")
        return result

    elapsed = round(time.time() - t0, 2)
    result["inference_time_s"] = elapsed
    raw = response.message.content or ""
    result["raw_output"] = raw[:3000]
    is_valid, parsed, err = validate_json(raw)
    result["json_valid"] = is_valid
    if is_valid and parsed is not None:
        result["elements"] = count_elements(parsed)
        result["status"] = "success"
    else:
        result["error"] = err
        result["status"] = "invalid_json"
    log(f"  Done {elapsed}s | JSON={is_valid} | elements={result['elements'].get('total',0)}")
    return result


def atomic_save(results: list[dict]) -> None:
    _tmp = RESULTS_JSON.with_suffix(".tmp")
    _tmp.write_text(json.dumps(results, indent=2, default=str))
    _tmp.rename(RESULTS_JSON)


def append_to_summary(results: list[dict], skipped: list[tuple]) -> None:
    """Append a new dated section to test_vision_model_summary.txt."""
    lines = []
    lines.append("")
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"  BATCH TEST RESULTS (tested {datetime.now().strftime('%Y-%m-%d')})")
    lines.append(f"  Models tested: {len(results)}   Models skipped: {len(skipped)}")
    lines.append("=" * 100)

    for r in results:
        tag = r["model"]
        gb = next((m[2] for m in MODELS_TO_TEST if m[0] == tag), 0.0)
        e = r.get("elements", {})
        status = r["status"].upper()
        lines.append("")
        score = _score(r)
        lines.append(f"[NEW] {tag:<35} ~{gb:.1f} GB    Score: {score}/6   {status}")
        lines.append(f"      Detections : {e.get('total', 0)}  "
                     f"(columns={e.get('columns',0)}, grid_lines={e.get('grid_lines',0)})")
        lines.append(f"      Time       : {r.get('inference_time_s','-')}s")
        lines.append(f"      JSON valid : {r['json_valid']}")
        if r.get("error"):
            lines.append(f"      Error      : {r['error'][:200]}")
        if r.get("raw_output") and r["status"] == "success":
            lines.append(f"      Sample out : {r['raw_output'][:300]}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("  SKIPPED MODELS (not pulled/tested)")
    lines.append("-" * 100)
    for tag, reason in skipped:
        lines.append(f"  {tag:<35} SKIP  Reason: {reason}")

    lines.append("")
    lines.append("  BATCH SUMMARY TABLE")
    lines.append("-" * 100)
    header = f"  {'Model':<40} {'GB':>5}  {'Time':>7} {'JSON':>6} {'Cols':>5} {'Grid':>5} {'Total':>6}  {'Status'}"
    lines.append(header)
    lines.append("  " + "-" * 95)
    for r in sorted(results, key=lambda x: -_score(x)):
        tag = r["model"][:39]
        gb = next((m[2] for m in MODELS_TO_TEST if m[0] == r["model"]), 0.0)
        t = f"{r['inference_time_s']:.1f}s" if r.get("inference_time_s") else "     -"
        jv = "VALID" if r.get("json_valid") else "   NO"
        e = r.get("elements", {})
        print_line = (f"  {tag:<40} {gb:>5.1f}  {t:>7} {jv:>6} "
                      f"{str(e.get('columns','-')):>5} {str(e.get('grid_lines','-')):>5} "
                      f"{str(e.get('total','-')):>6}  {r['status']}")
        lines.append(print_line)
    lines.append("=" * 100)

    with open(SUMMARY_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Appended results to {SUMMARY_FILE}")


def _score(r: dict) -> int:
    if r["status"] in ("timeout", "error", "pull_failed"):
        return 0
    e = r.get("elements", {})
    s = 0
    if r.get("json_valid"):
        s += 1
    if e.get("total", 0) > 0:
        s += 1
    if 5 <= e.get("total", 0) <= 200:
        s += 1
    # P/R/F1 not computed here — mark 0 for spatial accuracy
    return s


def main() -> None:
    log(f"=== Batch Test — {datetime.now().isoformat()} ===")
    log(f"Image: {IMAGE_PATH}")
    log(f"Models to test: {len(MODELS_TO_TEST)}")
    log(f"Models to skip: {len(SKIPPED_MODELS)}")

    try:
        installed_tags = {m.model for m in ollama.list().models}
    except Exception:
        installed_tags = set()

    results: list[dict] = []

    for i, (tag, desc, gb, pull_needed) in enumerate(MODELS_TO_TEST, 1):
        log(f"\n{'='*70}")
        log(f"[{i}/{len(MODELS_TO_TEST)}] {tag}  ({desc}, ~{gb}GB)")
        log(f"{'='*70}")

        if pull_needed and tag not in installed_tags:
            if not pull_model(tag):
                results.append({
                    "model": tag, "status": "pull_failed",
                    "error": "could not pull", "json_valid": False,
                    "elements": {}, "inference_time_s": None, "raw_output": "",
                })
                atomic_save(results)
                continue
            installed_tags.add(tag)

        result = run_one(tag, IMAGE_PATH)
        result["description"] = desc
        result["approx_gb"] = gb
        results.append(result)
        atomic_save(results)

    log(f"\n{'='*70}")
    log("All done — appending to summary")
    append_to_summary(results, SKIPPED_MODELS)

    # Print console table
    log(f"\n{'='*80}")
    log("RESULTS")
    log(f"{'='*80}")
    for r in results:
        e = r.get("elements", {})
        log(f"  {r['model']:<35} {str(r.get('inference_time_s','-')):>7}s  "
            f"JSON={'Y' if r['json_valid'] else 'N'}  "
            f"total={e.get('total','-')}  {r['status']}")


if __name__ == "__main__":
    main()
