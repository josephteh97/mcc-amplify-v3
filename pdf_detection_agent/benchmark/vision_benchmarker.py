#!/usr/bin/env python3
"""
vision_benchmarker.py — Recursive Vision Model Benchmarking for DfMA Element Detection
=======================================================================================
Tests every available Ollama vision model against a real structural floor plan tile.
Auto-pulls missing models, handles VRAM/timeout failures, logs everything.

Usage:
    conda run -n yolo python vision_benchmarker.py
    conda run -n yolo python vision_benchmarker.py --image /path/to/tile.png
    conda run -n yolo python vision_benchmarker.py --skip-pull   # only test installed models
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import ollama

from utils import TIMEOUT_SECONDS, count_elements, ollama_chat_with_timeout, validate_json

# ── Config ────────────────────────────────────────────────────────────────────

# Master list of 2026 vision-capable models to evaluate
# Format: (model_tag, description, approx_size_gb)
CANDIDATE_MODELS = [
    # ── Already benchmarked (Round 1) ────────────────────────────────────
    ("moondream:latest",                            "Moondream 1.7B — tiny, fast",                    1.7),
    ("qwen2.5vl:3b",                                "Qwen2.5-VL 3B — Alibaba multimodal",             3.2),
    ("qwen3-vl:4b",                                 "Qwen3-VL 4B — latest Qwen vision",               3.3),
    ("deepseek-ocr:3b",                             "DeepSeek-OCR 3B — document-focused",              6.7),
    ("llava:7b",                                    "LLaVA 7B — classic vision-language",              4.7),
    ("bakllava:7b",                                 "BakLLaVA 7B — improved LLaVA variant",            4.7),
    ("minicpm-v:8b",                                "MiniCPM-V 8B — compact multimodal",               5.5),
    ("aisingapore/Gemma-SEA-LION-v4-4B-VL:latest",  "SEA-LION 4B VL — Singapore regional",             3.3),
    ("granite3.2-vision:2b",                        "IBM Granite 3.2 Vision 2B",                       2.4),
    ("gemma3:4b",                                   "Gemma 3 4B — Google multimodal",                  3.3),
    # ── NEW: Round 2 — all remaining Ollama vision models ≤8GB ────────
    ("gemma3:1b",                                   "Gemma 3 1B — Google tiny vision",                 1.0),
    ("gemma3:12b",                                  "Gemma 3 12B — Google large vision",               8.1),
    ("qwen3-vl:2b",                                 "Qwen3-VL 2B — smallest Qwen vision",             2.0),
    ("qwen3-vl:8b",                                 "Qwen3-VL 8B — mid Qwen vision",                  5.5),
    ("qwen2.5vl:7b",                                "Qwen2.5-VL 7B — Alibaba mid vision",             6.0),
    ("qwen3.5:4b",                                  "Qwen3.5 4B — latest Qwen multimodal",            3.3),
    ("llava-phi3:3.8b",                             "LLaVA-Phi3 3.8B — Microsoft Phi3 vision",        2.9),
    ("llava-llama3:8b",                             "LLaVA-Llama3 8B — Meta Llama3 vision",           5.5),
    ("molmo:7b",                                    "Molmo 7B — Allen AI multimodal",                  4.7),
    ("llama3.2-vision:11b",                         "Llama 3.2 Vision 11B — Meta multimodal",          7.9),
    ("glm-ocr:latest",                              "GLM-OCR — THUDM document OCR",                   3.5),
    ("kimi-k2.5:latest",                            "Kimi K2.5 — Moonshot multimodal agentic",         7.0),
    ("ministral-3:3b",                              "Ministral 3B — Mistral edge vision",              2.3),
    ("gemma3:270m",                                 "Gemma 3 270M — ultra-tiny vision",                0.4),
    # ── NEW: Round 3 — Llama 4 MoE ──────────────────────────────────────
    ("llama4:16x17b",                               "Llama 4 Scout 16x17B MoE — Meta multimodal",     67.0),
]

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

LOG_FILE = Path(__file__).parent / "results.log"
RESULTS_JSON = Path(__file__).parent / "benchmark_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_model_installed(model_tag: str, installed_tags: set[str]) -> bool:
    return model_tag in installed_tags


def pull_model(model_tag: str) -> bool:
    """Pull a model, return True if successful."""
    log(f"  Pulling {model_tag}...")
    try:
        ollama.pull(model_tag)
        log(f"  Pull OK: {model_tag}")
        return True
    except Exception as e:
        log(f"  Pull FAILED: {model_tag} — {e}")
        return False


# ── Core benchmark ────────────────────────────────────────────────────────────

def benchmark_model(model_tag: str, image_path: str) -> dict:
    """
    Run a single model benchmark. Returns a result dict.
    """
    result = {
        "model": model_tag,
        "status": "pending",
        "inference_time_s": None,
        "json_valid": False,
        "raw_output": "",
        "elements": {},
        "error": "",
    }

    log(f"  Inferring with {model_tag}...")
    t0 = time.time()
    try:
        response = ollama_chat_with_timeout(
            TIMEOUT_SECONDS,
            model=model_tag,
            messages=[{
                "role": "user",
                "content": DETECTION_PROMPT,
                "images": [image_path],
            }],
            options={"num_predict": 4096, "temperature": 0.1},
            keep_alive=0,
        )
    except concurrent.futures.TimeoutError:
        elapsed = round(time.time() - t0, 2)
        result["inference_time_s"] = elapsed
        result["status"] = "timeout"
        result["error"] = f"exceeded {TIMEOUT_SECONDS}s"
        log(f"  TIMEOUT after {elapsed}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        result["inference_time_s"] = elapsed
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"  ERROR after {elapsed}s: {e}")
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

    log(f"  Done in {elapsed}s — JSON valid: {is_valid} — "
        f"elements: {result['elements'].get('total', 0)}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision model benchmark for DfMA detection")
    parser.add_argument("--image", default="/tmp/test_floorplan_tile.png",
                        help="Path to test image (default: rendered floor plan tile)")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Only test already-installed models, don't pull new ones")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model tags to test (default: all candidates)")
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        print("Run: conda run -n yolo python -c \"import fitz; ...\" first to render a tile")
        sys.exit(1)

    # Clear log
    LOG_FILE.write_text(f"=== Vision Model Benchmark — {datetime.now().isoformat()} ===\n")
    log(f"Image: {image_path} ({Path(image_path).stat().st_size // 1024} KB)")

    # Filter models if specific ones requested
    if args.models:
        candidates = [(m, d, s) for m, d, s in CANDIDATE_MODELS if m in args.models]
    else:
        candidates = CANDIDATE_MODELS

    try:
        installed_tags = {m.model for m in ollama.list().models}
    except Exception:
        installed_tags = set()

    results = []
    for i, (model_tag, description, approx_gb) in enumerate(candidates, 1):
        log(f"\n{'='*60}")
        log(f"[{i}/{len(candidates)}] {model_tag} — {description} (~{approx_gb}GB)")
        log(f"{'='*60}")

        # Check / pull model
        if not is_model_installed(model_tag, installed_tags):
            if args.skip_pull:
                log(f"  SKIPPED (not installed, --skip-pull)")
                results.append({
                    "model": model_tag, "description": description,
                    "status": "skipped", "error": "not installed",
                })
                continue
            if not pull_model(model_tag):
                results.append({
                    "model": model_tag, "description": description,
                    "status": "pull_failed", "error": "could not pull",
                })
                continue
            installed_tags.add(model_tag)

        # Run benchmark
        result = benchmark_model(model_tag, image_path)
        result["description"] = description
        result["approx_gb"] = approx_gb
        results.append(result)

        # Atomic incremental save — crash-safe for long runs
        _tmp = RESULTS_JSON.with_suffix(".tmp")
        _tmp.write_text(json.dumps(results, indent=2, default=str))
        _tmp.rename(RESULTS_JSON)

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_JSON}")

    # ── Print summary table ───────────────────────────────────────────────
    log(f"\n{'='*100}")
    log("FINAL RESULTS TABLE")
    log(f"{'='*100}")
    header = f"{'Model':<45} {'Time':>7} {'JSON':>6} {'Cols':>5} {'Grid':>5} {'Total':>6} {'Status':<15}"
    log(header)
    log("-" * 100)

    for r in results:
        model = r["model"][:44]
        t = f"{r.get('inference_time_s', '-'):.1f}s" if r.get("inference_time_s") else "-"
        jv = "VALID" if r.get("json_valid") else "NO"
        elems = r.get("elements", {})
        cols = str(elems.get("columns", "-"))
        grids = str(elems.get("grid_lines", "-"))
        total = str(elems.get("total", "-"))
        status = r.get("status", "?")
        log(f"{model:<45} {t:>7} {jv:>6} {cols:>5} {grids:>5} {total:>6} {status:<15}")

    log(f"\n{'='*100}")

    # ── Best model recommendation ─────────────────────────────────────────
    valid = [r for r in results if r.get("json_valid")]
    if valid:
        # Score: valid JSON + most elements + fastest
        def score(r):
            e = r.get("elements", {})
            return (e.get("total", 0) * 10
                    + e.get("valid_coords", 0) * 5
                    - (r.get("inference_time_s", 999)))
        best = max(valid, key=score)
        log(f"\nBEST MODEL: {best['model']}")
        log(f"  Elements: {best['elements']}")
        log(f"  Time: {best.get('inference_time_s')}s")
    else:
        log("\nNo model produced valid JSON.")

    log(f"\nFull log: {LOG_FILE}")
    log(f"JSON results: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
