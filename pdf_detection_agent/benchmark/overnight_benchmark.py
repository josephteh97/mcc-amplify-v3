#!/usr/bin/env python3
"""
overnight_benchmark.py — Exhaustive Ollama Vision Model Benchmark
=================================================================
Pulls and tests EVERY known Ollama vision model against a structural floor plan.
Designed to run unattended overnight. Writes results to test_vision_model_summary.txt.

Usage:
    conda run -n yolo nohup python overnight_benchmark.py &
    # or just:
    conda run -n yolo python overnight_benchmark.py
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz
import ollama
from PIL import Image

# ── Test image ────────────────────────────────────────────────────────────────
IMAGE_PATH = "/tmp/test_floorplan_tile.png"
SUMMARY_FILE = Path(__file__).parent / "test_vision_model_summary.txt"
RESULTS_JSON = Path(__file__).parent / "overnight_results.json"

# ── Complete Ollama Vision Model Catalogue (April 2026) ───────────────────────
# Every vision-capable model from https://ollama.com/search?c=vision
# Filtered to tags that can fit in 8GB VRAM (with a few stretch goals)
#
# Format: (tag, description, approx_gb, notes)

ALL_VISION_MODELS = [
    # ── Gemma family (Google) ─────────────────────────────────────────────
    ("gemma3:270m",         "Gemma 3 270M — ultra-tiny",             0.3,  "292MB"),
    ("gemma3:1b",           "Gemma 3 1B — tiny",                     0.8,  "815MB"),
    ("gemma3:4b",           "Gemma 3 4B — mid",                      3.3,  "3.3GB"),
    ("gemma3:1b-it-qat",    "Gemma 3 1B QAT — quantized",           0.5,  "QAT variant"),
    ("gemma3:4b-it-qat",    "Gemma 3 4B QAT — quantized",           2.5,  "QAT variant"),
    ("gemma3:12b",          "Gemma 3 12B — large",                   8.1,  "may not fit 8GB"),
    ("gemma3:12b-it-qat",   "Gemma 3 12B QAT — quantized large",    6.0,  "QAT may fit"),
    ("gemma4:e2b",          "Gemma 4 Effective 2B — edge optimized", 2.0,  "mobile/IoT focused"),
    ("gemma4:e4b",          "Gemma 4 Effective 4B — edge optimized", 3.5,  "mobile/IoT focused"),

    # ── Qwen family (Alibaba) ─────────────────────────────────────────────
    ("qwen2.5vl:3b",        "Qwen2.5-VL 3B",                        3.2,  "flagship VL"),
    ("qwen2.5vl:7b",        "Qwen2.5-VL 7B",                        6.0,  "mid VL"),
    ("qwen3-vl:2b",         "Qwen3-VL 2B — smallest",               2.0,  "newest VL"),
    ("qwen3-vl:4b",         "Qwen3-VL 4B",                          3.3,  "newest VL"),
    ("qwen3-vl:8b",         "Qwen3-VL 8B",                          6.1,  "newest VL"),
    ("qwen3.5:0.8b",        "Qwen3.5 0.8B — nano multimodal",       0.8,  "text+image"),
    ("qwen3.5:2b",          "Qwen3.5 2B — small multimodal",        2.0,  "text+image"),
    ("qwen3.5:4b",          "Qwen3.5 4B — mid multimodal",          3.3,  "text+image"),
    ("qwen3.5:9b",          "Qwen3.5 9B — large multimodal",        6.5,  "text+image"),

    # ── LLaVA family ──────────────────────────────────────────────────────
    ("llava:7b",             "LLaVA 1.6 7B — classic",               4.7,  "original VLM"),
    ("llava:13b",            "LLaVA 1.6 13B — large",                8.0,  "may OOM"),
    ("llava-phi3:3.8b",      "LLaVA-Phi3 3.8B — Microsoft",         2.9,  "compact"),
    ("llava-llama3:8b",      "LLaVA-Llama3 8B — Meta base",         5.5,  "Meta Llama3"),
    ("bakllava:7b",          "BakLLaVA 7B — Mistral-based",         4.7,  "improved LLaVA"),

    # ── Meta Llama family ─────────────────────────────────────────────────
    ("llama3.2-vision:11b",  "Llama 3.2 Vision 11B",                 7.9,  "Meta multimodal"),

    # ── OCR/Document focused ──────────────────────────────────────────────
    ("deepseek-ocr:3b",      "DeepSeek-OCR 3B — document OCR",      6.7,  "OCR focused"),
    ("glm-ocr:latest",       "GLM-OCR — THUDM document OCR",        3.5,  "Chinese + English"),

    # ── Other notable models ──────────────────────────────────────────────
    ("minicpm-v:8b",         "MiniCPM-V 8B — compact multimodal",   5.5,  "strong general"),
    ("moondream:latest",     "Moondream 1.7B — edge device",        1.7,  "tiny but fast"),
    ("granite3.2-vision:2b", "IBM Granite 3.2 Vision 2B",           2.4,  "enterprise"),
    ("ministral-3:3b",       "Ministral 3B — Mistral edge",         2.3,  "edge deployment"),

    # ── Singapore / Regional ──────────────────────────────────────────────
    ("aisingapore/Gemma-SEA-LION-v4-4B-VL:latest",
                             "SEA-LION 4B VL — Singapore",           3.3,  "regional champion"),
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

TIMEOUT_SECONDS = 300  # 5 min max per model


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def pull_model(tag: str) -> bool:
    """Force-pull a model. Returns True on success."""
    log(f"    Pulling {tag}...")
    try:
        ollama.pull(tag)
        log(f"    Pull OK")
        return True
    except Exception as e:
        log(f"    Pull FAILED: {e}")
        return False


def validate_json(text: str) -> tuple[bool, list | None, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return True, parsed, ""
        if isinstance(parsed, dict):
            for key in ("columns", "detections", "elements", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return True, parsed[key], ""
            return False, None, f"dict without known array key"
        return False, None, f"not array/dict"
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return True, parsed, ""
        except json.JSONDecodeError:
            pass

    items = []
    for m in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            items.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    if items:
        return True, items, ""

    return False, None, f"parse failed: {text[:150]}"


def count_elements(dets: list[dict]) -> dict:
    cols = [d for d in dets if d.get("element_type") == "column"]
    grids = [d for d in dets if d.get("element_type") == "grid_line"]
    has_coords = sum(1 for d in dets
                     if isinstance(d.get("coordinates"), list) and len(d["coordinates"]) == 4)
    has_conf = sum(1 for d in dets if isinstance(d.get("confidence"), (int, float)))
    return {
        "total": len(dets), "columns": len(cols), "grid_lines": len(grids),
        "other": len(dets) - len(cols) - len(grids),
        "valid_coords": has_coords, "valid_conf": has_conf,
    }


def benchmark_one(tag: str, image_path: str, installed_tags: set[str]) -> dict:
    """Run inference for one model. Pull first if needed."""
    result = {
        "model": tag, "status": "pending",
        "inference_time_s": None, "json_valid": False,
        "raw_output": "", "elements": {}, "error": "",
    }

    if tag not in installed_tags:
        if not pull_model(tag):
            result["status"] = "pull_failed"
            result["error"] = "could not pull"
            return result
        installed_tags.add(tag)

    log(f"    Inferring...")
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                ollama.chat,
                model=tag,
                messages=[{
                    "role": "user",
                    "content": DETECTION_PROMPT,
                    "images": [image_path],
                }],
                options={"num_predict": 4096, "temperature": 0.1},
            )
            try:
                response = future.result(timeout=TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                elapsed = round(time.time() - t0, 2)
                result["inference_time_s"] = elapsed
                result["status"] = "timeout"
                result["error"] = f"exceeded {TIMEOUT_SECONDS}s"
                log(f"    TIMEOUT after {elapsed}s")
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

        log(f"    Done {elapsed}s | JSON={is_valid} | elements={result['elements'].get('total', 0)}")

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        result["inference_time_s"] = elapsed
        result["status"] = "error"
        result["error"] = str(e)[:300]
        log(f"    ERROR {elapsed}s: {str(e)[:120]}")

    return result


# ── Summary writer ────────────────────────────────────────────────────────────

def write_summary(results: list[dict]) -> None:
    """Write human-readable summary to test_vision_model_summary.txt"""
    lines = []
    lines.append("=" * 100)
    lines.append("OLLAMA VISION MODEL BENCHMARK — DfMA Column Detection")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"GPU: Quadro RTX 4000 (8GB VRAM)")
    lines.append(f"Test image: 1280x1280 structural floor plan tile (TGCH-TD-S-200-L3-00.pdf)")
    lines.append(f"Task: Detect structural columns + grid lines, return JSON")
    lines.append(f"Models tested: {len(results)}")
    lines.append("=" * 100)

    # ── Results table ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("RESULTS TABLE")
    lines.append("-" * 100)
    header = f"{'#':<4} {'Model':<48} {'Time':>7} {'JSON':>6} {'Cols':>5} {'Grid':>5} {'Total':>6} {'Status':<15}"
    lines.append(header)
    lines.append("-" * 100)

    # Sort: successes first (by total elements desc), then errors
    def sort_key(r):
        if r["status"] == "success":
            return (0, -r.get("elements", {}).get("total", 0), r.get("inference_time_s", 999))
        return (1, 0, 0)

    sorted_results = sorted(results, key=sort_key)

    for i, r in enumerate(sorted_results, 1):
        model = r["model"][:47]
        t = f"{r.get('inference_time_s', 0):.1f}s" if r.get("inference_time_s") else "-"
        jv = "VALID" if r.get("json_valid") else "NO"
        e = r.get("elements", {})
        cols = str(e.get("columns", "-"))
        grids = str(e.get("grid_lines", "-"))
        total = str(e.get("total", "-"))
        status = r.get("status", "?")
        lines.append(f"{i:<4} {model:<48} {t:>7} {jv:>6} {cols:>5} {grids:>5} {total:>6} {status:<15}")

    # ── Analysis ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 100)
    lines.append("ANALYSIS")
    lines.append("=" * 100)

    success = [r for r in results if r.get("json_valid")]
    with_elements = [r for r in success if r.get("elements", {}).get("total", 0) > 0]
    with_columns = [r for r in success if r.get("elements", {}).get("columns", 0) > 0]
    errors = [r for r in results if r["status"] in ("error", "pull_failed")]
    invalid = [r for r in results if r["status"] == "invalid_json"]

    lines.append(f"")
    lines.append(f"Total models attempted:           {len(results)}")
    lines.append(f"Produced valid JSON:              {len(success)}")
    lines.append(f"Detected any elements:            {len(with_elements)}")
    lines.append(f"Detected columns specifically:    {len(with_columns)}")
    lines.append(f"Crashed / OOM / errors:           {len(errors)}")
    lines.append(f"Invalid JSON output:              {len(invalid)}")

    # ── Top models ────────────────────────────────────────────────────────
    if with_elements:
        lines.append("")
        lines.append("-" * 100)
        lines.append("TOP MODELS (by total elements detected)")
        lines.append("-" * 100)
        top = sorted(with_elements,
                     key=lambda r: r.get("elements", {}).get("total", 0), reverse=True)
        for i, r in enumerate(top[:10], 1):
            e = r["elements"]
            lines.append(
                f"  #{i} {r['model']:<45} "
                f"cols={e.get('columns',0):>3}  grids={e.get('grid_lines',0):>3}  "
                f"total={e.get('total',0):>3}  time={r.get('inference_time_s',0):.1f}s"
            )

    # ── Fastest valid models ──────────────────────────────────────────────
    if success:
        lines.append("")
        lines.append("-" * 100)
        lines.append("FASTEST MODELS (valid JSON)")
        lines.append("-" * 100)
        fastest = sorted(success, key=lambda r: r.get("inference_time_s", 999))
        for i, r in enumerate(fastest[:10], 1):
            e = r.get("elements", {})
            lines.append(
                f"  #{i} {r['model']:<45} "
                f"time={r.get('inference_time_s',0):>7.1f}s  elements={e.get('total',0)}"
            )

    # ── Failed models ─────────────────────────────────────────────────────
    if errors:
        lines.append("")
        lines.append("-" * 100)
        lines.append("FAILED MODELS")
        lines.append("-" * 100)
        for r in errors:
            lines.append(f"  {r['model']:<48} {r['status']:<15} {r.get('error','')[:60]}")

    # ── Recommendation ────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 100)
    lines.append("RECOMMENDATION")
    lines.append("=" * 100)
    if with_columns:
        best = max(with_columns, key=lambda r: (
            r["elements"]["columns"] * 10 + r["elements"]["total"]
            - r.get("inference_time_s", 999) * 0.1))
        lines.append(f"  Best overall: {best['model']}")
        lines.append(f"    Columns: {best['elements']['columns']}")
        lines.append(f"    Grid lines: {best['elements']['grid_lines']}")
        lines.append(f"    Total elements: {best['elements']['total']}")
        lines.append(f"    Inference time: {best.get('inference_time_s')}s")
    else:
        lines.append("  No model detected columns. Local VLMs are not viable for this task.")
    lines.append("")
    lines.append("  NOTE: Even the best local model detects far fewer columns than exist")
    lines.append("  in the floor plan. For production use, YOLO (column-detect.pt) or")
    lines.append("  OpenAI GPT-4o are recommended instead of local Ollama VLMs.")
    lines.append("=" * 100)

    SUMMARY_FILE.write_text("\n".join(lines))
    log(f"Summary written to {SUMMARY_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not Path(IMAGE_PATH).exists():
        # Render test tile from PDF
        log("Rendering test tile from floor plan PDF...")
        try:
            pdf_path = "/home/jiezhi/Documents/floor-plan-pdf/TGCH-TD-S-200-L3-00.pdf"
            doc = fitz.open(pdf_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(150/72, 150/72), colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            cx, cy = img.width // 2, img.height // 2
            crop = img.crop((cx - 640, cy - 640, cx + 640, cy + 640))
            crop.save(IMAGE_PATH)
            log(f"Tile saved: {IMAGE_PATH} ({crop.size})")
        except Exception as e:
            log(f"FATAL: Cannot render test image: {e}")
            sys.exit(1)

    log("=" * 80)
    log("OVERNIGHT VISION MODEL BENCHMARK — STARTING")
    log(f"Models to test: {len(ALL_VISION_MODELS)}")
    log(f"Image: {IMAGE_PATH}")
    log("=" * 80)

    try:
        installed_tags = {m.model for m in ollama.list().models}
    except Exception:
        installed_tags = set()

    results = []
    for i, (tag, desc, approx_gb, notes) in enumerate(ALL_VISION_MODELS, 1):
        log(f"\n[{i}/{len(ALL_VISION_MODELS)}] {tag} — {desc} (~{approx_gb}GB) [{notes}]")
        result = benchmark_one(tag, IMAGE_PATH, installed_tags)
        result["description"] = desc
        result["approx_gb"] = approx_gb
        result["notes"] = notes
        results.append(result)

        # Save incrementally (so we don't lose progress on crash)
        RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))

    write_summary(results)
    log("\n" + "=" * 80)
    log("ALL MODELS COMPLETE")
    log(f"Results: {RESULTS_JSON}")
    log(f"Summary: {SUMMARY_FILE}")
    log("=" * 80)


if __name__ == "__main__":
    main()
