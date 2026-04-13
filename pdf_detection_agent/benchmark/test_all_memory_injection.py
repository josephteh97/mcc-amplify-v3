"""
Memory injection test — ALL remaining testable VLMs
Runs 3 phases per model: BASELINE → ANCHOR → ADD-ONLY
GT: 86 columns from human-annotated image (correct GT, 1024px space)

Models tested here (7):
  gemma3:12b, gemma4:e2b, granite3.2-vision:2b, llama3.2-vision:11b,
  qwen2.5vl:3b, moondream:latest, llava-phi3:3.8b

Skipped (logged in results):
  aliases       : gemma4:latest, glm-ocr:bf16, qwen3-vl:latest, mistral-small3.1:24b
  CLIP crash    : bakllava:7b, llava:7b, llava-llama3:8b
  GGML crash    : glm-ocr:q8_0
  OOM           : llama4:16x17b
  TIMEOUT       : ministral-3:14b, mistral-small3.1:latest
  text-only     : ministral-3:latest/:3b/:8b, qwen3.5:2b, translategemma, devstral
  thinking/no-op: qwen3-vl:2b/:8b, qwen3.5:9b/:0.8b
  OCR-only      : deepseek-ocr:3b
  already done  : gemma4:e4b, qwen2.5vl:7b, minicpm-v:8b, gemma3:4b,
                  qwen3-vl:4b, glm-ocr:latest, qwen3.5:4b
"""
import json, sys
sys.path.insert(0, ".")
from utils import (
    extract_gt_boxes, extract_bboxes, evaluate, dedup, run_inference,
    build_tile, P_BASE, P_ANCHOR, P_ADDONLY,
)

ANNOT_PATH   = "/home/jiezhi/tmp/test_floorplan_tile.png"
CLEAN_PATH   = "/tmp/test_floorplan_tile.png"
RESULTS_PATH = "/tmp/all_memory_injection_results.json"

TILE_PATH    = build_tile(CLEAN_PATH, 1024)
GT           = extract_gt_boxes(ANNOT_PATH, 1024)
ANCHOR_BOXES = [GT[int(i * (len(GT)-1) / 9)] for i in range(10)]
print(f"GT: {len(GT)} columns (1024px)\n")

MODELS = [
    dict(tag="gemma3:12b",           timeout=300),
    dict(tag="gemma4:e2b",           timeout=600),
    dict(tag="granite3.2-vision:2b", timeout=120),
    dict(tag="llama3.2-vision:11b",  timeout=420),
    dict(tag="qwen2.5vl:3b",         timeout=300),
    dict(tag="moondream:latest",     timeout=120),
    dict(tag="llava-phi3:3.8b",      timeout=120),
]

SKIPPED = {
    "gemma4:latest":          "alias of gemma4:e4b (already tested)",
    "glm-ocr:bf16":           "alias of glm-ocr:latest (already tested)",
    "qwen3-vl:latest":        "alias of qwen3-vl:8b (thinking, 0 output)",
    "mistral-small3.1:24b":   "alias of mistral-small3.1:latest (TIMEOUT)",
    "bakllava:7b":            "CLIP runner crash (HTTP 500)",
    "llava:7b":               "CLIP runner crash (HTTP 500)",
    "llava-llama3:8b":        "CLIP runner crash (HTTP 500)",
    "glm-ocr:q8_0":           "GGML tensor assertion crash",
    "llama4:16x17b":          "OOM — requires 58 GB RAM",
    "ministral-3:14b":        "TIMEOUT (676s on 8 GB VRAM)",
    "mistral-small3.1:latest":"TIMEOUT (987–1686s on 8 GB VRAM)",
    "ministral-3:latest":     "text-only — no vision encoder",
    "ministral-3:3b":         "text-only — no vision encoder",
    "ministral-3:8b":         "text-only — no vision encoder",
    "qwen3.5:2b":             "text-only — no vision encoder",
    "translategemma":         "text-only — no vision encoder",
    "devstral":               "text-only — no vision encoder",
    "qwen3-vl:2b":            "thinking model — 0 output (token budget exhausted)",
    "qwen3-vl:8b":            "thinking model — 0 output (token budget exhausted)",
    "qwen3.5:9b":             "thinking model — 0 output",
    "qwen3.5:0.8b":           "thinking model — 0 output",
    "deepseek-ocr:3b":        "OCR-only — no spatial detection",
}

ALREADY_TESTED = [
    "gemma4:e4b", "qwen2.5vl:7b", "minicpm-v:8b",
    "gemma3:4b", "qwen3-vl:4b", "glm-ocr:latest", "qwen3.5:4b",
]

all_results = {}

for cfg in MODELS:
    tag = cfg["tag"]; to = cfg["timeout"]
    print(f"\n{'#'*60}\n# {tag}\n{'#'*60}")

    print("  [1/3] BASELINE...")
    r_base = run_inference(tag, P_BASE, TILE_PATH, timeout_s=to)
    r_base["metrics"] = evaluate(r_base["dets"], GT)
    print(f"        dets={r_base['n_dets']}  TP={r_base['metrics']['tp']}  "
          f"F1={r_base['metrics']['f1']:.2f}  t={r_base['elapsed']}s")

    print("  [2/3] ANCHOR...")
    r_anch = run_inference(tag, P_ANCHOR(ANCHOR_BOXES), TILE_PATH, timeout_s=to)
    r_anch["metrics"] = evaluate(r_anch["dets"], GT)
    print(f"        dets={r_anch['n_dets']}  TP={r_anch['metrics']['tp']}  "
          f"F1={r_anch['metrics']['f1']:.2f}  t={r_anch['elapsed']}s")

    pool = dedup(ANCHOR_BOXES + [b for b in extract_bboxes(r_anch["dets"])
                                 if 8 <= b[2]-b[0] <= 100 and 8 <= b[3]-b[1] <= 100])

    print("  [3/3] ADD-ONLY...")
    r_add = run_inference(tag, P_ADDONLY(pool), TILE_PATH, timeout_s=to)
    r_add["metrics"] = evaluate(r_add["dets"], GT)
    print(f"        dets={r_add['n_dets']}  TP={r_add['metrics']['tp']}  "
          f"F1={r_add['metrics']['f1']:.2f}  t={r_add['elapsed']}s")

    all_results[tag] = dict(baseline=r_base, anchor=r_anch, addonly=r_add, pool_size=len(pool))

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "gt_n": len(GT),
            "results": {k: {p: {kk: vv for kk, vv in v.items() if kk != "dets"}
                            for p, v in res.items() if isinstance(v, dict)}
                        for k, res in all_results.items()},
            "skipped": SKIPPED,
            "already_tested": ALREADY_TESTED,
        }, f, indent=2)
    print(f"  -> saved to {RESULTS_PATH}")

print(f"\n{'='*70}")
print(f"MEMORY INJECTION SUMMARY  (GT={len(GT)} columns)")
print(f"{'='*70}")
print(f"{'Model':<28} {'BASELINE':>10} {'ANCHOR':>10} {'ADD-ONLY':>10}  Verdict")
print(f"{'-'*70}")
for tag, res in all_results.items():
    b = res["baseline"]["metrics"]["f1"]
    a = res["anchor"]["metrics"]["f1"]
    d = res["addonly"]["metrics"]["f1"]
    best = max(b, a, d)
    verdict = "HELPS" if best > b + 0.02 else ("HURTS" if best < b - 0.02 else "no change")
    print(f"  {tag:<26}  {b:>8.3f}  {a:>8.3f}  {d:>8.3f}   {verdict}")
print(f"{'-'*70}")
print(f"\nSkipped ({len(SKIPPED)} models): see {RESULTS_PATH}")
print(f"Already tested ({len(ALREADY_TESTED)} models): {', '.join(ALREADY_TESTED)}")
