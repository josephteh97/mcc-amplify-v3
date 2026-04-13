"""
Memory injection test — remaining 4 models not yet in results JSON
(llama3.2-vision:11b, qwen2.5vl:3b, moondream:latest, llava-phi3:3.8b)

Loads existing 3-model results and appends, then saves back to same JSON.
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
    dict(tag="llama3.2-vision:11b", timeout=420),
    dict(tag="qwen2.5vl:3b",        timeout=300),
    dict(tag="moondream:latest",    timeout=120),
    dict(tag="llava-phi3:3.8b",     timeout=120),
]

with open(RESULTS_PATH) as f:
    existing = json.load(f)
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

    merged_results = dict(existing.get("results", {}))
    for k, v in all_results.items():
        merged_results[k] = {p: {kk: vv for kk, vv in pv.items() if kk != "dets"}
                             for p, pv in v.items() if isinstance(pv, dict)}
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "gt_n": len(GT),
            "results": merged_results,
            "skipped": existing.get("skipped", {}),
            "already_tested": existing.get("already_tested", []),
        }, f, indent=2)
    print(f"  -> saved to {RESULTS_PATH}")

print(f"\n{'='*70}")
print(f"REMAINING MODELS SUMMARY  (GT={len(GT)} columns)")
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
