"""
Iterative Memory Injection Test — Self-Evolution Candidates
Tests 3 models × 5 rounds each.
Each round feeds confirmed TPs into the next round's anchor pool.
Annotated image saved after every round for human visual assessment.

Round structure:
  Round 1: BASELINE — no injection, zero-shot
  Round 2: ANCHOR   — 10 spread GT seeds, "find remaining"
  Round 3: ANCHOR   — pool = seeds + R2 TPs
  Round 4: ANCHOR   — pool = R3 pool + R3 TPs
  Round 5: ANCHOR   — pool = R4 pool + R4 TPs

Image legend (1024×1024 output):
  GREEN  = all 86 GT columns
  CYAN   = confirmed pool carried from previous rounds
  RED    = new detections this round
  Header = model, round, TP/FP/FN/P/R/F1, pool size
"""
import json, sys, os
sys.path.insert(0, ".")
from utils import (
    extract_gt_boxes, extract_bboxes, evaluate, get_tp_boxes, dedup,
    run_inference, build_tile, P_BASE, P_ANCHOR,
)
from PIL import Image as _Image, ImageDraw

ANNOT_PATH   = "/home/jiezhi/tmp/test_floorplan_tile.png"
CLEAN_PATH   = "/tmp/test_floorplan_tile.png"
OUT_DIR      = "/tmp/mi_iter"
RESULTS_PATH = "/tmp/mi_iter_results.json"
os.makedirs(OUT_DIR, exist_ok=True)

TILE_PATH  = build_tile(CLEAN_PATH, 1024)
GT         = extract_gt_boxes(ANNOT_PATH, 1024)
SEED_BOXES = [GT[int(i * (len(GT)-1) / 9)] for i in range(10)]
print(f"GT: {len(GT)} columns (1024px)\n")


def save_annotated(dets, pool, metrics, round_label, model_tag, out_path):
    img  = _Image.open(TILE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    for g in GT:
        draw.rectangle(g, outline=(0, 210, 0), width=2)
    for b in pool:
        draw.rectangle(b, outline=(0, 210, 220), width=2)
    for b in extract_bboxes(dets):
        draw.rectangle(b, outline=(230, 30, 30), width=3)
    m = metrics
    hdr = (f"{model_tag}  {round_label}  |  "
           f"dets={len(extract_bboxes(dets))}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
           f"P={m['p']:.2f}  R={m['r']:.2f}  F1={m['f1']:.2f}  pool={len(pool)}")
    draw.rectangle([0, 0, 1024, 22], fill=(0, 0, 0))
    draw.text((4, 4), hdr, fill=(255, 230, 0))
    draw.rectangle([0, 1002, 1024, 1024], fill=(20, 20, 20))
    draw.text((4, 1006), "GREEN=GT(86)   CYAN=confirmed pool   RED=current detections", fill=(180, 180, 180))
    img.save(out_path)
    print(f"    -> {out_path}")


MODELS = [
    dict(tag="qwen2.5vl:3b",        timeout=300),
    dict(tag="llama3.2-vision:11b",  timeout=420),
    dict(tag="gemma3:4b",            timeout=120),
]
ROUNDS = 5

all_results = {}

for cfg in MODELS:
    tag  = cfg["tag"]
    to   = cfg["timeout"]
    safe = tag.replace(":", "_").replace(".", "_")
    print(f"\n{'#'*60}\n# {tag}\n{'#'*60}")

    model_rounds = []
    pool = []

    for rnd in range(1, ROUNDS + 1):
        if rnd == 1:
            label       = "R1-BASELINE"
            prompt      = P_BASE
            pool_before = []
        elif rnd == 2:
            pool        = dedup(list(SEED_BOXES))
            label       = "R2-ANCHOR(seed=10)"
            prompt      = P_ANCHOR(pool)
            pool_before = pool
        else:
            label       = f"R{rnd}-ANCHOR(pool={len(pool)})"
            prompt      = P_ANCHOR(pool)
            pool_before = pool

        print(f"  [{rnd}/{ROUNDS}] {label}  pool_size={len(pool_before)}...")
        r = run_inference(tag, prompt, TILE_PATH, timeout_s=to)
        m = evaluate(r["dets"], GT)
        print(f"    dets={r['n_dets']:>4}  TP={m['tp']:>3}  FP={m['fp']:>3}  FN={m['fn']:>3}  "
              f"P={m['p']:.2f}  R={m['r']:.2f}  F1={m['f1']:.2f}  t={r['elapsed']}s")

        img_path = f"{OUT_DIR}/{safe}_r{rnd}.png"
        save_annotated(r["dets"], pool_before, m, label, tag, img_path)

        if rnd >= 2:
            new_tps = get_tp_boxes(r["dets"], GT)
            pool    = dedup(pool + new_tps)
            if new_tps:
                print(f"    pool grew: +{len(new_tps)} new TPs → pool now {len(pool)}")

        model_rounds.append(dict(
            round=rnd, label=label,
            pool_size_before=len(pool_before),
            pool_size_after=len(pool),
            n_dets=r["n_dets"], metrics=m,
            elapsed=r["elapsed"], error=r.get("error"),
            image=img_path,
        ))

    all_results[tag] = model_rounds

    print(f"\n  {'Round/Label':<28} {'Dets':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}  Pool→After")
    print(f"  {'-'*70}")
    for rr in model_rounds:
        m = rr["metrics"]
        print(f"  {rr['label']:<28} {rr['n_dets']:>5} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['f1']:>6.3f}  {rr['pool_size_before']}→{rr['pool_size_after']}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"gt_n": len(GT), "rounds": ROUNDS,
               "models": {tag: [{k: v for k, v in r.items() if k not in ("dets",)}
                                for r in rounds]
                          for tag, rounds in all_results.items()}}, f, indent=2)
print(f"\nSaved: {RESULTS_PATH}")

print(f"\n{'='*70}")
print(f"ITERATIVE MI SUMMARY  (GT={len(GT)}, 5 rounds)")
print(f"{'='*70}")
for tag, rounds in all_results.items():
    f1s = [f"{r['metrics']['f1']:.3f}" for r in rounds]
    tps = [str(r['metrics']['tp']) for r in rounds]
    print(f"\n  {tag}")
    print(f"  Round :  " + "  ".join(f"R{r['round']}" for r in rounds))
    print(f"  F1    :  " + "  ".join(f"{f:>5}" for f in f1s))
    print(f"  TP    :  " + "  ".join(f"{t:>5}" for t in tps))
    print(f"  Pool→ :  " + "  ".join(f"{r['pool_size_after']:>5}" for r in rounds))
print(f"\nImages: {OUT_DIR}/")
