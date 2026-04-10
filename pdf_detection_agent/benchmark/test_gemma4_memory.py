"""
gemma4:e4b — baseline + memory injection test
GT: 86 columns extracted from human-annotated image (red circles)
Coordinates in 1024px space (model inference size)

Phases:
  1. BASELINE    — plain prompt, no hints
  2. ANCHOR      — 5 GT boxes given as examples, "find ALL remaining"
  3. ADD-ONLY    — growing pool given, "find ADDITIONAL not in this list"
"""
import json, time, sys, sqlite3, cv2, numpy as np
sys.path.insert(0, ".")
from utils import ollama_chat_with_timeout, validate_json
from PIL import Image as _Image

TAG         = "gemma4:e4b"
VISION_SIZE = 1024
ORIG_PATH   = "/tmp/test_floorplan_tile.png"       # clean floor plan
ANNOT_PATH  = "/home/jiezhi/tmp/test_floorplan_tile.png"  # human-annotated GT
TILE_PATH   = "/tmp/gemma4_mem_tile_1024.png"

# ── Prepare 1024px tile ──────────────────────────────────────────────────────
_Image.open(ORIG_PATH).resize((VISION_SIZE, VISION_SIZE), _Image.LANCZOS).save(TILE_PATH)
print(f"Tile: {TILE_PATH}")

# ── Extract 86 GT boxes from annotated image (in 1024px space) ───────────────
annot = cv2.imread(ANNOT_PATH)
hsv   = cv2.cvtColor(annot, cv2.COLOR_BGR2HSV)
m1    = cv2.inRange(hsv, (0,  100, 80), (10, 255, 255))
m2    = cv2.inRange(hsv, (165, 100, 80), (180, 255, 255))
red   = cv2.dilate(cv2.bitwise_or(m1, m2),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

S = VISION_SIZE / 1280   # 1280 → 1024
gt_boxes_1024 = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if 200 < cv2.contourArea(c) < 15000 and w <= 200 and h <= 200:
        cx, cy = (x + x + w) // 2, (y + y + h) // 2
        hw = max(12, int(w * S * 0.5))
        hh = max(12, int(h * S * 0.5))
        ccx, ccy = int(cx * S), int(cy * S)
        gt_boxes_1024.append([ccx-hw, ccy-hh, ccx+hw, ccy+hh])

gt_boxes_1024.sort(key=lambda b: (b[1]//80, b[0]))
print(f"GT boxes (1024px): {len(gt_boxes_1024)}")

# ── Helper: evaluate detections vs GT ────────────────────────────────────────
def evaluate(dets, tol=25):
    matched_gt = set()
    tp = 0
    for d in dets:
        bbox = d if isinstance(d, list) else d.get("bbox", [])
        if len(bbox) != 4: continue
        cx = (bbox[0]+bbox[2])/2; cy = (bbox[1]+bbox[3])/2
        for i, g in enumerate(gt_boxes_1024):
            if i in matched_gt: continue
            gx = (g[0]+g[2])/2; gy = (g[1]+g[3])/2
            if abs(cx-gx) < tol and abs(cy-gy) < tol:
                tp += 1; matched_gt.add(i); break
    fp = len(dets) - tp
    fn = len(gt_boxes_1024) - tp
    p  = tp/(tp+fp) if tp+fp > 0 else 0
    r  = tp/(tp+fn) if tp+fn > 0 else 0
    f1 = 2*p*r/(p+r) if p+r > 0 else 0
    return dict(tp=tp, fp=fp, fn=fn, p=round(p,3), r=round(r,3), f1=round(f1,3))

# ── Detection helper ──────────────────────────────────────────────────────────
def run_phase(name, prompt):
    print(f"\n{'='*64}")
    print(f"PHASE: {name}")
    print(f"{'='*64}")
    t0 = time.time()
    response = ollama_chat_with_timeout(
        360, model=TAG,
        messages=[{"role": "user", "content": prompt, "images": [TILE_PATH]}],
        options={"num_predict": 8192, "temperature": 0.1},
        keep_alive=0,
    )
    elapsed = time.time() - t0
    raw = response["message"]["content"] if response else ""
    is_valid, parsed, err = validate_json(raw)
    dets = parsed if (is_valid and isinstance(parsed, list)) else []
    metrics = evaluate(dets)
    print(f"Time      : {elapsed:.1f}s")
    print(f"Valid JSON : {is_valid}  ({err})")
    print(f"Detections: {len(dets)}")
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
          f"P={metrics['p']:.2f}  R={metrics['r']:.2f}  F1={metrics['f1']:.2f}")
    print(f"Raw (first 600):\n{raw[:600]}")
    return dict(phase=name, elapsed=elapsed, n_dets=len(dets),
                valid=is_valid, dets=dets, metrics=metrics, raw=raw)

# ── Prompts ───────────────────────────────────────────────────────────────────
BASE_PROMPT = """You are a structural engineer. Analyze this architectural floor plan tile and detect ALL structural columns.

Structural columns appear as small GRAY-FILLED squares or rectangles (~20-40px) at structural grid line intersections. They are labeled nearby with text like "C1", "C2", "RCB2", "SB1" etc.

Return ONLY valid JSON — a list of detections:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

Rules:
- bbox pixel coordinates within this 1024x1024 image
- Do NOT detect walls, doors, text labels, grid balloons, or room outlines
- Each column is a small filled square ~20-40px wide
- Be thorough — detect EVERY column you can see
"""

def anchor_prompt(sample_boxes):
    examples = "\n".join(f"  {b}" for b in sample_boxes)
    return f"""You are a structural engineer analyzing an architectural floor plan tile.

I have already confirmed these structural columns in this image (bbox = [x1,y1,x2,y2] in 1024x1024 pixels):
{examples}

These are small GRAY-FILLED squares at structural grid intersections (labeled C1, C2, RCB2, SB1 etc).

Now find ALL REMAINING columns that are NOT already in the list above.
Return ONLY valid JSON — a list:
[
  {{"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}}
]
Include ONLY new columns not already listed. Be thorough — there are many more.
"""

def addonly_prompt(known_boxes):
    known_str = "\n".join(f"  {b}" for b in known_boxes[:40])  # cap to avoid token overrun
    n = len(known_boxes)
    return f"""You are a structural engineer analyzing an architectural floor plan tile.

I have already found {n} structural columns in this image (bbox = [x1,y1,x2,y2] in 1024x1024 pixels):
{known_str}{"..." if n>40 else ""}

These are small GRAY-FILLED squares at structural grid intersections.

Find ADDITIONAL structural columns that are NOT in the list above.
Return ONLY valid JSON — a list of NEW detections only:
[
  {{"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}}
]
Do NOT repeat columns already listed. Focus on areas/rows not yet covered.
"""

# ── Run phases ────────────────────────────────────────────────────────────────
results = {}

# Phase 1: Baseline
r1 = run_phase("BASELINE", BASE_PROMPT)
results["baseline"] = r1

# Phase 2: Anchor — use 10 GT boxes spread across the image as anchors
# Pick every ~8th box to give spatial spread
anchor_idxs = list(range(0, len(gt_boxes_1024), max(1, len(gt_boxes_1024)//10)))[:10]
anchor_boxes = [gt_boxes_1024[i] for i in anchor_idxs]
r2 = run_phase("ANCHOR (10 GT examples → find remaining)", anchor_prompt(anchor_boxes))
results["anchor"] = r2

# Accumulate: anchor boxes + any new valid detections from anchor phase
def dedup(boxes, tol=20):
    kept = []
    for b in boxes:
        cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
        if all(abs(cx-(k[0]+k[2])/2)>tol or abs(cy-(k[1]+k[3])/2)>tol for k in kept):
            kept.append(b)
    return kept

# Build accumulated pool: start from anchor GT boxes
pool = list(anchor_boxes)
for d in r2["dets"]:
    bbox = d if isinstance(d, list) else d.get("bbox", [])
    if len(bbox) == 4:
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
        if 10 <= w <= 80 and 10 <= h <= 80:  # valid-sized box
            pool.append(bbox)
pool = dedup(pool)
print(f"\nAccumulated pool after ANCHOR phase: {len(pool)} boxes")

# Phase 3: Add-only with accumulated pool
r3 = run_phase("ADD-ONLY (accumulated pool → find additional)", addonly_prompt(pool))
results["addonly"] = r3

# Accumulate further
for d in r3["dets"]:
    bbox = d if isinstance(d, list) else d.get("bbox", [])
    if len(bbox) == 4:
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
        if 10 <= w <= 80 and 10 <= h <= 80:
            pool.append(bbox)
pool = dedup(pool)
print(f"\nAccumulated pool after ADD-ONLY phase: {len(pool)} boxes")

# Phase 4: Second add-only round
r4 = run_phase("ADD-ONLY round 2", addonly_prompt(pool))
results["addonly2"] = r4

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"SUMMARY  (GT={len(gt_boxes_1024)} columns)")
print(f"{'='*64}")
for k, r in results.items():
    m = r["metrics"]
    print(f"  {k:35s}  dets={r['n_dets']:3d}  TP={m['tp']:3d}  FP={m['fp']:3d}  "
          f"FN={m['fn']:3d}  P={m['p']:.2f}  R={m['r']:.2f}  F1={m['f1']:.2f}  time={r['elapsed']:.0f}s")

with open("/tmp/gemma4_memory_test.json", "w") as f:
    json.dump({k: {kk: vv for kk,vv in v.items() if kk != "dets"} for k,v in results.items()}, f, indent=2)
print("\nSaved: /tmp/gemma4_memory_test.json")
