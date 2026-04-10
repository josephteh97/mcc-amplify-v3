"""
gemma4:e4b — Visual Reference Harness Test

Harness design (mirroring the GPT-4o few-shot approach):
  Instead of injecting text coordinates, send IMAGES of what columns
  look like as visual few-shot examples alongside the main tile.

Phases:
  1. BASELINE        — main tile only, plain prompt
  2. PATCH-REFS      — 8 individual column patch images + main tile
  3. ANNOTATED-CROP  — annotated floor section (columns marked green) + main tile
  4. PANEL-REF       — single reference panel (grid of column examples) + main tile
  5. PANEL+ANNOT     — panel + annotated crop + main tile (maximum visual context)
"""
import json, time, sys, os, cv2, numpy as np
sys.path.insert(0, ".")
from utils import ollama_chat_with_timeout, validate_json
from PIL import Image as _Image, ImageDraw

TAG         = "gemma4:e4b"
VISION_SIZE = 1024
CLEAN_PATH  = "/tmp/test_floorplan_tile.png"
ANNOT_PATH  = "/home/jiezhi/tmp/test_floorplan_tile.png"
TILE_PATH   = "/tmp/gemma4h_tile_1024.png"
REF_DIR     = "/tmp/gemma4_refs"
os.makedirs(REF_DIR, exist_ok=True)

# ── Resize tile ───────────────────────────────────────────────────────────────
_Image.open(CLEAN_PATH).resize((VISION_SIZE, VISION_SIZE), _Image.LANCZOS).save(TILE_PATH)
print(f"Tile: {TILE_PATH}")

# ── Extract 86 GT boxes (1024px space) from annotated image ──────────────────
annot_bgr = cv2.imread(ANNOT_PATH)
hsv       = cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2HSV)
m1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
m2 = cv2.inRange(hsv, (165, 100, 80), (180, 255, 255))
red = cv2.dilate(cv2.bitwise_or(m1, m2),
                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
S = VISION_SIZE / 1280
gt_boxes = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if 200 < cv2.contourArea(c) < 15000 and w <= 200 and h <= 200:
        cx, cy = (x + x + w) // 2, (y + y + h) // 2
        hw = max(12, int(w * S * 0.5))
        hh = max(12, int(h * S * 0.5))
        ccx, ccy = int(cx * S), int(cy * S)
        gt_boxes.append([ccx - hw, ccy - hh, ccx + hw, ccy + hh])
gt_boxes.sort(key=lambda b: (b[1] // 80, b[0]))
print(f"GT boxes: {len(gt_boxes)}")

# ── Build reference images ────────────────────────────────────────────────────
clean_bgr = cv2.imread(CLEAN_PATH)

# 1. Individual column patches — pick 8 spatially diverse samples
PAD = 22
patch_paths = []
indices = [int(i * (len(gt_boxes)-1) / 7) for i in range(8)]
for k, idx in enumerate(indices):
    x1, y1, x2, y2 = gt_boxes[idx]
    cx, cy = (x1+x2)//2, (y1+y2)//2
    # Coordinates are already in 1024px space; scale back to 1280 for clean img
    cx1280 = int(cx / S); cy1280 = int(cy / S)
    crop = clean_bgr[max(0, cy1280-PAD):cy1280+PAD, max(0, cx1280-PAD):cx1280+PAD]
    if crop.size == 0: continue
    crop_r = cv2.resize(crop, (64, 64))
    # Draw green box to mark the column
    cv2.rectangle(crop_r, (12, 12), (52, 52), (0, 200, 0), 2)
    path = f"{REF_DIR}/patch_{k:02d}.png"
    cv2.imwrite(path, crop_r)
    patch_paths.append(path)
print(f"Saved {len(patch_paths)} column patches")

# 2. Reference panel — all patches in a 4×2 grid with labels
panel = np.ones((160, 300, 3), dtype=np.uint8) * 255
cv2.putText(panel, "STRUCTURAL COLUMN EXAMPLES", (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
for k, pp in enumerate(patch_paths):
    patch = cv2.imread(pp)
    r, c = k // 4, k % 4
    panel[20 + r*68:20 + r*68 + 64, 4 + c*74:4 + c*74 + 64] = patch
panel_path = f"{REF_DIR}/reference_panel.png"
cv2.imwrite(panel_path, panel)
print(f"Reference panel: {panel_path}")

# 3. Annotated crop — show a section of the floor plan with GT boxes drawn
# Use the middle-right area which has clear, regular columns
CROP = (300, 80, 900, 420)  # x1,y1,x2,y2 in 1280px
cx1, cy1, cx2, cy2 = CROP
crop_annot = clean_bgr[cy1:cy2, cx1:cx2].copy()
shown = 0
for b in gt_boxes:
    # Convert 1024px back to 1280px
    bx1 = int(b[0]/S); by1 = int(b[1]/S); bx2 = int(b[2]/S); by2 = int(b[3]/S)
    if bx1 >= cx1 and bx2 <= cx2 and by1 >= cy1 and by2 <= cy2:
        cv2.rectangle(crop_annot, (bx1-cx1, by1-cy1), (bx2-cx1, by2-cy1), (0, 200, 0), 2)
        shown += 1
crop_annot_r = cv2.resize(crop_annot, (512, 256))
annot_crop_path = f"{REF_DIR}/annotated_crop.png"
cv2.imwrite(annot_crop_path, crop_annot_r)
print(f"Annotated crop ({shown} columns shown): {annot_crop_path}")

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(dets, tol=25):
    matched = set()
    tp = 0
    for d in dets:
        bbox = d if isinstance(d, list) else d.get("bbox", [])
        if len(bbox) != 4: continue
        cx = (bbox[0]+bbox[2])/2; cy = (bbox[1]+bbox[3])/2
        for i, g in enumerate(gt_boxes):
            if i in matched: continue
            gx = (g[0]+g[2])/2; gy = (g[1]+g[3])/2
            if abs(cx-gx) < tol and abs(cy-gy) < tol:
                tp += 1; matched.add(i); break
    fp = len(dets) - tp; fn = len(gt_boxes) - tp
    p  = tp/(tp+fp) if tp+fp>0 else 0
    r  = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return dict(tp=tp, fp=fp, fn=fn, p=round(p,3), r=round(r,3), f1=round(f1,3))

def run_phase(name, prompt, images):
    print(f"\n{'='*64}")
    print(f"PHASE: {name}")
    print(f"Images: {[os.path.basename(i) for i in images]}")
    print(f"{'='*64}")
    t0 = time.time()
    response = ollama_chat_with_timeout(
        420, model=TAG,
        messages=[{"role": "user", "content": prompt, "images": images}],
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
    print(f"Raw (first 500):\n{raw[:500]}")
    return dict(phase=name, elapsed=elapsed, n_dets=len(dets),
                valid=is_valid, metrics=metrics, raw=raw, dets=dets)

# ── Prompts ───────────────────────────────────────────────────────────────────
BASELINE_PROMPT = """You are a structural engineer. Analyze this architectural floor plan tile and detect ALL structural columns.

Structural columns are small GRAY-FILLED squares or rectangles (~20-40px) sitting at structural grid line intersections. They are often labeled nearby with "C1", "C2", "RCB2", "SB1" etc.

Return ONLY valid JSON — a flat list:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

Rules:
- bbox pixel coordinates within this 1024x1024 image
- Do NOT detect walls, text, grid lines, room outlines, or door swings
- Each column is a small gray-filled square ~20-40px wide
- Detect EVERY column visible — be thorough
"""

PATCH_REF_PROMPT = """You are a structural engineer analyzing an architectural floor plan.

The FIRST 8 images are close-up examples of structural columns from this exact floor plan. Each shows a small gray-filled square/rectangle (marked with green box) at a grid intersection. These columns are labeled nearby with text like "C1", "C2", "RCB2".

The LAST image is the full 1024x1024 floor plan tile.

Detect ALL structural columns in the LAST image that match the appearance of the examples.

Return ONLY valid JSON — a flat list:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

- bbox pixel coordinates within the 1024x1024 floor plan
- Each column is a small gray-filled square ~20-40px wide
- Be thorough — detect every column you can see
"""

PANEL_REF_PROMPT = """You are a structural engineer analyzing an architectural floor plan.

IMAGE 1 (reference panel): Shows 8 examples of structural columns from this exact floor plan. Each is a small gray-filled square/rectangle sitting at a structural grid intersection.

IMAGE 2 (floor plan): The full 1024x1024 floor plan tile to analyze.

Find ALL elements in IMAGE 2 that match the column appearance shown in IMAGE 1.

Return ONLY valid JSON — a flat list:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

- bbox coordinates in the 1024x1024 floor plan (IMAGE 2)
- Small gray-filled squares ~20-40px wide at grid intersections
- Be thorough — there are approximately 80+ columns
"""

ANNOT_CROP_PROMPT = """You are a structural engineer analyzing an architectural floor plan.

IMAGE 1 (annotated section): A section of the floor plan with structural columns marked by GREEN BOXES. The columns are the small gray-filled square symbols inside the green boxes, located at structural grid intersections.

IMAGE 2 (full floor plan): The complete 1024x1024 floor plan tile to analyze.

Using IMAGE 1 as a visual guide for what columns look like, detect ALL structural columns in IMAGE 2.

Return ONLY valid JSON — a flat list:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

- bbox coordinates in IMAGE 2 (the 1024x1024 floor plan)
- Match the gray-filled square symbols shown in IMAGE 1
- Be thorough — there are approximately 80+ columns across the full plan
"""

PANEL_ANNOT_PROMPT = """You are a structural engineer analyzing an architectural floor plan.

IMAGE 1 (column examples): 8 close-up examples of structural columns — small gray-filled squares at grid intersections.
IMAGE 2 (annotated section): A section of the floor plan with columns marked in GREEN — showing these columns in their drawing context.
IMAGE 3 (full floor plan): The complete 1024x1024 tile to analyze.

Using IMAGES 1 and 2 as visual reference, detect ALL structural columns in IMAGE 3.

Return ONLY valid JSON — a flat list:
[
  {"type": "column", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
]

- bbox coordinates in IMAGE 3 (the 1024x1024 floor plan)
- Small gray-filled squares ~20-40px wide at structural grid intersections
- There are approximately 80+ columns — be thorough
"""

# ── Run all phases ────────────────────────────────────────────────────────────
results = {}

r1 = run_phase("BASELINE", BASELINE_PROMPT, [TILE_PATH])
results["baseline"] = r1

r2 = run_phase("PATCH-REFS (8 patches + tile)", PATCH_REF_PROMPT,
               patch_paths + [TILE_PATH])
results["patch_refs"] = r2

r3 = run_phase("PANEL-REF (panel + tile)", PANEL_REF_PROMPT,
               [panel_path, TILE_PATH])
results["panel_ref"] = r3

r4 = run_phase("ANNOTATED-CROP (annot crop + tile)", ANNOT_CROP_PROMPT,
               [annot_crop_path, TILE_PATH])
results["annot_crop"] = r4

r5 = run_phase("PANEL+ANNOT (panel + annot crop + tile)", PANEL_ANNOT_PROMPT,
               [panel_path, annot_crop_path, TILE_PATH])
results["panel_annot"] = r5

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"SUMMARY  (GT={len(gt_boxes)} columns, tol=25px)")
print(f"{'='*64}")
print(f"{'Phase':<40}  {'Dets':>5}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'P':>5}  {'R':>5}  {'F1':>5}  {'Time':>6}")
for k, r in results.items():
    m = r["metrics"]
    print(f"  {r['phase']:<38}  {r['n_dets']:>5}  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}  "
          f"{m['p']:>5.2f}  {m['r']:>5.2f}  {m['f1']:>5.2f}  {r['elapsed']:>5.0f}s")

# Save results (without full dets to keep file small)
save = {k: {kk: vv for kk, vv in v.items() if kk not in ("dets", "raw")} for k, v in results.items()}
for k, v in results.items():
    save[k]["raw_preview"] = v["raw"][:400]
with open("/tmp/gemma4_harness_results.json", "w") as f:
    json.dump(save, f, indent=2)
print("\nSaved: /tmp/gemma4_harness_results.json")

# ── Visualise best phase ──────────────────────────────────────────────────────
best_key = max(results, key=lambda k: results[k]["metrics"]["f1"])
best     = results[best_key]
print(f"\nBest phase: {best_key}  F1={best['metrics']['f1']:.2f}")

tile_img = _Image.open(TILE_PATH).convert("RGB")
draw     = ImageDraw.Draw(tile_img)
for g in gt_boxes:
    draw.rectangle(g, outline=(0, 200, 0), width=2)
for d in best["dets"]:
    bbox = d if isinstance(d, list) else d.get("bbox", [])
    if len(bbox) == 4:
        draw.rectangle(bbox, outline=(220, 30, 30), width=3)
tile_img.save("/tmp/gemma4_harness_best.png")
print("Best-phase annotated image: /tmp/gemma4_harness_best.png")
