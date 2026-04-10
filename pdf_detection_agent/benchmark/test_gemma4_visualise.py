"""
Run gemma4:e4b on the test floorplan tile and produce an annotated image.
  RED   boxes = model detections
  GREEN boxes = confirmed GT columns from detections.db
  BLUE  boxes = overlapping (model det within 30px of a GT box)

Output saved to /tmp/gemma4_e4b_annotated.png
"""
import json, time, sys, sqlite3
sys.path.insert(0, ".")
from utils import ollama_chat_with_timeout, validate_json
from PIL import Image as _Image, ImageDraw, ImageFont

TAG         = "gemma4:e4b"
VISION_SIZE = 1024
ORIG_PATH   = "/tmp/test_floorplan_tile.png"
TILE_PATH   = "/tmp/test_tile_1024.png"
OUT_PATH    = "/tmp/gemma4_e4b_annotated.png"

# Resize tile to 1024px for inference
_Image.open(ORIG_PATH).resize((VISION_SIZE, VISION_SIZE), _Image.LANCZOS).save(TILE_PATH)
print(f"Tile prepared: {TILE_PATH}")

PROMPT = """You are a structural engineer. Analyze this architectural floor plan tile and detect all structural columns.

Return ONLY valid JSON:
{
  "detections": [
    {
      "type": "column",
      "shape": "square|rectangle|round",
      "confidence": 0.0-1.0,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

Rules:
- bbox pixel values within this 1024x1024 image
- Columns are small solid/hatched squares or rectangles at structural grid intersections
- Do NOT detect grid balloons, walls, doors, text, or dimension lines
- Report every column you can see — be thorough
"""

print(f"Running {TAG}...")
t0 = time.time()
response = ollama_chat_with_timeout(
    300, model=TAG,
    messages=[{"role": "user", "content": PROMPT, "images": [TILE_PATH]}],
    options={"num_predict": 4096, "temperature": 0.1},
    keep_alive=0,
)
elapsed = time.time() - t0
raw = response["message"]["content"] if response else ""
is_valid, parsed, err = validate_json(raw)
dets = parsed if (is_valid and isinstance(parsed, list)) else []
print(f"Time: {elapsed:.1f}s  |  Valid: {is_valid}  |  Detections: {len(dets)}")
print(f"Raw output:\n{raw[:800]}")

# --- Load GT boxes from detections.db (confirmed corrections) ---
S_GT = VISION_SIZE / 1280  # GT coords are in 1280px space → scale to 1024px
gt_boxes = []
try:
    conn = sqlite3.connect(
        "/home/jiezhi/Documents/mcc-amplify-v2/pdf_detection_agent/detections.db"
    )
    rows = conn.execute(
        "SELECT bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM corrections WHERE action='confirm'"
    ).fetchall()
    conn.close()
    for r in rows:
        x1, y1, x2, y2 = r[0]*1280*S_GT, r[1]*1280*S_GT, r[2]*1280*S_GT, r[3]*1280*S_GT
        gt_boxes.append([round(x1), round(y1), round(x2), round(y2)])
    print(f"GT boxes loaded: {len(gt_boxes)}")
except Exception as e:
    print(f"GT load failed: {e}")

# --- Draw annotated image ---
img = _Image.open(TILE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

def box_centre(b):
    return (b[0]+b[2])/2, (b[1]+b[3])/2

def near_gt(bbox, tol=30):
    if len(bbox) != 4: return False
    cx, cy = box_centre(bbox)
    for g in gt_boxes:
        gx, gy = box_centre(g)
        if abs(cx-gx) < tol and abs(cy-gy) < tol:
            return True
    return False

# Draw GT boxes in green (thin)
for g in gt_boxes:
    draw.rectangle(g, outline=(0, 200, 0), width=2)

# Draw model detections
det_count = 0
overlap_count = 0
for d in dets:
    bbox = d.get("bbox", [])
    if len(bbox) != 4:
        continue
    det_count += 1
    if near_gt(bbox):
        colour = (0, 120, 255)   # blue = overlaps GT
        overlap_count += 1
    else:
        colour = (220, 30, 30)   # red = no GT match
    draw.rectangle(bbox, outline=colour, width=3)
    # Label with index
    draw.text((bbox[0]+2, bbox[1]+2), str(det_count), fill=colour)

# Legend
legend_lines = [
    f"RED   = model det (no GT match): {det_count - overlap_count}",
    f"BLUE  = model det overlaps GT:   {overlap_count}",
    f"GREEN = confirmed GT columns:    {len(gt_boxes)}",
    f"Total model dets: {det_count}  |  Time: {elapsed:.0f}s",
]
y = 10
for line in legend_lines:
    draw.rectangle([8, y-1, 8+len(line)*7, y+14], fill=(0, 0, 0, 180))
    draw.text((10, y), line, fill=(255, 255, 255))
    y += 18

img.save(OUT_PATH)
print(f"\nAnnotated image saved: {OUT_PATH}")
print(f"Open with: eog {OUT_PATH}  OR  xdg-open {OUT_PATH}")
print(f"\nDetection breakdown:")
print(f"  Model dets total    : {det_count}")
print(f"  Overlap with GT     : {overlap_count}  ({100*overlap_count/max(det_count,1):.0f}%)")
print(f"  No GT match (FP?)   : {det_count - overlap_count}")
print(f"\nAll model bbox coords:")
for i, d in enumerate(dets):
    bbox = d.get("bbox", [])
    w = bbox[2]-bbox[0] if len(bbox)==4 else "?"
    h = bbox[3]-bbox[1] if len(bbox)==4 else "?"
    gt_hit = near_gt(bbox)
    print(f"  [{i:02d}] {bbox}  {w}x{h}px  conf={d.get('confidence','?')}  {'<<GT' if gt_hit else ''}")

# Save raw result
with open("/tmp/gemma4_e4b_raw.json", "w") as f:
    json.dump({"elapsed": elapsed, "n_dets": det_count, "dets": dets, "raw": raw}, f, indent=2)
print(f"\nRaw results saved: /tmp/gemma4_e4b_raw.json")
