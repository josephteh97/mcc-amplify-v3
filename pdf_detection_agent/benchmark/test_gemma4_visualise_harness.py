"""
gemma4:e4b harness — run each phase and save annotated image immediately.
GREEN = GT columns, RED = model detections
"""
import json, time, sys, os, cv2, numpy as np
sys.path.insert(0, ".")
from utils import ollama_chat_with_timeout, validate_json
from PIL import Image as _Image, ImageDraw

TAG        = "gemma4:e4b"
TILE_PATH  = "/tmp/gemma4h_tile_1024.png"   # already exists
ANNOT_PATH = "/home/jiezhi/tmp/test_floorplan_tile.png"
REF_DIR    = "/tmp/gemma4_refs"             # already exists

# ── GT ───────────────────────────────────────────────────────────────────────
annot = cv2.imread(ANNOT_PATH)
hsv   = cv2.cvtColor(annot, cv2.COLOR_BGR2HSV)
m1 = cv2.inRange(hsv,(0,100,80),(10,255,255))
m2 = cv2.inRange(hsv,(165,100,80),(180,255,255))
red = cv2.dilate(cv2.bitwise_or(m1,m2),
                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
cnts,_ = cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
S=1024/1280; gt=[]
for c in cnts:
    x,y,w,h=cv2.boundingRect(c)
    if 200<cv2.contourArea(c)<15000 and w<=200 and h<=200:
        cx,cy=(x+x+w)//2,(y+y+h)//2
        hw=max(12,int(w*S*0.5)); hh=max(12,int(h*S*0.5))
        gt.append([int(cx*S)-hw,int(cy*S)-hh,int(cx*S)+hw,int(cy*S)+hh])
print(f"GT: {len(gt)} columns")

# ── Eval ─────────────────────────────────────────────────────────────────────
def evaluate(dets, tol=25):
    matched=set(); tp=0
    for d in dets:
        b = d if isinstance(d,list) else (d.get("bbox") or d.get("box_2d") or [])
        if len(b)!=4: continue
        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
        for i,g in enumerate(gt):
            if i in matched: continue
            if abs(cx-(g[0]+g[2])/2)<tol and abs(cy-(g[1]+g[3])/2)<tol:
                tp+=1; matched.add(i); break
    fp=len(dets)-tp; fn=len(gt)-tp
    p=tp/(tp+fp) if tp+fp>0 else 0
    r=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return tp,fp,fn,round(p,2),round(r,2),round(f1,2)

def save_annotated(dets, phase_name, out_path):
    img  = _Image.open(TILE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    # GT green
    for g in gt:
        draw.rectangle(g, outline=(0,200,0), width=2)
    # model detections red
    bbox_list = []
    for d in dets:
        b = d if isinstance(d,list) else (d.get("bbox") or d.get("box_2d") or [])
        if len(b)==4:
            draw.rectangle(list(b), outline=(220,30,30), width=3)
            bbox_list.append(b)
    tp,fp,fn,p,r,f1 = evaluate(dets)
    hdr = f"{phase_name}  |  dets={len(bbox_list)}  TP={tp}  FP={fp}  FN={fn}  P={p}  R={r}  F1={f1}"
    draw.rectangle([0,0,1024,22], fill=(0,0,0))
    draw.text((4,4), hdr, fill=(255,255,0))
    img.save(out_path)
    print(f"  -> {out_path}")
    return tp,fp,fn,p,r,f1

def run(name, prompt, images, out_path):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    t0=time.time()
    resp = ollama_chat_with_timeout(
        420, model=TAG,
        messages=[{"role":"user","content":prompt,"images":images}],
        options={"num_predict":8192,"temperature":0.1},
        keep_alive=0,
    )
    elapsed=time.time()-t0
    raw = resp["message"]["content"] if resp else ""
    ok,parsed,err = validate_json(raw)
    dets = parsed if (ok and isinstance(parsed,list)) else []
    print(f"Time: {elapsed:.0f}s  Valid: {ok}  Dets: {len(dets)}")
    tp,fp,fn,p,r,f1 = save_annotated(dets, name, out_path)
    print(f"TP={tp}  FP={fp}  FN={fn}  P={p}  R={r}  F1={f1}")
    return dict(name=name,elapsed=elapsed,n=len(dets),tp=tp,fp=fp,fn=fn,p=p,r=r,f1=f1)

# ── Reference image paths ─────────────────────────────────────────────────────
patch_paths   = sorted([f"{REF_DIR}/{f}" for f in os.listdir(REF_DIR) if f.startswith("patch_")])
panel_path    = f"{REF_DIR}/reference_panel.png"
annot_crop    = f"{REF_DIR}/annotated_crop.png"

# ── Prompts ───────────────────────────────────────────────────────────────────
P_BASE = """You are a structural engineer. Detect ALL structural columns in this 1024x1024 architectural floor plan tile.

Columns are small GRAY-FILLED squares/rectangles (~20-40px) at structural grid intersections, labeled nearby with C1, C2, RCB2, SB1, etc.

Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]

Every column must be detected. Do NOT invent regular grids — only mark what you actually see."""

P_PATCH = """You are a structural engineer. The first 8 images are close-up examples of structural column symbols from this exact floor plan — each is a small gray-filled square marked with a green box.

The LAST image is the full 1024x1024 floor plan. Find every element in it that matches those column examples.

Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]

Coordinates must be within the last image (1024x1024). Do not invent regular patterns."""

P_PANEL = """You are a structural engineer.
IMAGE 1: Reference panel showing 8 structural column examples (small gray-filled squares at grid intersections).
IMAGE 2: Full 1024x1024 floor plan — detect all columns that match the examples in IMAGE 1.

Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]

Coordinates are in IMAGE 2. Do not invent regularly-spaced grids — only mark what you actually see."""

P_ANNOT = """You are a structural engineer.
IMAGE 1: A section of the floor plan with structural columns circled in GREEN. Study these carefully — the columns are small gray-filled squares at grid intersections.
IMAGE 2: The full 1024x1024 floor plan — detect ALL columns using IMAGE 1 as your visual guide.

Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]

Coordinates are in IMAGE 2 only. Match the exact visual appearance from IMAGE 1."""

P_BOTH = """You are a structural engineer.
IMAGE 1: 8 close-up examples of structural columns (small gray squares, green box marks the column).
IMAGE 2: A section of the floor plan showing those same columns in context, circled in GREEN.
IMAGE 3: Full 1024x1024 floor plan — detect ALL structural columns.

Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]

Coordinates are in IMAGE 3. Use images 1 and 2 to understand what columns look like. Do not invent patterns."""

# ── Run ───────────────────────────────────────────────────────────────────────
results = []
results.append(run("1-BASELINE",        P_BASE,  [TILE_PATH],                              "/tmp/vis_1_baseline.png"))
results.append(run("2-PATCH-REFS",      P_PATCH, patch_paths+[TILE_PATH],                  "/tmp/vis_2_patch_refs.png"))
results.append(run("3-PANEL-REF",       P_PANEL, [panel_path, TILE_PATH],                  "/tmp/vis_3_panel_ref.png"))
results.append(run("4-ANNOTATED-CROP",  P_ANNOT, [annot_crop, TILE_PATH],                  "/tmp/vis_4_annot_crop.png"))
results.append(run("5-PANEL+ANNOT",     P_BOTH,  [panel_path, annot_crop, TILE_PATH],      "/tmp/vis_5_panel_annot.png"))

print(f"\n{'='*60}\nSUMMARY  GT={len(gt)}\n{'='*60}")
print(f"{'Phase':<30} {'Dets':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>5} {'R':>5} {'F1':>5} {'s':>5}")
for r in results:
    print(f"  {r['name']:<28} {r['n']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
          f"{r['p']:>5.2f} {r['r']:>5.2f} {r['f1']:>5.2f} {r['elapsed']:>4.0f}s")

with open("/tmp/gemma4_vis_results.json","w") as f:
    json.dump(results,f,indent=2)
