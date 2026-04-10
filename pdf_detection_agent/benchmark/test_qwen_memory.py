"""
Two-phase episodic memory test for qwen2.5vl:7b
Phase 1 — Baseline: plain detection prompt, no hints
Phase 2 — Memory:   inject 5 confirmed GT boxes as few-shot correction context
Goal: does the model improve when given episodic correction examples?
"""
import json, time, sys
sys.path.insert(0, ".")
from utils import TIMEOUT_SECONDS, ollama_chat_with_timeout, validate_json

TAG        = "qwen2.5vl:7b"
IMAGE_PATH = "/tmp/test_floorplan_tile.png"

# Ground truth: 5 canonical column centres (averaged from confirmed corrections)
# These represent the leftmost column in each row — chosen as clear, unambiguous examples
GT_EXAMPLES = [
    {"bbox": [66, 161, 117, 210], "shape": "square", "confidence": 0.99, "notes": "confirmed GT"},
    {"bbox": [64, 409, 115, 462], "shape": "square", "confidence": 0.99, "notes": "confirmed GT"},
    {"bbox": [64, 664, 115, 704], "shape": "square", "confidence": 0.99, "notes": "confirmed GT"},
    {"bbox": [75, 916, 115, 952], "shape": "square", "confidence": 0.99, "notes": "confirmed GT"},
    {"bbox": [72, 1165, 120, 1216], "shape": "square", "confidence": 0.99, "notes": "confirmed GT"},
]

BASE_PROMPT = """You are a structural engineer. Analyze this architectural floor plan tile and detect all structural columns.

Return ONLY valid JSON in this exact format:
{
  "detections": [
    {
      "type": "column",
      "shape": "square|rectangle|round|i_beam",
      "confidence": 0.0-1.0,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

Rules:
- bbox coordinates are pixel values within this 1280x1280 image
- Columns are small (30-60 px) solid/filled/hatched structural elements at grid intersections
- Do NOT detect grid balloons (hollow circles at grid line ends with axis labels)
- Do NOT detect walls, doors, dimension lines, or text
- Report exact pixel coordinates you can see — do not guess or space evenly
"""

MEMORY_PROMPT = BASE_PROMPT + f"""

EPISODIC MEMORY — confirmed correct detections from a human expert for this exact image:
{json.dumps(GT_EXAMPLES, indent=2)}

Using these confirmed examples as visual anchors, find ALL remaining columns in the image
that follow the same appearance pattern. The confirmed examples show you exactly what
a structural column looks like in this drawing style.
"""

def run_phase(phase_name, prompt):
    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        response = ollama_chat_with_timeout(
            TIMEOUT_SECONDS, model=TAG,
            messages=[{"role": "user", "content": prompt, "images": [IMAGE_PATH]}],
            options={"num_predict": 4096, "temperature": 0.1},
            keep_alive=0,
        )
        elapsed = time.time() - t0
        raw = response["message"]["content"] if response else ""
        is_valid, parsed, err = validate_json(raw)
        dets = parsed if (is_valid and isinstance(parsed, list)) else []

        print(f"Time      : {elapsed:.1f}s")
        print(f"Valid JSON : {is_valid}")
        print(f"Detections: {len(dets)}")

        if dets:
            print(f"\nAll detections:")
            for i, d in enumerate(dets):
                bbox = d.get("bbox", [])
                conf = d.get("confidence", "?")
                shape = d.get("shape", "?")
                # Check against GT — is this near any GT box? (simple centre-distance check)
                near_gt = False
                if len(bbox) == 4:
                    cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                    for gt in GT_EXAMPLES:
                        gx = (gt["bbox"][0]+gt["bbox"][2])/2
                        gy = (gt["bbox"][1]+gt["bbox"][3])/2
                        if abs(cx-gx) < 60 and abs(cy-gy) < 60:
                            near_gt = True
                            break
                marker = " <<NEAR GT" if near_gt else ""
                print(f"  [{i:02d}] bbox={bbox}  conf={conf}  shape={shape}{marker}")

        # Hallucination check: are coords evenly spaced?
        x_centers = []
        y_centers = []
        for d in dets:
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                x_centers.append((bbox[0]+bbox[2])/2)
                y_centers.append((bbox[1]+bbox[3])/2)

        if len(x_centers) >= 3:
            x_sorted = sorted(set(round(x/50)*50 for x in x_centers))
            y_sorted = sorted(set(round(y/50)*50 for y in y_centers))
            x_gaps = [x_sorted[i+1]-x_sorted[i] for i in range(len(x_sorted)-1)]
            y_gaps = [y_sorted[i+1]-y_sorted[i] for i in range(len(y_sorted)-1)]
            x_uniform = len(set(x_gaps)) <= 2 and len(x_gaps) >= 2
            y_uniform = len(set(y_gaps)) <= 2 and len(y_gaps) >= 2
            print(f"\nHallucination check:")
            print(f"  Unique X centres (50px bins): {x_sorted}")
            print(f"  Unique Y centres (50px bins): {y_sorted}")
            print(f"  X gaps uniform: {x_uniform}  {x_gaps}")
            print(f"  Y gaps uniform: {y_uniform}  {y_gaps}")
            if x_uniform and y_uniform:
                print(f"  >> LIKELY HALLUCINATION — perfectly regular grid")
            else:
                print(f"  >> Coords appear non-uniform — possible real detections")

        return {"phase": phase_name, "elapsed": elapsed, "valid": is_valid,
                "n_dets": len(dets), "dets": dets, "raw_snippet": raw[:300]}

    except Exception as e:
        elapsed = time.time() - t0
        print(f"Time   : {elapsed:.1f}s")
        print(f"ERROR  : {e}")
        return {"phase": phase_name, "elapsed": elapsed, "valid": False, "error": str(e)}


results = {}
results["baseline"] = run_phase("BASELINE (no memory)", BASE_PROMPT)
results["memory"]   = run_phase("MEMORY (5 GT examples injected)", MEMORY_PROMPT)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for k, v in results.items():
    print(f"  {k:30s}  dets={v.get('n_dets','?'):3}  valid={v.get('valid')}  time={v.get('elapsed',0):.0f}s")

with open("/tmp/qwen_memory_test.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /tmp/qwen_memory_test.json")
