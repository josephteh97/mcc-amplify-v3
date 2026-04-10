#!/usr/bin/env python3
"""Quick single-model test — writes result to /tmp/single_model_result.json"""
import json, sys, time, concurrent.futures
from utils import TIMEOUT_SECONDS, count_elements, ollama_chat_with_timeout, validate_json

IMAGE_PATH = "/tmp/test_floorplan_tile.png"
TAG = sys.argv[1] if len(sys.argv) > 1 else "ministral-3:8b"

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

print(f"Testing {TAG}...", flush=True)
t0 = time.time()
result = dict(tag=TAG, status="pending", elapsed=None, is_valid=False, elements={}, err="", raw="")
try:
    response = ollama_chat_with_timeout(
        TIMEOUT_SECONDS,
        model=TAG,
        messages=[{"role": "user", "content": DETECTION_PROMPT, "images": [IMAGE_PATH]}],
        options={"num_predict": 4096, "temperature": 0.1},
        keep_alive=0,
    )
    elapsed = round(time.time() - t0, 2)
    raw = response.message.content or ""
    is_valid, parsed, err = validate_json(raw)
    elements = count_elements(parsed) if (is_valid and parsed) else {}
    status = "success" if is_valid else "invalid_json"
    result.update(elapsed=elapsed, is_valid=is_valid, elements=elements, status=status, err=err, raw=raw[:3000])
except concurrent.futures.TimeoutError:
    elapsed = round(time.time() - t0, 2)
    result.update(elapsed=elapsed, status="timeout", err=f"exceeded {TIMEOUT_SECONDS}s")
except Exception as e:
    elapsed = round(time.time() - t0, 2)
    result.update(elapsed=elapsed, status="error", err=str(e)[:300])

print(f"Status : {result['status']}")
print(f"Time   : {result['elapsed']}s")
print(f"Valid  : {result['is_valid']}")
print(f"Elements: {result['elements']}")
if result['err']:
    print(f"Error  : {result['err']}")
print(f"--- RAW OUTPUT (first 2000 chars) ---")
print(result['raw'][:2000])

with open("/tmp/single_model_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to /tmp/single_model_result.json")
