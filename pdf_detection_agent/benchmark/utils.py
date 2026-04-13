"""
utils.py — Shared utilities for vision model benchmark scripts
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import Any

import cv2
import numpy as np
import ollama
from PIL import Image as _PIL_Image

TIMEOUT_SECONDS = 300  # 5 min — shared across all benchmark scripts
EVAL_TOL        = 25   # centroid match tolerance (px)
DEDUP_TOL       = 20   # centroid dedup tolerance (px)
RAW_TRUNCATE    = 600  # characters to keep from raw model output

P_BASE = """Analyze this architectural floor plan tile and detect ALL structural columns.
Columns are small gray-filled squares/rectangles (~20-40px) at structural grid intersections,
labeled nearby with C1, C2, RCB2, SB1, etc.
Return ONLY a JSON list:
[{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}]
Coordinates within this 1024x1024 image. Every column must be detected."""


def P_ANCHOR(pool: list) -> str:
    shown = pool[:40]
    ex    = "\n".join(f"  {b}" for b in shown)
    tail  = f"\n  ... and {len(pool)-40} more confirmed" if len(pool) > 40 else ""
    return (f"I have confirmed {len(pool)} structural columns in this 1024x1024 floor plan:\n"
            f"{ex}{tail}\n"
            f"These are small gray-filled squares at structural grid intersections.\n"
            f"Find ALL REMAINING columns NOT in the list above.\n"
            f"Return ONLY a JSON list:\n"
            f'[{{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}}]\n'
            f"Include ONLY new columns not already listed.")


def P_ADDONLY(known_boxes: list) -> str:
    ex = "\n".join(f"  {b}" for b in known_boxes[:30])
    return (f"I have already found {len(known_boxes)} structural columns in this 1024x1024 floor plan:\n"
            f"{ex}{'...' if len(known_boxes) > 30 else ''}\n"
            f"Find ADDITIONAL columns NOT in the list above.\n"
            f"Return ONLY a JSON list:\n"
            f'[{{"type":"column","bbox":[x1,y1,x2,y2],"confidence":0.95}}]\n'
            f"Do NOT repeat columns already listed.")


def ollama_chat_with_timeout(timeout: int = TIMEOUT_SECONDS, **kwargs) -> Any:
    """Run ollama.chat(**kwargs) with a hard wall-clock timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(ollama.chat, **kwargs).result(timeout=timeout)


def validate_json(text: str) -> tuple[bool, list | None, str]:
    """Attempt to parse model output as a JSON array.
    Returns (is_valid, parsed_list_or_none, error_message)."""
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
            return False, None, "dict without known array key"
        return False, None, "not array/dict"
    except json.JSONDecodeError:
        pass

    # Extract [...] using plain string search — avoids greedy regex backtracking
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return True, parsed, ""
        except json.JSONDecodeError:
            pass

    # Last resort: collect individual {...} objects
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
    """Count detection elements in a single pass. Keys: total, columns, grid_lines,
    other, valid_coords, valid_conf."""
    columns = grid_lines = valid_coords = valid_conf = other = 0
    for d in dets:
        et = d.get("element_type")
        if et == "column":
            columns += 1
        elif et == "grid_line":
            grid_lines += 1
        else:
            other += 1
        if isinstance(d.get("coordinates"), list) and len(d["coordinates"]) == 4:
            valid_coords += 1
        if isinstance(d.get("confidence"), (int, float)):
            valid_conf += 1
    return {
        "total": len(dets),
        "columns": columns,
        "grid_lines": grid_lines,
        "other": other,
        "valid_coords": valid_coords,
        "valid_conf": valid_conf,
    }


def extract_gt_boxes(annot_path: str, target_size: int = 1024) -> list:
    """Extract GT column boxes from a human-annotated image via red HSV mask."""
    annot = cv2.imread(annot_path)
    hsv   = cv2.cvtColor(annot, cv2.COLOR_BGR2HSV)
    m1    = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    m2    = cv2.inRange(hsv, (165, 100, 80), (180, 255, 255))
    red   = cv2.dilate(cv2.bitwise_or(m1, m2),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    S = target_size / 1280
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 200 < cv2.contourArea(c) < 15000 and w <= 200 and h <= 200:
            cx, cy = (x + x + w) // 2, (y + y + h) // 2
            hw = max(12, int(w * S * 0.5))
            hh = max(12, int(h * S * 0.5))
            boxes.append([int(cx*S)-hw, int(cy*S)-hh, int(cx*S)+hw, int(cy*S)+hh])
    boxes.sort(key=lambda b: (b[1]//80, b[0]))
    return boxes


def extract_bboxes(dets: list) -> list:
    """Extract [x1,y1,x2,y2] from detection dicts or raw bbox lists."""
    out = []
    for d in dets:
        b = d if isinstance(d, list) else (d.get("bbox") or d.get("box_2d") or d.get("coordinates") or [])
        if len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            out.append(b)
    return out


def evaluate(dets: list, gt: list, tol: int = EVAL_TOL) -> dict:
    """Compute TP/FP/FN/P/R/F1 for detections against GT boxes."""
    matched = set(); tp = 0
    boxes = extract_bboxes(dets)
    for b in boxes:
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        for i, g in enumerate(gt):
            if i in matched: continue
            if abs(cx-(g[0]+g[2])/2) < tol and abs(cy-(g[1]+g[3])/2) < tol:
                tp += 1; matched.add(i); break
    fp = len(boxes) - tp; fn = len(gt) - tp
    p = tp/(tp+fp) if tp+fp > 0 else 0
    r = tp/(tp+fn) if tp+fn > 0 else 0
    f1 = 2*p*r/(p+r) if p+r > 0 else 0
    return dict(tp=tp, fp=fp, fn=fn, p=round(p,3), r=round(r,3), f1=round(f1,3))


def get_tp_boxes(dets: list, gt: list, tol: int = EVAL_TOL) -> list:
    """Return detected bboxes that match a GT box (true positives only)."""
    matched = set(); tp_boxes = []
    for b in extract_bboxes(dets):
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        for i, g in enumerate(gt):
            if i in matched: continue
            if abs(cx-(g[0]+g[2])/2) < tol and abs(cy-(g[1]+g[3])/2) < tol:
                tp_boxes.append(b); matched.add(i); break
    return tp_boxes


def dedup(boxes: list, tol: int = DEDUP_TOL) -> list:
    """Deduplicate boxes by centroid proximity, keeping the first occurrence."""
    kept = []
    for b in boxes:
        if not (isinstance(b, list) and len(b) == 4): continue
        if not all(isinstance(v, (int, float)) for v in b): continue
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        if all(abs(cx-(k[0]+k[2])/2) > tol or abs(cy-(k[1]+k[3])/2) > tol for k in kept):
            kept.append(b)
    return kept


def build_tile(clean_path: str, size: int = 1024) -> str:
    """Resize clean_path to size×size, cache to /tmp, return the path."""
    tile_path = f"/tmp/ami_tile_{size}.png"
    if not os.path.exists(tile_path):
        _PIL_Image.open(clean_path).resize((size, size), _PIL_Image.LANCZOS).save(tile_path)
    return tile_path


def run_inference(tag: str, prompt: str, tile_path: str, timeout_s: int = 420) -> dict:
    """Run Ollama vision inference and return parsed detections."""
    t0 = time.time()
    try:
        resp = ollama_chat_with_timeout(
            timeout_s, model=tag,
            messages=[{"role": "user", "content": prompt, "images": [tile_path]}],
            options={"num_predict": 4096, "temperature": 0.1},
            keep_alive=0,
        )
        raw = resp["message"]["content"] if resp else ""
        err = None
    except Exception as e:
        raw = ""; err = str(e)
    elapsed = time.time() - t0
    ok, parsed, _ = validate_json(raw)
    dets = parsed if (ok and isinstance(parsed, list)) else []
    return dict(elapsed=round(elapsed,1), raw=raw[:RAW_TRUNCATE], valid=ok,
                dets=dets, n_dets=len(dets), error=err)
