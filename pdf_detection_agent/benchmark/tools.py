"""
tools.py — Structured-output plugins for qwen3.5:9b column detection
=====================================================================
Two approaches to force the thinking model to emit JSON instead of
consuming all tokens in its reasoning chain:

  1. Tool/function calling  — model MUST call report_detections(columns=[...])
  2. Structured output      — format=JSON_SCHEMA constrains token sampling

Both return the same normalised dict so callers are interchangeable.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Any

import ollama

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3.5:9b"
TIMEOUT_SECONDS = 300  # 5 min — matches overnight_benchmark

# ── Tool schema (function calling) ────────────────────────────────────────────

DETECTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_detections",
        "description": (
            "Report every structural column and grid line found in the floor plan image tile. "
            "Call this function ONCE with ALL detections. Do not call it per-element."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": (
                        "Structural load-bearing columns. These are small filled, hatched, or "
                        "outlined squares, rectangles, or circles — typically 30-80 px across — "
                        "located at grid intersections. NOT grid balloons (hollow circles at "
                        "grid line ends with axis labels like A/B/1/2)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "integer", "description": "Left edge pixel x"},
                            "y1": {"type": "integer", "description": "Top edge pixel y"},
                            "x2": {"type": "integer", "description": "Right edge pixel x"},
                            "y2": {"type": "integer", "description": "Bottom edge pixel y"},
                            "shape": {
                                "type": "string",
                                "enum": ["square", "rectangle", "round", "i_beam"],
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Detection confidence 0.0–1.0",
                            },
                        },
                        "required": ["x1", "y1", "x2", "y2", "shape", "confidence"],
                    },
                },
                "grid_lines": {
                    "type": "array",
                    "description": "Dashed or light structural reference lines forming the grid.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "integer"},
                            "y1": {"type": "integer"},
                            "x2": {"type": "integer"},
                            "y2": {"type": "integer"},
                            "orientation": {
                                "type": "string",
                                "enum": ["horizontal", "vertical"],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": ["x1", "y1", "x2", "y2", "orientation", "confidence"],
                    },
                },
            },
            "required": ["columns", "grid_lines"],
        },
    },
}

# ── Structured output JSON schema ──────────────────────────────────────────────

DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[x1, y1, x2, y2] pixel bounding box",
                    },
                    "shape": {
                        "type": "string",
                        "enum": ["square", "rectangle", "round", "i_beam"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["coordinates", "shape", "confidence"],
            },
        },
        "grid_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["horizontal", "vertical"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["coordinates", "orientation", "confidence"],
            },
        },
    },
    "required": ["columns", "grid_lines"],
}

# ── Prompts ────────────────────────────────────────────────────────────────────

_TOOL_PROMPT = (
    "You are a structural engineer reviewing an architectural floor plan tile. "
    "Call report_detections() with every structural column and grid line you can see. "
    "Columns are small solid or hatched shapes (30–80 px) at grid intersections — "
    "not the hollow circles at grid line ends (those are axis balloons, ignore them). "
    "Provide tight bounding boxes: x1/y1 = top-left corner, x2/y2 = bottom-right corner, in pixels."
)

_SCHEMA_PROMPT = (
    "You are a structural engineer reviewing an architectural floor plan tile. "
    "Output a JSON object with 'columns' (small solid/hatched shapes at grid intersections, "
    "30–80 px across) and 'grid_lines' (dashed reference lines). "
    "Use tight pixel bounding boxes as [x1, y1, x2, y2]. "
    "Do not include grid axis balloons (hollow circles at line ends with labels A/B/1/2)."
)


# ── Detection functions ────────────────────────────────────────────────────────

def detect_with_tool_calling(
    image_path: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Approach 1 — Tool/function calling.

    The model is required to call report_detections() with typed arguments,
    which guarantees structured output even for thinking-mode models.

    Returns:
        {
            "approach": "tool_calling",
            "tool_called": bool,
            "columns": [...],      # list of {x1,y1,x2,y2,shape,confidence}
            "grid_lines": [...],
            "inference_time_s": float,
            "thinking_chars": int,
            "content": str,        # any free-form text the model also output
        }
    """
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ollama.chat,
            model=model,
            messages=[{"role": "user", "content": _TOOL_PROMPT, "images": [image_path]}],
            tools=[DETECTION_TOOL],
        )
        try:
            response = future.result(timeout=TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            return {
                "approach": "tool_calling", "tool_called": False, "timed_out": True,
                "columns": [], "grid_lines": [],
                "inference_time_s": round(time.time() - t0, 2),
                "thinking_chars": 0, "content": f"timeout>{TIMEOUT_SECONDS}s",
            }
    elapsed = round(time.time() - t0, 2)

    tool_calls = getattr(response.message, "tool_calls", None) or []
    thinking = getattr(response.message, "thinking", None) or ""
    content = response.message.content or ""

    columns: list[dict] = []
    grid_lines: list[dict] = []

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn and getattr(fn, "name", None) == "report_detections":
            args = fn.arguments
            if isinstance(args, str):
                args = json.loads(args)
            columns = args.get("columns", [])
            grid_lines = args.get("grid_lines", [])

    return {
        "approach": "tool_calling",
        "tool_called": bool(tool_calls),
        "timed_out": False,
        "columns": columns,
        "grid_lines": grid_lines,
        "inference_time_s": elapsed,
        "thinking_chars": len(thinking),
        "content": content,
    }


def detect_with_structured_output(
    image_path: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Approach 2 — Structured output via format=JSON_SCHEMA.

    Constrains token sampling so every generated token must conform to
    DETECTION_SCHEMA, forcing valid JSON regardless of thinking mode.

    Returns same shape as detect_with_tool_calling() but approach='structured_output'
    and coordinates are [x1,y1,x2,y2] lists instead of separate x1/y1/x2/y2 keys.
    """
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ollama.chat,
            model=model,
            messages=[{"role": "user", "content": _SCHEMA_PROMPT, "images": [image_path]}],
            format=DETECTION_SCHEMA,
        )
        try:
            response = future.result(timeout=TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            return {
                "approach": "structured_output", "schema_enforced": True, "timed_out": True,
                "columns": [], "grid_lines": [],
                "inference_time_s": round(time.time() - t0, 2),
                "thinking_chars": 0, "content": f"timeout>{TIMEOUT_SECONDS}s",
            }
    elapsed = round(time.time() - t0, 2)

    thinking = getattr(response.message, "thinking", None) or ""
    raw = response.message.content or ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    # Normalise to same x1/y1/x2/y2 shape as tool_calling approach
    def _normalise_col(c: dict) -> dict:
        coords = c.get("coordinates", [0, 0, 0, 0])
        return {
            "x1": coords[0], "y1": coords[1],
            "x2": coords[2], "y2": coords[3],
            "shape": c.get("shape", "unknown"),
            "confidence": c.get("confidence", 0.0),
        }

    def _normalise_grid(g: dict) -> dict:
        coords = g.get("coordinates", [0, 0, 0, 0])
        return {
            "x1": coords[0], "y1": coords[1],
            "x2": coords[2], "y2": coords[3],
            "orientation": g.get("orientation", "unknown"),
            "confidence": g.get("confidence", 0.0),
        }

    columns = [_normalise_col(c) for c in parsed.get("columns", [])]
    grid_lines = [_normalise_grid(g) for g in parsed.get("grid_lines", [])]

    return {
        "approach": "structured_output",
        "schema_enforced": True,
        "timed_out": False,
        "columns": columns,
        "grid_lines": grid_lines,
        "inference_time_s": elapsed,
        "thinking_chars": len(thinking),
        "content": raw[:500],
    }


def print_result(r: dict[str, Any]) -> None:
    """Pretty-print a detection result."""
    print(f"  Approach        : {r['approach']}")
    if r["approach"] == "tool_calling":
        print(f"  Tool called     : {r['tool_called']}")
    else:
        print(f"  Schema enforced : {r.get('schema_enforced', False)}")
    if r.get("timed_out"):
        print(f"  Timed out       : True (>{TIMEOUT_SECONDS}s)")
    print(f"  Inference time  : {r['inference_time_s']}s")
    print(f"  Thinking chars  : {r['thinking_chars']}")
    print(f"  Columns found   : {len(r['columns'])}")
    print(f"  Grid lines found: {len(r['grid_lines'])}")
    if r["columns"]:
        print("  Sample columns  :")
        for c in r["columns"][:5]:
            w = c["x2"] - c["x1"]
            h = c["y2"] - c["y1"]
            print(f"    [{c['x1']},{c['y1']},{c['x2']},{c['y2']}]  {w}x{h}px  "
                  f"{c['shape']}  conf={c['confidence']:.2f}")
    if r["grid_lines"]:
        print("  Sample grid_lines:")
        for g in r["grid_lines"][:3]:
            print(f"    [{g['x1']},{g['y1']},{g['x2']},{g['y2']}]  {g['orientation']}  "
                  f"conf={g['confidence']:.2f}")


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    image = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_floorplan_tile.png"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print(f"Model : {model}")
    print(f"Image : {image}")

    print("\n" + "=" * 60)
    print("APPROACH 1 — Tool / Function Calling")
    print("=" * 60)
    r1 = detect_with_tool_calling(image, model)
    print_result(r1)

    print("\n" + "=" * 60)
    print("APPROACH 2 — Structured Output (format=schema)")
    print("=" * 60)
    r2 = detect_with_structured_output(image, model)
    print_result(r2)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if r1["tool_called"] and r1["columns"]:
        print(f"  Tool calling worked: {len(r1['columns'])} columns via function args")
    elif r2["columns"]:
        print(f"  Structured output worked: {len(r2['columns'])} columns via schema")
    else:
        print("  Both approaches returned 0 columns.")
        print("  The model may be unable to localise small symbols (30-80px) even with forced JSON.")
        print("  Next step: fine-tune qwen3.5:9b on floor plan annotations, or use YOLO/GPT-4o.")
