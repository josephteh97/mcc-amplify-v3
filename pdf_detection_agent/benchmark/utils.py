"""
utils.py — Shared utilities for vision model benchmark scripts
"""
from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

import ollama

TIMEOUT_SECONDS = 300  # 5 min — shared across all benchmark scripts


def ollama_chat_with_timeout(timeout: int = TIMEOUT_SECONDS, **kwargs) -> Any:
    """
    Run ollama.chat(**kwargs) with a hard wall-clock timeout.
    Raises concurrent.futures.TimeoutError if the call exceeds timeout seconds.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(ollama.chat, **kwargs).result(timeout=timeout)


def validate_json(text: str) -> tuple[bool, list | None, str]:
    """
    Attempt to parse model output as a JSON array.
    Returns (is_valid, parsed_list_or_none, error_message).
    """
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
