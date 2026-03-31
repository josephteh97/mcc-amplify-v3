"""
Agentic grid line detector — orchestration, prompts, and agentic loop.

Tool functions and memory management live in tools.py.
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama_client
from tools import (
    tool_render_pdf,
    tool_detect_grid,
    tool_verify,
    tool_zoom_margin,
    tool_reconcile,
    tool_read_reference,
    tool_memory_lookup,
    tool_memory_save_run,
    _parse_json,
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.85
MAX_RETRIES = 3


# ── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert construction document analyst. You will be shown an architectural
floor plan and must identify all grid lines and their labels.

Grid lines are the regularly-spaced reference lines that form the structural grid
of a building. They run horizontally and vertically across the floor plan and are
labelled at the margins — typically letters (A, B, C...) for one axis and numbers
(1, 2, 3...) for the other, though some plans use other conventions.

When given a floor plan image, you must:
1. Carefully examine all four margins of the drawing
2. Identify every grid line label you can see
3. Determine whether each label belongs to a horizontal or vertical grid line
4. Count the total number of distinct grid lines
5. Return ONLY valid JSON — no explanation, no markdown fences

Output format:
{
  "total_grid_lines": <int>,
  "vertical_labels": ["1", "2", "3", ...],
  "horizontal_labels": ["A", "B", "C", ...],
  "confidence": <float 0.0-1.0>,
  "notes": "<optional: anything unusual about this drawing>"
}

If you are uncertain about a label, include it with a "?" suffix e.g. "C?".
If confidence is below 0.85, explain why in the notes field.
"""

MEMORY_HINT_PROMPT = """
{system_prompt}

CORRECTION MEMORY: A human expert previously reviewed this exact floor plan and confirmed:
  Vertical grid lines:   {vertical_labels}
  Horizontal grid lines: {horizontal_labels}
  Notes: {notes}

Use this as strong prior knowledge. Verify visually, but defer to this answer
unless you see clear evidence to the contrary.
"""

VERIFICATION_PROMPT = """
Look at this floor plan again carefully. You previously identified these grid lines:
{previous_result}

Please verify by checking:
- Are there any labels in the margins you may have missed?
- Do the labels you found appear on both sides of the drawing (top/bottom or left/right)?
- Are there any labels you included that you are now less certain about?

Return the same JSON format with any corrections. If your original answer was correct,
return it unchanged.
"""

ZOOMED_PROMPT = """
This is a cropped view of the {side} margin of a floor plan.
List every grid line label you can see here, in order from left to right (or top to bottom).
Return ONLY a JSON array of label strings, e.g. ["A", "B", "C"] or ["1", "2", "3"].
"""

RECONCILE_PROMPT = """
You have performed multiple passes on a floor plan to detect grid lines.
Here are all the results:

Full image detection:
{full_result}

Margin scans:
- Top margin labels:    {top}
- Bottom margin labels: {bottom}
- Left margin labels:   {left}
- Right margin labels:  {right}

Please reconcile any differences and produce a single authoritative final answer.
Consider:
- Labels that appear on opposing margins (top/bottom or left/right) are strong evidence
- Labels found only in one scan may be noise or may be genuine (use your judgment)
- Prefer labels that appear consistently across multiple passes

Return ONLY valid JSON in this format:
{{
  "total_grid_lines": <int>,
  "vertical_labels": ["1", "2", ...],
  "horizontal_labels": ["A", "B", ...],
  "confidence": <float 0.0-1.0>,
  "notes": "<reconciliation reasoning>"
}}
"""

HINT_PROMPT = """
{system_prompt}

ADDITIONAL CONTEXT: A working approach on a similar floor plan did the following:
{hint}

Use this as guidance and try again on the image below.
"""


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _results_differ(a: dict, b: dict) -> bool:
    return (
        sorted(a.get("vertical_labels", [])) != sorted(b.get("vertical_labels", []))
        or sorted(a.get("horizontal_labels", [])) != sorted(b.get("horizontal_labels", []))
    )


def _select_floor_plan_page(image_paths: list, verbose: bool) -> str:
    if len(image_paths) == 1:
        return image_paths[0]

    # Send only page 1 as a proxy to avoid uploading all pages; ask model to pick
    prompt = (
        f"This PDF has {len(image_paths)} pages. "
        "This image is page 1. Which page number is most likely to contain "
        "the primary architectural floor plan showing the structural grid from above? "
        "If this page looks like the main floor plan, answer 1. "
        f"Otherwise choose from 1 to {len(image_paths)}. "
        'Return ONLY JSON: {"page": <int>}'
    )
    raw = ollama_client.query_vision(image_paths[0], prompt)
    result = _parse_json(raw)
    page_num = max(1, min(int(result.get("page", 1)), len(image_paths)))
    if verbose:
        print(f"  [agent] Selected page {page_num} of {len(image_paths)} as primary floor plan.")
    return image_paths[page_num - 1]


# ── AGENTIC LOOP ──────────────────────────────────────────────────────────────

def run(pdf_path: str, verbose: bool = False) -> dict:
    """
    Run the full agentic grid line detection workflow.

    Args:
        pdf_path: Path to the PDF floor plan.
        verbose:  If True, include a "steps_taken" field in the return value.

    Returns:
        dict with keys: total_grid_lines, vertical_labels, horizontal_labels,
        confidence, notes, _rendered_image, run_id.
        "used_reference_hint": True is set when the notebook fallback triggered.
        "used_correction_memory": True is set when a human correction was applied.
    """
    steps_taken = []
    used_reference_hint = False
    used_correction_memory = False
    attempt = 0
    final_result = None
    current_dpi = 200
    floor_plan_image = None
    run_id = str(uuid.uuid4())

    def log(msg):
        if verbose:
            print(f"  [agent] {msg}")
        steps_taken.append(msg)

    # ── MEMORY LOOKUP — check for human corrections before any detection ───────
    correction = tool_memory_lookup(pdf_path)
    if correction:
        detect_prompt = MEMORY_HINT_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            vertical_labels=correction["vertical_labels"],
            horizontal_labels=correction["horizontal_labels"],
            notes=correction.get("notes", ""),
        )
        used_correction_memory = True
        log(
            f"Correction memory: found prior correction — "
            f"V={correction['vertical_labels']} H={correction['horizontal_labels']}"
        )
    else:
        detect_prompt = SYSTEM_PROMPT

    while attempt < MAX_RETRIES:
        attempt += 1
        log(f"=== Attempt {attempt} (DPI={current_dpi}) ===")

        # ── STEP 1: RENDER ────────────────────────────────────────────────────
        log("Step 1: Rendering PDF to images.")
        render_result = tool_render_pdf(pdf_path, dpi=current_dpi)
        image_paths = render_result["images"]
        log(f"  Rendered {render_result['page_count']} page(s).")

        floor_plan_image = _select_floor_plan_page(image_paths, verbose)
        log(f"  Using image: {os.path.basename(floor_plan_image)}")

        # ── STEP 2: INITIAL DETECTION ─────────────────────────────────────────
        log("Step 2: Initial grid detection.")
        if used_correction_memory:
            log("  Using correction memory hint in prompt.")

        initial_result = tool_detect_grid(floor_plan_image, detect_prompt)
        log(
            f"  Initial: {initial_result.get('total_grid_lines')} lines, "
            f"confidence={initial_result.get('confidence', 0):.2f}"
        )

        # ── STEP 3: VERIFY ────────────────────────────────────────────────────
        log("Step 3: Verification pass.")
        verified_result = tool_verify(floor_plan_image, initial_result, VERIFICATION_PROMPT)
        discrepancy = _results_differ(initial_result, verified_result)
        log("  Discrepancy detected." if discrepancy else "  Verification confirmed initial result.")
        log(
            f"  Verified: {verified_result.get('total_grid_lines')} lines, "
            f"confidence={verified_result.get('confidence', 0):.2f}"
        )

        confidence = float(verified_result.get("confidence", 0))

        # ── STEP 4: MARGIN SCAN ───────────────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD or discrepancy:
            log("Step 4: Margin scan (low confidence or discrepancy).")

            # All four margin scans are independent — run in parallel
            def _scan(side, img=floor_plan_image):
                return side, tool_zoom_margin(img, side, ZOOMED_PROMPT)["labels"]

            margin_results = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_scan, s): s for s in ("top", "bottom", "left", "right")}
                for future in as_completed(futures):
                    side, labels = future.result()
                    margin_results[side] = labels
                    log(f"  {side.capitalize()} margin: {labels}")

            log("  Reconciling full detection with margin scans.")
            reconciled = tool_reconcile(verified_result, margin_results, RECONCILE_PROMPT)
            log(
                f"  Reconciled: {reconciled.get('total_grid_lines')} lines, "
                f"confidence={reconciled.get('confidence', 0):.2f}"
            )
            final_result = reconciled
            confidence = float(reconciled.get("confidence", 0))
        else:
            log("Step 4: Skipped (confidence is sufficient).")
            final_result = verified_result

        # ── STEP 5: REFERENCE FALLBACK ────────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD and not used_reference_hint:
            log("Step 5: Reference fallback — reading final_success.ipynb hint.")
            hint = tool_read_reference()
            log(f"  Hint summary: {hint[:120]}...")

            hint_prompt = HINT_PROMPT.format(system_prompt=SYSTEM_PROMPT, hint=hint)
            hinted_result = tool_detect_grid(floor_plan_image, hint_prompt)
            log(
                f"  After hint: {hinted_result.get('total_grid_lines')} lines, "
                f"confidence={hinted_result.get('confidence', 0):.2f}"
            )
            used_reference_hint = True

            if float(hinted_result.get("confidence", 0)) > confidence:
                final_result = hinted_result
                confidence = float(hinted_result.get("confidence", 0))
        elif not used_reference_hint:
            log("Step 5: Skipped (confidence is sufficient).")

        # ── STEP 6: RETRY AT HIGHER DPI ───────────────────────────────────────
        if confidence >= CONFIDENCE_THRESHOLD:
            log(f"Step 6: Confidence {confidence:.2f} >= {CONFIDENCE_THRESHOLD}. Done.")
            break

        if attempt < MAX_RETRIES:
            current_dpi = 300
            log(f"Step 6: Confidence {confidence:.2f} still low. Retrying at DPI={current_dpi}.")
        else:
            log(f"Step 6: Max retries reached. Returning best result (confidence={confidence:.2f}).")

    # ── STEP 7: FINALIZE ──────────────────────────────────────────────────────
    if final_result is None:
        final_result = {
            "total_grid_lines": 0,
            "vertical_labels": [],
            "horizontal_labels": [],
            "confidence": 0.0,
            "notes": "Detection failed to produce any result.",
        }

    v = final_result.get("vertical_labels", [])
    h = final_result.get("horizontal_labels", [])
    final_result["total_grid_lines"] = len(v) + len(h)
    final_result["run_id"] = run_id

    if floor_plan_image:
        final_result["_rendered_image"] = floor_plan_image

    if used_reference_hint:
        final_result["used_reference_hint"] = True
    if used_correction_memory:
        final_result["used_correction_memory"] = True

    if verbose:
        final_result["steps_taken"] = steps_taken

    # Persist to SQLite
    tool_memory_save_run(pdf_path, final_result, current_dpi, run_id)

    return final_result
