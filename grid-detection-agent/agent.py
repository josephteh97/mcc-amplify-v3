"""
Agentic grid line detector — all orchestration, prompts, and tools live here.
"""

import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

import ollama_client
import pdf_renderer

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.85
MAX_RETRIES = 3
REFERENCE_SCRIPT = "/tmp/reference_approach.py"
MARGIN_FRACTION = 0.12

_reference_hint_cache: str | None = None


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


# ── TOOLS ─────────────────────────────────────────────────────────────────────

def tool_render_pdf(pdf_path: str, dpi: int = 200) -> dict:
    """Convert PDF to images. Returns image paths and page count."""
    images = pdf_renderer.pdf_to_images(pdf_path, dpi=dpi)
    return {"images": images, "page_count": len(images)}


def tool_detect_grid(image_path: str, extra_prompt: str = "") -> dict:
    """Send the full floor plan image to SEA-LION for initial grid detection."""
    prompt = extra_prompt if extra_prompt else SYSTEM_PROMPT
    raw = ollama_client.query_vision(image_path, prompt)
    return _parse_json(raw)


def tool_verify(image_path: str, previous_result: dict) -> dict:
    """Second-pass verification: send the image + previous result back to SEA-LION."""
    prompt = VERIFICATION_PROMPT.format(
        previous_result=json.dumps(previous_result, indent=2)
    )
    raw = ollama_client.query_vision(image_path, prompt)
    return _parse_json(raw)


def tool_zoom_margin(image_path: str, side: str) -> dict:
    """
    Crop a margin band from the image and query SEA-LION for labels in that band.

    Args:
        image_path: Path to the full floor plan PNG.
        side: One of "top", "bottom", "left", "right".

    Returns:
        {"labels": list[str]}
    """
    img = Image.open(image_path)
    w, h = img.size
    band = int(MARGIN_FRACTION * min(w, h))

    boxes = {
        "top":    (0, 0, w, band),
        "bottom": (0, h - band, w, h),
        "left":   (0, 0, band, h),
        "right":  (w - band, 0, w, h),
    }
    if side not in boxes:
        raise ValueError(f"side must be one of {list(boxes.keys())}, got: {side!r}")

    crop = img.crop(boxes[side])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"margin_{side}_")
    tmp_path = tmp.name
    crop.save(tmp_path, "PNG")
    tmp.close()

    try:
        prompt = ZOOMED_PROMPT.format(side=side)
        raw = ollama_client.query_vision(tmp_path, prompt)
    finally:
        os.unlink(tmp_path)

    return {"labels": _parse_json(raw, array=True)}


def tool_reconcile(full_result: dict, margin_results: dict) -> dict:
    """
    Ask SEA-LION to reconcile the full-image detection with all four margin scans.

    Args:
        full_result: dict from tool_verify or tool_detect_grid.
        margin_results: {"top": [...], "bottom": [...], "left": [...], "right": [...]}

    Returns:
        Final authoritative result dict.
    """
    prompt = RECONCILE_PROMPT.format(
        full_result=json.dumps(full_result, indent=2),
        top=margin_results.get("top", []),
        bottom=margin_results.get("bottom", []),
        left=margin_results.get("left", []),
        right=margin_results.get("right", []),
    )
    raw = ollama_client.query_text(prompt)
    return _parse_json(raw)


def tool_read_reference() -> str:
    """
    LAST RESORT ONLY — read /tmp/reference_approach.py (from final_success.ipynb)
    and return a concise summary of the meaningful detection logic.

    The caller passes this hint to SEA-LION:
        "A previous approach that worked on a similar floor plan did the following:
         {hint}. Use this as guidance and try again."
    """
    global _reference_hint_cache
    if _reference_hint_cache is not None:
        return _reference_hint_cache

    if not os.path.isfile(REFERENCE_SCRIPT):
        return (
            "Reference script not available. "
            "Generate it with: jupyter nbconvert --to script "
            "~/Documents/test-grid-detector/final_success.ipynb --stdout "
            f"> {REFERENCE_SCRIPT}"
        )

    with open(REFERENCE_SCRIPT) as f:
        code = f.read()

    skip_re = re.compile("|".join([
        r"^\s*import ",
        r"^\s*from \w+ import",
        r"^\s*get_ipython\(",
        r"^\s*#\s*In\[",
        r"^\s*display\(",
        r"^\s*plt\.",
        r"^\s*fig\.",
        r"^\s*ax\.",
        r"^\s*print\(",
        r"^\s*IPython",
        r"^\s*%",
    ]))

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        "\n".join(line for line in code.splitlines() if not skip_re.match(line)),
    ).strip()

    summary_prompt = (
        "The following is Python code from a working grid line detector for "
        "construction floor plans. Summarise in 3-5 bullet points the KEY STEPS "
        "and TECHNIQUES used to identify grid line labels — focus on what a vision "
        "model should look for, not on implementation details.\n\n"
        f"```python\n{cleaned[:4000]}\n```"
    )
    _reference_hint_cache = ollama_client.query_text(summary_prompt)
    return _reference_hint_cache


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _parse_json(raw: str, array: bool = False):
    """
    Extract and parse the first JSON object or array from a model response string.

    Args:
        raw: Raw model output, possibly wrapped in markdown fences.
        array: If True, search for a JSON array ([...]); otherwise a JSON object ({...}).

    Returns:
        Parsed list (array=True) or dict (array=False). Falls back to empty on error.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    if array:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    else:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {
                "total_grid_lines": 0,
                "vertical_labels": [],
                "horizontal_labels": [],
                "confidence": 0.0,
                "notes": f"Could not parse JSON from response: {raw[:200]}",
            }
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            return {
                "total_grid_lines": 0,
                "vertical_labels": [],
                "horizontal_labels": [],
                "confidence": 0.0,
                "notes": f"JSON parse error: {exc}. Raw: {raw[:200]}",
            }


def _results_differ(a: dict, b: dict) -> bool:
    """Return True if the two detection results have meaningfully different labels."""
    return (
        sorted(a.get("vertical_labels", [])) != sorted(b.get("vertical_labels", []))
        or sorted(a.get("horizontal_labels", [])) != sorted(b.get("horizontal_labels", []))
    )


def _select_floor_plan_page(image_paths: list, verbose: bool) -> str:
    """
    For multi-page PDFs, ask SEA-LION which page is the primary floor plan.
    Returns the path to the selected image.
    """
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
        verbose: If True, include a "steps_taken" field in the return value.

    Returns:
        dict with keys: total_grid_lines, vertical_labels, horizontal_labels,
                        confidence, notes, _rendered_image (path to floor plan PNG),
                        and (if verbose) steps_taken.
        "used_reference_hint": True is added if the fallback was triggered.
    """
    steps_taken = []
    used_reference_hint = False
    attempt = 0
    final_result = None
    current_dpi = 200
    floor_plan_image = None

    def log(msg):
        if verbose:
            print(f"  [agent] {msg}")
        steps_taken.append(msg)

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
        initial_result = tool_detect_grid(floor_plan_image)
        log(
            f"  Initial: {initial_result.get('total_grid_lines')} lines, "
            f"confidence={initial_result.get('confidence', 0):.2f}"
        )

        # ── STEP 3: VERIFY ────────────────────────────────────────────────────
        log("Step 3: Verification pass.")
        verified_result = tool_verify(floor_plan_image, initial_result)
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
            def _scan(side):
                return side, tool_zoom_margin(floor_plan_image, side)["labels"]

            margin_results = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_scan, s): s for s in ("top", "bottom", "left", "right")}
                for future in as_completed(futures):
                    side, labels = future.result()
                    margin_results[side] = labels
                    log(f"  {side.capitalize()} margin: {labels}")

            log("  Reconciling full detection with margin scans.")
            reconciled = tool_reconcile(verified_result, margin_results)
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
            hinted_result = tool_detect_grid(floor_plan_image, extra_prompt=hint_prompt)
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

    if floor_plan_image:
        final_result["_rendered_image"] = floor_plan_image

    if used_reference_hint:
        final_result["used_reference_hint"] = True

    if verbose:
        final_result["steps_taken"] = steps_taken

    return final_result
