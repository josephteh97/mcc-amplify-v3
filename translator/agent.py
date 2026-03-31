"""
translator/agent.py — BIM-Translator Agent
============================================
Inherits from BaseAgent.  Uses ONLY tools from translator/tools.py
and its own private translator/memory.sqlite.

Responsibility:
  1. Check memory for prior API success patterns before building.
  2. Transform validated geometry to world-mm coordinates.
  3. Map geometry to Revit Transaction JSON (ModelBuilder.cs schema).
  4. Send to the Windows Revit Add-in via revit_api_client.
  5. Self-correction loop on Revit API failures:
       a. Log error to translator/memory.sqlite.
       b. Query memory for similar past corrections.
       c. Adjust transaction JSON and retry (up to MAX_RETRIES).
       d. If still failing, emit a "Refinement Request" to the controller.
  6. Persist API success patterns.

Output: { "ok", "rvt_path", "warnings", "error_log", "run_id", "job_id",
          "refinement_request": None | {...} }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Local imports (isolation: only translator/ tools are used) ─────────────
_ROOT = Path(__file__).parent.parent
_HERE = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base_agent import BaseAgent, memory_first

# Load this agent's private tools under a unique module key
_T                     = BaseAgent._load_agent_tools("translator_tools", _HERE / "tools.py")
coordinate_transformer = _T.coordinate_transformer
revit_schema_mapper    = _T.revit_schema_mapper
revit_api_client       = _T.revit_api_client
memory_io              = _T.memory_io

MAX_RETRIES = 3   # self-correction attempts before escalating

# ── Error pattern → correction lookup (seed from v1 experience) ────────────
# Extend at runtime via memory_io.save_pattern() after each failure.
_KNOWN_CORRECTIONS = {
    "Wall cannot be created":
        "Ensure wall length > 0 and start ≠ end point.",
    "Invalid family symbol":
        "Family not loaded in session — call load_family before place_instance.",
    "extrusion error":
        "Column section < 200 mm. Apply DfMA minimum (200 mm).",
    "overlapping":
        "Duplicate element at same location — remove duplicate from columns list.",
    "HostObject is not valid":
        "Door/window host wall ID does not exist — set host_wall_id to null.",
    "Level not found":
        "Level name mismatch. Ensure levels list is created before placing elements.",
}


class BIMTranslatorAgent(BaseAgent):
    """
    BIM-Translator Agent.

    Payload in (from ValidationAgent via controller):
        {
          "job_id":            str,
          "pdf_path":          str,
          "pdf_hash":          str,
          "feature_signature": str,
          "validation_status": str,
          "geometry":          {...},      — corrected geometry with pixel coords
          "project_context":   {...},
          "_validation_issues": [...],    — injected by controller for context
        }

    Payload out:
        {
          "ok":                 bool,
          "run_id":             str,
          "agent":              "BIMTranslatorAgent",
          "job_id":             str,
          "rvt_path":           str | None,
          "warnings":           [str],
          "error_log":          str | None,
          "refinement_request": None | {type, reason, issues, hint},
          "element_counts":     dict,
          "transaction_json":   dict,     — included for audit / re-run
        }
    """

    def __init__(self):
        super().__init__(agent_dir=Path(__file__).parent)

    @memory_first
    def _process(self, payload: dict) -> dict:
        job_id      = payload.get("job_id", self.run_id)
        feature_sig = payload.get("feature_signature", "Unknown")
        proj_ctx    = payload.get("project_context", {})
        geometry    = payload.get("geometry", {})

        self._log(f"job_id={job_id} | feature_sig={feature_sig}")

        # ── Step 1: Memory hint from @memory_first ─────────────────────────────
        hint = payload.get("_memory_hint")
        if hint:
            self._log(f"Memory hint: {hint.get('summary','')[:120]}")
            # Adjust project context from past correction data
            cd = hint.get("correction_data", {})
            if isinstance(cd, dict):
                proj_ctx = {**proj_ctx, **cd}

        # ── Step 2: Query memory for prior success patterns ────────────────────
        prior_cols = memory_io.query_patterns("column", outcome="success")
        if prior_cols:
            self._log(
                f"Memory: {len(prior_cols)} prior column success pattern(s). "
                f"Top: {prior_cols[0].get('family_name')}"
            )

        # ── Step 3: Transform pixel coordinates → world mm ─────────────────────
        self._log("Running coordinate_transformer …")
        transform_result = coordinate_transformer(geometry, project_context=proj_ctx)

        if not transform_result["ok"]:
            return self._fail(job_id, transform_result.get("error", "coordinate_transformer failed"))

        world_geom = transform_result["world_geometry"]
        scale_info = transform_result["scale_info"]
        if transform_result["warnings"]:
            for w in transform_result["warnings"]:
                self._log(f"  [scale warning] {w}")

        self._log(
            f"Scale: {scale_info['px_per_mm']:.4f} px/mm "
            f"(source: {scale_info['source']})"
        )

        # ── Step 4: Map to Revit Transaction JSON ─────────────────────────────
        self._log("Running revit_schema_mapper …")
        map_result = revit_schema_mapper(world_geom, project_context=proj_ctx,
                                          scale_info=scale_info)

        if not map_result["ok"]:
            return self._fail(job_id, map_result.get("error", "revit_schema_mapper failed"))

        transaction_json = map_result["transaction_json"]
        el_counts        = map_result["element_counts"]
        self._log(
            f"Transaction: {el_counts.get('columns',0)} columns, "
            f"{el_counts.get('walls',0)} walls, "
            f"{el_counts.get('grids',0)} grid lines"
        )

        if map_result["unmapped_warnings"]:
            for w in map_result["unmapped_warnings"]:
                self._log(f"  [family warning] {w}")

        # ── Step 5: Send to Revit Add-in with self-correction loop ─────────────
        final_result = None
        last_error   = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._log(f"Revit API call — attempt {attempt}/{MAX_RETRIES}")
            api_result = revit_api_client(transaction_json, job_id=job_id)

            if api_result["ok"]:
                self._log(
                    f"Revit build SUCCESS: {api_result['rvt_path']} | "
                    f"{len(api_result['warnings'])} warning(s)"
                )
                # Persist success patterns for all elements
                for el_type in ("column", "wall", "door", "window"):
                    elements = transaction_json.get(el_type + "s", [])
                    for el in elements[:5]:  # sample first 5 to avoid DB bloat
                        memory_io.save_pattern(
                            element_type       = el_type,
                            family_name        = el.get("family", ""),
                            type_name          = el.get("type_name", ""),
                            parameters         = {k: el.get(k) for k in ("width", "depth", "height")},
                            outcome            = "success",
                        )
                final_result = api_result
                break

            # ── Failure: self-correction ───────────────────────────────────────
            error_text = api_result.get("error_log", "")
            last_error = error_text
            self._log(f"  [Revit error] {error_text[:200]}")

            # Log the failure
            memory_io.save_pattern(
                element_type    = "unknown",
                family_name     = "",
                type_name       = "",
                parameters      = {},
                outcome         = "failure",
                error_message   = error_text,
            )
            self._save_correction(
                feature_signature = feature_sig,
                error_code        = "REVIT_API",
                error_pattern     = error_text[:100],
                correction_desc   = "Revit API error — pending correction",
            )

            # Query memory for a past correction on this error
            past_failures = memory_io.query_patterns(
                "unknown", outcome="failure",
                error_substring=error_text[:60],
            )
            if past_failures and past_failures[0].get("correction_applied"):
                corr_hint = past_failures[0]["correction_applied"]
                self._log(f"  Memory correction hint: {corr_hint[:120]}")
            else:
                corr_hint = _match_known_correction(error_text)
                if corr_hint:
                    self._log(f"  Built-in correction: {corr_hint}")

            if not corr_hint:
                self._log("  No correction found — will escalate after retries.")
                continue

            # Apply correction to transaction JSON
            transaction_json, applied = _apply_correction(
                transaction_json, error_text, corr_hint
            )
            if applied:
                self._log(f"  Applied correction: {applied}")
                memory_io.save_pattern(
                    element_type       = "unknown",
                    family_name        = "",
                    type_name          = "",
                    parameters         = {},
                    outcome            = "failure",
                    error_message      = error_text,
                    correction_applied = applied,
                )

        # ── Step 6: Handle unresolved failure → refinement request ────────────
        refinement_request = None
        if final_result is None or not final_result.get("ok"):
            refinement_request = {
                "type":   "revit_api_failure",
                "reason": "BIM-Translator exhausted all self-correction attempts",
                "error":  last_error,
                "hint": (
                    "Validation Agent should check geometry for topology errors. "
                    "Ensure all column centres snap to grid intersections."
                ),
            }
            self._log(
                f"Refinement request emitted after {MAX_RETRIES} failed attempt(s)."
            )
            # Append to human-readable lessons log
            self._append_lesson({
                "agent":             "BIMTranslatorAgent",
                "feature_signature": feature_sig,
                "error":             last_error,
                "action":            "Escalated to controller for re-validation",
            })

        # ── Step 7: Persist run summary ────────────────────────────────────────
        rvt_path = (final_result or {}).get("rvt_path")
        warnings = (final_result or {}).get("warnings", [])
        memory_io.save_run(
            run_id          = self.run_id,
            job_id          = job_id,
            status          = "success" if (final_result and final_result.get("ok")) else "failure",
            elements_placed = sum(el_counts.values()),
            revit_warnings  = warnings,
            error_message   = last_error or "",
        )

        return {
            "job_id":             job_id,
            "rvt_path":           rvt_path,
            "warnings":           warnings,
            "error_log":          last_error,
            "refinement_request": refinement_request,
            "element_counts":     el_counts,
            "transaction_json":   transaction_json,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fail(self, job_id: str, error: str) -> dict:
        self._log(f"[FAIL] {error}")
        return {
            "ok":                False,
            "job_id":            job_id,
            "rvt_path":          None,
            "warnings":          [],
            "error_log":         error,
            "refinement_request": None,
            "element_counts":    {},
            "transaction_json":  {},
        }


# ── Correction helpers ────────────────────────────────────────────────────────

def _match_known_correction(error_text: str) -> str:
    """Return a built-in correction hint for a known Revit error pattern."""
    error_lower = error_text.lower()
    for pattern, correction in _KNOWN_CORRECTIONS.items():
        if pattern.lower() in error_lower:
            return correction
    return ""


def _apply_correction(
    transaction: dict,
    error_text: str,
    correction_hint: str,
) -> tuple[dict, str]:
    """
    Apply a heuristic correction to the transaction JSON based on the error.
    Returns (modified_transaction, description_of_what_changed).
    """
    applied = ""
    error_lower = error_text.lower()

    # Remove zero-length walls
    if "wall cannot be created" in error_lower or "zero" in error_lower:
        before = len(transaction.get("walls", []))
        transaction["walls"] = [
            w for w in transaction.get("walls", [])
            if not _is_zero_length(w)
        ]
        removed = before - len(transaction["walls"])
        if removed:
            applied = f"Removed {removed} zero-length wall(s)"

    # Clamp column sections to 200 mm minimum
    if "extrusion" in error_lower or "section" in error_lower:
        clamped = 0
        for col in transaction.get("columns", []):
            for dim in ("width", "depth"):
                if col.get(dim, 999) < 200:
                    col[dim] = 200
                    clamped += 1
        if clamped:
            applied = f"Clamped {clamped} column dimension(s) to 200 mm minimum"

    # Remove duplicate columns (same location rounded to 50 mm)
    if "overlapping" in error_lower or "duplicate" in error_lower:
        seen: set = set()
        unique_cols = []
        for col in transaction.get("columns", []):
            loc  = col.get("location", {})
            key  = (round(loc.get("x", 0) / 50) * 50,
                    round(loc.get("y", 0) / 50) * 50)
            if key not in seen:
                seen.add(key)
                unique_cols.append(col)
        removed = len(transaction.get("columns", [])) - len(unique_cols)
        if removed:
            transaction["columns"] = unique_cols
            applied = f"Removed {removed} duplicate column(s)"

    # Null out invalid host_wall_id for doors/windows
    if "hostobject" in error_lower:
        for el in transaction.get("doors", []) + transaction.get("windows", []):
            el["host_wall_id"] = None
        applied = "Cleared invalid host_wall_id on all openings"

    return transaction, applied


def _is_zero_length(wall: dict) -> bool:
    s = wall.get("start_point", {})
    e = wall.get("end_point",   {})
    dx = abs(s.get("x", 0) - e.get("x", 0))
    dy = abs(s.get("y", 0) - e.get("y", 0))
    return dx < 1e-6 and dy < 1e-6


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <validated_payload.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)

    agent  = BIMTranslatorAgent()
    result = agent.run(payload)
    print(json.dumps(result, indent=2, default=str))
