"""
validation/agent.py — Validation Agent (DfMA Harness)
=======================================================
Inherits from BaseAgent.  Uses ONLY tools from validation/tools.py
and its own private validation/memory.sqlite.

Responsibility:
  1. Check memory for prior resolutions on similar floor plan signatures.
  2. Run geometry_checker on the raw geometry payload.
  3. Close open wall loops with loop_closer.
  4. Persist all successful resolutions to memory.
  5. Enrich the validated payload with Singapore project context.
  6. Return a "Validated Schema" ready for the BIM-Translator.

Sends a "Refinement Request" back to the caller if geometry cannot be
corrected autonomously (used by the controller's feedback loop).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Local imports (isolation: only validation/ tools are used) ─────────────
_ROOT = Path(__file__).parent.parent
_HERE = Path(__file__).parent
for _p in (str(_ROOT), str(_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base_agent import BaseAgent, memory_first

# Load this agent's private tools under a unique module key
_T                        = BaseAgent._load_agent_tools("validation_tools", _HERE / "tools.py")
geometry_checker          = _T.geometry_checker
loop_closer               = _T.loop_closer
standard_thickness_lookup = _T.standard_thickness_lookup
memory_io                 = _T.memory_io

# ── Singapore / MCC default project context ────────────────────────────────
_DEFAULT_PROJECT_CONTEXT = {
    "project_name":                "MCC Singapore",
    "standard":                    "SS CP 65 / BCA DfMA Advisory 2021",
    "wall_thickness_interior_mm":  200,
    "wall_thickness_exterior_mm":  300,
    "floor_to_floor_mm":           3000,
    "default_column_size_mm":      200,
    "grid_snap_tolerance_mm":      50,
}


class ValidationAgent(BaseAgent):
    """
    DfMA Validation Harness.

    Payload in  (from controller, sourced from Detection Agent):
        {
          "job_id":            str,
          "pdf_path":          str,
          "pdf_hash":          str,
          "feature_signature": str,
          "grid":              {...},
          "columns":           [...],
          "walls":             [...],
          "doors":             [...],
          "windows":           [...],
          "metadata":          {...},
          "project_context":   {...}   (optional — controller may inject)
        }

    Payload out (consumed by BIM-Translator):
        {
          "ok":                bool,
          "run_id":            str,
          "agent":             "ValidationAgent",
          "validation_status": "passed" | "warnings" | "failed",
          "issues":            [...],
          "corrections":       [...],
          "refinement_request": None | {...},  — set if human re-check needed
          "geometry":          {...},           — corrected geometry
          "project_context":   {...},
        }
    """

    def __init__(self):
        super().__init__(agent_dir=Path(__file__).parent)

    @memory_first
    def _process(self, payload: dict) -> dict:
        feature_sig = payload.get("feature_signature", "Unknown")
        proj_ctx    = {**_DEFAULT_PROJECT_CONTEXT, **payload.get("project_context", {})}

        self._log(f"Feature signature: {feature_sig}")
        self._log(f"Project context: {proj_ctx.get('standard')}")

        # ── Step 1: Apply memory hint (injected by @memory_first) ─────────────
        hint = payload.get("_memory_hint")
        if hint:
            self._log(f"Memory hint active: {hint.get('summary','')[:100]}")
            # Adjust project context from memory if hint contains dimension data
            correction_data = hint.get("correction_data", {})
            if isinstance(correction_data, dict):
                proj_ctx.update({k: v for k, v in correction_data.items()
                                 if k in proj_ctx})

        # ── Step 2: Query memory for prior resolutions on this signature ───────
        prior_resolutions = memory_io.query_resolutions(feature_sig)
        if prior_resolutions:
            self._log(
                f"Found {len(prior_resolutions)} prior resolution(s) for this signature. "
                f"Top: [{prior_resolutions[0]['rule_code']}] {prior_resolutions[0]['rule_applied'][:80]}"
            )

        # ── Step 3: Run DfMA geometry checks ──────────────────────────────────
        self._log("Running geometry_checker …")
        check_result = geometry_checker(payload, project_context=proj_ctx)
        status       = check_result["status"]
        issues       = check_result["issues"]
        corrections  = check_result["corrections"]
        geometry     = check_result["geometry"]

        self._log(
            f"geometry_checker: status={status}, "
            f"{len(issues)} issue(s), {len(corrections)} correction(s)"
        )

        # ── Step 4: Close open wall loops ─────────────────────────────────────
        self._log("Running loop_closer …")
        loop_result = loop_closer(geometry)
        geometry    = loop_result["geometry"]
        if loop_result["gaps_closed"] > 0:
            self._log(
                f"loop_closer: closed {loop_result['gaps_closed']} gap(s) — "
                + "; ".join(loop_result["log"])
            )
        else:
            self._log("loop_closer: no open loops found.")

        # ── Step 5: Persist successful resolutions to memory ──────────────────
        for corr in corrections:
            rule   = corr.get("rule", "")
            field  = corr.get("field", "")
            # Infer element_type from field path
            e_type = field.split("[")[0] if "[" in field else field.split(".")[0]
            memory_io.save_resolution(
                feature_signature = feature_sig,
                element_type      = e_type,
                rule_code         = rule,
                original_value    = corr.get("original"),
                corrected_value   = corr.get("corrected"),
                rule_applied      = (
                    standard_thickness_lookup(e_type).get("standard", rule)
                    if e_type in ("wall", "column") else rule
                ),
            )

        # Also save to agent_corrections (inherited memory) for @memory_first
        for corr in corrections:
            self._save_correction(
                feature_signature = feature_sig,
                error_code        = corr.get("rule", ""),
                error_pattern     = corr.get("field", ""),
                correction_desc   = (
                    f"Changed {corr['field']} from {corr['original']} → {corr['corrected']}"
                ),
                correction_data   = {
                    corr["field"]: corr["corrected"],
                },
            )

        # ── Step 6: Persist run summary ───────────────────────────────────────
        memory_io.save_run(
            run_id            = self.run_id,
            feature_signature = feature_sig,
            status            = status,
            issues_count      = len(issues),
            corrections_count = len(corrections),
        )

        # ── Step 7: Human-readable lessons learned ────────────────────────────
        if corrections:
            self._append_lesson({
                "agent":             "ValidationAgent",
                "feature_signature": feature_sig,
                "corrections":       corrections,
                "status":            status,
            })

        # ── Step 8: Determine if refinement request is needed ─────────────────
        # Fatal errors that the agent cannot auto-correct (C3, W2)
        fatal_codes  = {"C3", "W2"}
        fatal_issues = [iss for iss in issues if iss["code"] in fatal_codes and iss["severity"] == "error"]
        refinement_request = None
        if fatal_issues:
            refinement_request = {
                "type":    "geometry_refinement",
                "reason":  "Uncorrectable geometry errors require re-detection",
                "issues":  fatal_issues,
                "hint":    (
                    "Detection Agent should re-run at higher DPI or with "
                    "manual coordinate correction for flagged elements."
                ),
            }
            self._log(
                f"Refinement request raised: {len(fatal_issues)} fatal issue(s). "
                f"Codes: {[f['code'] for f in fatal_issues]}"
            )

        # ── Step 9: Build validated payload ───────────────────────────────────
        validated_payload = {
            "job_id":              payload.get("job_id"),
            "pdf_path":            payload.get("pdf_path"),
            "pdf_hash":            payload.get("pdf_hash"),
            "feature_signature":   feature_sig,
            "validation_status":   status,
            "issues":              issues,
            "corrections":         corrections,
            "refinement_request":  refinement_request,
            "geometry":            geometry,
            "project_context":     proj_ctx,
        }

        self._log(
            f"Validation complete: {status}. "
            f"Refinement needed: {refinement_request is not None}"
        )
        return validated_payload


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <raw_geometry.json> [project_context.json]")
        sys.exit(1)

    raw_path = sys.argv[1]
    ctx_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(raw_path, encoding="utf-8") as f:
        payload = json.load(f)

    if ctx_path:
        with open(ctx_path, encoding="utf-8") as f:
            payload["project_context"] = json.load(f)

    agent  = ValidationAgent()
    result = agent.run(payload)
    print(json.dumps(result, indent=2))
