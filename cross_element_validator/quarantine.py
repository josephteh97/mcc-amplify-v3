"""
quarantine.py — Non-blocking quarantine for suspicious detections.

Quarantined detections are NOT removed from the pipeline.
They are:
  • Flagged with status="quarantined" and a reason string
  • Collected separately so the frontend EditPanel can surface them
  • Serialisable to JSON for the /api/status response

Design principle: the pipeline always produces a result. A quarantined
detection is a soft warning, not a hard failure. The user confirms,
reclassifies, or dismisses flagged items before or after rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuarantinedDetection:
    """A single detection flagged for human review."""
    detection:        dict              # original detection dict (mutated in-place)
    element_type:     str               # "column" | "wall" | "beam" | …
    reasons:          list[str]         # e.g. ["overlap_conflict:wall_023"]
    checks_failed:    list[str]         # check names that flagged this
    plausibility:     float             # 0.0–1.0 overall plausibility score
    suggested_action: str               # human-readable suggestion

    def to_dict(self) -> dict:
        return {
            "element_id":      self.detection.get("id"),
            "element_type":    self.element_type,
            "bbox_page":       self.detection.get("bbox_page"),
            "reasons":         self.reasons,
            "checks_failed":   self.checks_failed,
            "plausibility":    round(self.plausibility, 3),
            "suggested_action": self.suggested_action,
            "original_notes":  self.detection.get("notes"),
            "type_mark":       self.detection.get("type_mark"),
        }


class QuarantineManager:
    """
    Accumulates quarantined detections across all check passes.

    Usage:
        qm = QuarantineManager()
        qm.add(det, element_type="column", reasons=[...], ...)
        payload = qm.to_edit_panel_payload()
    """

    def __init__(self) -> None:
        self._items: list[QuarantinedDetection] = []

    def add(
        self,
        detection:        dict,
        element_type:     str,
        reasons:          list[str],
        checks_failed:    list[str],
        plausibility:     float,
        suggested_action: str = "Verify element type and position against structural drawings.",
    ) -> None:
        detection["_quarantined"]  = True
        detection["_plausibility"] = round(plausibility, 3)
        detection["_reasons"]      = reasons
        self._items.append(QuarantinedDetection(
            detection        = detection,
            element_type     = element_type,
            reasons          = reasons,
            checks_failed    = checks_failed,
            plausibility     = plausibility,
            suggested_action = suggested_action,
        ))

    def __len__(self) -> int:
        return len(self._items)

    def to_edit_panel_payload(self) -> list[dict]:
        """Return serialisable list for frontend EditPanel quarantine panel."""
        return [item.to_dict() for item in self._items]

    def summary(self) -> str:
        if not self._items:
            return "No quarantined detections."
        counts: dict[str, int] = {}
        for item in self._items:
            counts[item.element_type] = counts.get(item.element_type, 0) + 1
        parts = ", ".join(f"{v} {k}(s)" for k, v in counts.items())
        return f"Quarantined: {parts}"
