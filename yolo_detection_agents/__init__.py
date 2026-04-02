"""
yolo_detection_agents — Per-element YOLO detection agents for mcc-amplify-v3.

Each construction element type has its own dedicated agent backed by its own
YOLOv11 weights file.  Currently implemented:

    YOLOColumnAgent  — detects structural columns (column-detect.pt)

Planned:
    YOLOBeamAgent    — beam-detect.pt
    YOLOWallAgent    — wall-detect.pt
"""

from .column_agent import YOLOColumnAgent

__all__ = ["YOLOColumnAgent"]
