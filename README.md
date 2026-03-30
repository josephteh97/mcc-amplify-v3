## Grid Detector Agent (`grid-detection-agent/`)

Agentic workflow that reads PDF construction floor plans and detects grid lines
the way a human analyst would — by visually examining the drawing and reading
the margin labels.

**How it works:**
The agent renders the floor plan as an image and uses SEA-LION (local Ollama
vision model) to identify grid line labels. It verifies its own answer with a
second pass, scans individual margins when uncertain, and reconciles any
discrepancies before returning a final result.

**Setup:**
```bash
ollama pull aisingapore/Gemma-SEA-LION-v4-4B-VL:latest   # confirm tag: ollama list
pip install -r requirements.txt
sudo apt install poppler-utils
```

**Run:**
```bash
cd grid-detection-agent
python main.py --pdf path/to/floorplan.pdf [--verbose] [--annotate]
```

**Output:**
```json
{
  "total_grid_lines": 14,
  "vertical_labels": ["1", "2", "3", "4", "5", "6", "7", "8"],
  "horizontal_labels": ["A", "B", "C", "D", "E", "F"],
  "confidence": 0.96
}
```
