# Skill: detect_columns

## Description
Detect structural columns in architectural floor plan images or PDF files
using the local Ollama vision model.

## Trigger phrases
- "detect columns in ..."  /  "find columns in ..."
- "how many columns are in ..."  /  "locate the columns ..."

## Key concepts

### Grid lines & balloons
Floor plans use thin dash-dot **centrelines** defining the structural grid.
**Grid balloons** — small *open* circles at the ends of grid lines — carry axis labels
(A, B, C... / 1, 2, 3...). Do NOT confuse them with columns.

### Where columns appear
Structural columns sit at the **intersections** of grid lines as small, solid, filled shapes.

## Column shapes
| Shape       | Description                                           |
|-------------|-------------------------------------------------------|
| `square`    | Filled black/dark square at a grid intersection       |
| `rectangle` | Filled dark rectangle (non-square) at a grid point    |
| `circle`    | Filled solid circle (round column)                    |
| `i_beam`    | I-beam or H-section profile symbol                    |

## What is NOT a column
- Open/hollow circles (grid balloons)
- Dimension lines, arrowheads, wall segments, door arcs
- North arrows, scale bars, title block content

## Output format
```json
{
  "file": "floor_plan.pdf",
  "page": 0,
  "total_columns": 42,
  "detections": [
    { "id": 1, "bbox_page": [x1, y1, x2, y2],
      "shape": "square", "confidence": 0.94,
      "notes": "solid black square at grid A-3 intersection" }
  ],
  "stats": { "by_shape": {"square": 28, "circle": 10, "rectangle": 4},
             "avg_confidence": 0.87, "tiles": 48 }
}
```

Confidence is a float **0.0–1.0**.
