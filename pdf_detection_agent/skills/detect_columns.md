# Skill: detect_columns

## Description
Detect structural columns in architectural floor plan images or PDF files
using the local Ollama vision model.

## Trigger phrases
- "detect columns in ..."  /  "find columns in ..."
- "how many columns are in ..."  /  "locate the columns ..."

## What is a column
Columns are the load-bearing pillars of a concrete building — they carry the
weight of every floor and roof above. Without them, the structure cannot stand.

## Column shapes
| Shape          | Description                                                      |
|----------------|------------------------------------------------------------------|
| `square`       | Filled black/dark square at a grid intersection                  |
| `rectangle`    | Filled dark rectangle (non-square) at a grid point               |
| `round`        | Filled solid circle (round column)                               |
| `i_beam`       | I-beam or H-section profile symbol, no outer casing             |
| `square_round` | Round column enclosed in a square concrete casing                |
| `i_square`     | I-beam column enclosed in a square concrete casing               |

Some columns have a footing (wider base) below them — **ignore the footing**
and draw the bounding box around the column section only.

## Where columns appear
Columns sit at **intersections** of the structural grid (thin dash-dot centrelines).
**Grid balloons** — small *open* circles at grid line ends with axis labels — are NOT columns.

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
  "stats": { "by_shape": {"square": 28, "round": 10, "rectangle": 4},
             "avg_confidence": 0.87, "tiles": 48 }
}
```

Confidence is a float **0.0–1.0**.
