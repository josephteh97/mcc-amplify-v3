# Skill: detect_columns

## Description
Detect structural columns in architectural floor plan images or PDF files
using OpenAI GPT-4o vision model.

## Trigger phrases
- "detect columns in ..."  /  "find columns in ..."
- "how many columns are in ..."  /  "locate the columns ..."

## What is a column
Columns are the load-bearing pillars of a concrete building — they carry the
weight of every floor and roof above. Without them, the structure cannot stand.

## Column shapes
Columns may be drawn **solid filled, outlined/hollow with thick border, or cross-hatched** —
all are valid. Column ID labels nearby (C1, C2, H-C1, H-C9...) are a strong indicator.

| Shape          | Description                                                              |
|----------------|--------------------------------------------------------------------------|
| `square`       | Square shape (filled, outlined, or hatched) at a structural position     |
| `rectangle`    | Non-square rectangle (filled, outlined, or hatched)                      |
| `round`        | Circular column — NOT a grid balloon with an axis label                  |
| `i_beam`       | I-beam or H-section profile, no outer casing                             |
| `square_round` | Round column inside a square concrete casing                             |
| `i_square`     | I-beam column inside a square concrete casing                            |

Some columns have a footing (wider base) below them — **ignore the footing**
and draw the bounding box around the column section only.

## Where columns appear
Columns sit at **intersections** of the structural grid (thin dash-dot centrelines).
**Grid balloons** — small *open* circles at grid line ends with axis labels — are NOT columns.

## What is NOT a column
- Open/hollow circles (grid balloons)
- Dimension lines, arrowheads, wall segments, door arcs
- North arrows, scale bars, title block content

## Detection pipeline
1. PDF rendered at 150 DPI via PyMuPDF
2. Sliding-window tiling (1280x1280, step 1080)
3. Pre-filter: tiles with <1% dark pixels are skipped
4. Each tile resized to 1024x1024, sent to GPT-4o with reference images
5. Structured JSON output parsed, coordinates scaled to page space
6. NMS + cell dedup across all tiles

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
             "avg_confidence": 0.87, "tiles": 48, "tiles_skipped": 5 }
}
```

Confidence is a float **0.0–1.0**.
