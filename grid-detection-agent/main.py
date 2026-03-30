"""
CLI entry point for the agentic grid line detector.

Usage:
    python main.py --pdf path/to/floorplan.pdf [--verbose] [--annotate]
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont


# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect grid lines in a PDF construction floor plan using SEA-LION."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF floor plan file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step agent log and include steps_taken in JSON output.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help=(
            "Draw detected grid labels onto the rendered floor plan image "
            "and save as <pdf_stem>_annotated.png."
        ),
    )
    return parser.parse_args()


# ── ANNOTATION ────────────────────────────────────────────────────────────────

def annotate_image(pdf_path: str, result: dict) -> str:
    """
    Draw the detected grid labels at the approximate margin positions on
    the first rendered page of the floor plan.

    Returns the path to the saved annotated image.
    """
    import pdf_renderer

    images = pdf_renderer.pdf_to_images(pdf_path, dpi=200)
    if not images:
        print("Warning: could not render PDF for annotation.", file=sys.stderr)
        return ""

    img = Image.open(images[0]).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except IOError:
        font = ImageFont.load_default()

    vertical_labels = result.get("vertical_labels", [])
    horizontal_labels = result.get("horizontal_labels", [])

    # Draw vertical labels (columns) along the top margin, evenly spaced
    if vertical_labels:
        n = len(vertical_labels)
        step = w // (n + 1)
        for i, label in enumerate(vertical_labels):
            x = step * (i + 1)
            y = 10
            draw.rectangle([x - 20, y, x + 20, y + 34], fill=(255, 255, 0))
            draw.text((x - 15, y + 2), label, fill=(0, 0, 0), font=font)

    # Draw horizontal labels (rows) along the left margin, evenly spaced
    if horizontal_labels:
        n = len(horizontal_labels)
        step = h // (n + 1)
        for i, label in enumerate(horizontal_labels):
            x = 10
            y = step * (i + 1)
            draw.rectangle([x, y - 17, x + 40, y + 17], fill=(0, 200, 255))
            draw.text((x + 2, y - 14), label, fill=(0, 0, 0), font=font)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(pdf_path)), f"{stem}_annotated.png")
    img.save(out_path)
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate PDF
    if not os.path.isfile(args.pdf):
        print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    if not args.pdf.lower().endswith(".pdf"):
        print(f"Warning: file does not have a .pdf extension: {args.pdf}", file=sys.stderr)

    # ollama_client verifies SEA-LION availability on import
    try:
        import agent
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Processing: {args.pdf}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

    result = agent.run(args.pdf, verbose=args.verbose)

    # Print final JSON to stdout
    print(json.dumps(result, indent=2))

    # Annotate if requested
    if args.annotate:
        out_path = annotate_image(args.pdf, result)
        if out_path:
            print(f"\nAnnotated image saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
