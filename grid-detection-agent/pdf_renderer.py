"""
PDF → PNG image conversion utility using pdf2image (poppler backend).
"""

import tempfile
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """
    Convert each page of a PDF to a PNG image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Render resolution. 200 is a good default for label readability.

    Returns:
        List of absolute paths to the saved PNG files (one per page).

    Raises:
        PDFInfoNotInstalledError: If poppler-utils is not installed.
    """
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except PDFInfoNotInstalledError:
        raise PDFInfoNotInstalledError(
            "poppler-utils is required. Install with: sudo apt install poppler-utils"
        )

    import os
    out_dir = tempfile.mkdtemp(prefix="grid_agent_")
    image_paths = []
    for i, page in enumerate(pages):
        path = os.path.join(out_dir, f"page_{i + 1:03d}.png")
        page.save(path, "PNG")
        image_paths.append(path)

    return image_paths


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering them."""
    return pdfinfo_from_path(pdf_path)["Pages"]
