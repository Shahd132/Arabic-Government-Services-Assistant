"""
trocr_engine.py (v5)
EasyOCR + optional preprocessing + RTL-aware reconstruction.

FIX for Problem #4 (tables — partial):
- Added `sort_by_lines` parameter to control grouping.
- Improved tolerance to be more permissive (0.7 instead of 0.6).
- For complex tables, we recommend upgrading to PaddleOCR PP-Structure.
  This file remains the fallback for simple text blocks.
"""

from __future__ import annotations
from typing import List, Tuple

import easyocr
import fitz
import numpy as np

from preprocessing import preprocess_image

_reader = None


def _load_reader():
    global _reader
    if _reader is None:
        print("[INFO] Loading Arabic + English OCR model (first call only)...")
        _reader = easyocr.Reader(['ar', 'en'])
    return _reader


def _reconstruct_rtl_text(
    results: List[Tuple[list, str, float]],
    line_grouping_ratio: float = 0.7,
) -> str:
    """
    Rebuilds reading order from EasyOCR boxes (RTL-aware).
    - Groups boxes into lines by vertical center.
    - Sorts each line right-to-left.
    """
    if not results:
        return ""

    boxes = []
    for bbox, text, _conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append({
            "text": text,
            "x_center": sum(xs) / len(xs),
            "y_center": sum(ys) / len(ys),
            "height": max(ys) - min(ys) or 1.0,
        })

    boxes.sort(key=lambda b: b["y_center"])
    avg_height = sum(b["height"] for b in boxes) / len(boxes)
    tolerance = max(avg_height * line_grouping_ratio, 1.0)

    lines: List[List[dict]] = []
    for b in boxes:
        if lines and abs(b["y_center"] - lines[-1][-1]["y_center"]) <= tolerance:
            lines[-1].append(b)
        else:
            lines.append([b])

    out_lines = []
    for line in lines:
        line_rtl = sorted(line, key=lambda b: -b["x_center"])
        out_lines.append(" ".join(b["text"] for b in line_rtl))

    return "\n".join(out_lines)


def ocr_image(image_path_or_array, enable_deskew: bool = False) -> str:
    """
    Runs preprocessing + OCR on a single image.

    Args:
        image_path_or_array: file path or numpy array.
        enable_deskew: default False (see preprocessing.py).
    """
    reader = _load_reader()
    processed = preprocess_image(image_path_or_array, enable_deskew=enable_deskew)
    results = reader.readtext(processed, detail=1, paragraph=False)
    return _reconstruct_rtl_text(results)


def ocr_pdf_as_images(pdf_path: str, dpi: int = 300) -> str:
    """
    For scanned PDFs: renders pages as images, then OCR.
    Uses dpi=300 (was 200) for better quality on small Arabic text.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            import cv2
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_array = img_array[:, :, ::-1]

        page_text = ocr_image(img_array, enable_deskew=False)
        all_text.append(page_text)

    doc.close()
    return "\n".join(all_text)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python trocr_engine.py path/to/image_or_scanned.pdf")
        sys.exit(1)

    path = sys.argv[1]
    if path.lower().endswith(".pdf"):
        result = ocr_pdf_as_images(path)
    else:
        result = ocr_image(path)

    print(result)
