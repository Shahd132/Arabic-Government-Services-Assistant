"""
trocr_engine.py (v4 - EasyOCR + preprocessing + RTL-aware reconstruction)
For SCANNED documents / images (no text layer).
 
CHANGES FROM v3:
1. paragraph=True removed. It groups EasyOCR's detected boxes and orders
   them left-to-right, which is the WRONG reading direction for Arabic
   (RTL) -- individual words came out correctly recognized, but scrambled
   at the sentence/line level. We now read boxes with paragraph=False
   (raw per-box results + bounding boxes) and reconstruct reading order
   ourselves: group boxes into lines by vertical position, then order each
   line right-to-left by horizontal position.
2. Runs preprocessing.py (deskew, denoise, contrast normalization) before
   OCR -- previously admitted as a known gap in this file's own docstring.
 
Usage:
    from trocr_engine import ocr_image
    text = ocr_image("scanned_page.png")
 
    # or for a scanned PDF (treated as images, page by page):
    from trocr_engine import ocr_pdf_as_images
    text = ocr_pdf_as_images("scanned_file.pdf")
"""
 
from __future__ import annotations
from typing import List, Tuple
 
import easyocr
import fitz  # PyMuPDF, used here only to turn PDF pages into images
 
from preprocessing import preprocess_image
 
# Loaded once, reused across calls -- loading the model is slow, don't do it per-image.
_reader = None
 
 
def _load_reader():
    global _reader
    if _reader is None:
        print("[INFO] Loading Arabic + English OCR model (first call only)...")
        # 'ar' = Arabic, 'en' = English -- covers both languages the project needs
        _reader = easyocr.Reader(['ar', 'en'])
    return _reader
 
 
def _reconstruct_rtl_text(
    results: List[Tuple[list, str, float]],
    line_grouping_ratio: float = 0.6,
) -> str:
    """
    Rebuilds reading order from EasyOCR's raw per-box results.
 
    results: list of (bbox, text, confidence), where bbox is 4 (x, y) points.
 
    Approach:
      1. Compute each box's vertical center and height.
      2. Sort boxes top-to-bottom, then greedily group boxes into "lines"
         whose vertical centers are within `line_grouping_ratio * avg_height`
         of each other (handles OCR boxes on the same visual line landing
         at slightly different y-coordinates).
      3. Within each line, sort right-to-left by horizontal center --
         this is the Arabic reading direction, and what paragraph=True
         got wrong by assuming left-to-right.
      4. Join words within a line with spaces, join lines with newlines.
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
 
 
def ocr_image(image_path_or_array) -> str:
    """
    Runs preprocessing + OCR on a single image (file path or np.ndarray)
    and returns text in correct Arabic (RTL) reading order.
    """
    reader = _load_reader()
    processed = preprocess_image(image_path_or_array)
    # detail=1 -> raw (bbox, text, confidence) tuples we reconstruct order from,
    # instead of paragraph=True's built-in (LTR-only) grouping.
    results = reader.readtext(processed, detail=1, paragraph=False)
    return _reconstruct_rtl_text(results)
 
 
def ocr_pdf_as_images(pdf_path: str, dpi: int = 200) -> str:
    """
    For scanned PDFs: renders each page as an image, then runs OCR on each
    page. Pages are passed to ocr_image() as in-memory arrays (via
    preprocessing.preprocess_image, which accepts either a path or an
    array) -- no temp files needed.
    """
    import numpy as np
 
    doc = fitz.open(pdf_path)
    all_text = []
 
    zoom = dpi / 72  # PyMuPDF default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
 
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA -> BGR for OpenCV/EasyOCR
            import cv2
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB -> BGR
            img_array = img_array[:, :, ::-1]
 
        page_text = ocr_image(img_array)
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
