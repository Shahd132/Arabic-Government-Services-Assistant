"""
trocr_engine.py (v3 - EasyOCR, working version)
For SCANNED documents / images (no text layer).

HISTORY / WHY EASYOCR:
We originally tried TrOCR (microsoft/trocr-base-printed), a Transformer-based
OCR model. It works well on English but was trained mostly on Latin script,
so on Arabic input it produced fluent-looking but meaningless English-like
text (hallucination) -- it doesn't actually know Arabic letterforms.

EasyOCR has built-in Arabic support (no fine-tuning needed to get started)
and correctly reads Arabic characters. Current accuracy is still limited by
input image quality (photos vs. clean scans) and would improve further with:
  - image preprocessing (deskew, denoise, contrast normalization)
  - fine-tuning on the project's Arabic OCR dataset

This is documented as a known limitation / next step, not treated as final.

Usage:
    from trocr_engine import ocr_image
    text = ocr_image("scanned_page.png")

    # or for a scanned PDF (treated as images, page by page):
    from trocr_engine import ocr_pdf_as_images
    text = ocr_pdf_as_images("scanned_file.pdf")
"""

import easyocr
import fitz  # PyMuPDF, used here only to turn PDF pages into images

# Loaded once, reused across calls -- loading the model is slow, don't do it per-image.
_reader = None


def _load_reader():
    global _reader
    if _reader is None:
        print("[INFO] Loading Arabic + English OCR model (first call only)...")
        # 'ar' = Arabic, 'en' = English -- covers both languages the project needs
        _reader = easyocr.Reader(['ar', 'en'])
    return _reader


def ocr_image(image_path: str) -> str:
    """
    Runs OCR on a single image file and returns the extracted text.
    `paragraph=True` groups nearby text into readable blocks instead of
    returning every single detected word/box separately.
    """
    reader = _load_reader()
    results = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)


def ocr_pdf_as_images(pdf_path: str, dpi: int = 200) -> str:
    """
    For scanned PDFs: renders each page as an image, then runs OCR on each page.
    Returns all pages' text joined together.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    zoom = dpi / 72  # PyMuPDF default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = f"/tmp/_page_{page_num}.png"
        pix.save(img_path)

        page_text = ocr_image(img_path)
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
