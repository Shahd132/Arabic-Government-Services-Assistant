"""
document_detector.py
Decides whether a PDF is "digital" (has real embedded text) or "scanned"
(just images of pages, needs OCR).

Usage:
    from document_detector import is_digital_pdf
    if is_digital_pdf("some_file.pdf"):
        ... use pdf_extractor.py
    else:
        ... use trocr_engine.py
"""

import fitz  # this is PyMuPDF


def is_digital_pdf(pdf_path: str, min_chars_per_page: int = 20) -> bool:
    """
    Returns True if the PDF has a real text layer (digital),
    False if it looks like a scanned image (needs OCR).

    Logic: open each page, try to pull text directly. If we consistently
    get almost nothing back, it's scanned.
    """
    doc = fitz.open(pdf_path)
    total_chars = 0
    pages_checked = 0

    for page in doc:
        text = page.get_text().strip()
        total_chars += len(text)
        pages_checked += 1
        if pages_checked >= 3:  # no need to check the whole doc, first few pages tell us enough
            break

    doc.close()

    avg_chars = total_chars / max(pages_checked, 1)
    return avg_chars >= min_chars_per_page


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_detector.py path/to/file.pdf")
        sys.exit(1)

    path = sys.argv[1]
    result = is_digital_pdf(path)
    print(f"{path} -> {'DIGITAL (extract directly)' if result else 'SCANNED (needs OCR)'}")
