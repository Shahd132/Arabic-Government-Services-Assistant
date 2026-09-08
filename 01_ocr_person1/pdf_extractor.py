"""
pdf_extractor.py
For DIGITAL PDFs only (already has a text layer).
Pulls text out directly with PyMuPDF -- no OCR needed, no model, just fast.

Usage:
    from pdf_extractor import extract_text_from_pdf
    text = extract_text_from_pdf("some_file.pdf")
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a digital PDF, page by page, joined with newlines.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages_text.append(text)

    doc.close()
    full_text = "\n".join(pages_text)
    return full_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py path/to/file.pdf")
        sys.exit(1)

    path = sys.argv[1]
    extracted = extract_text_from_pdf(path)
    print(extracted[:1000])  # just print first 1000 chars as a sanity check
    print(f"\n\n[Total characters extracted: {len(extracted)}]")
