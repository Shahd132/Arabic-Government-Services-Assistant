"""
pdf_extractor.py
For DIGITAL PDFs only (already has a text layer).

FIX for Problem #5:
- Previously used `page.get_text()` which flattens everything into one
  long string and loses line structure.
- Now uses `page.get_text("blocks")` which returns text blocks
  (paragraphs, table cells, etc.) separately.
- Blocks are sorted top-to-bottom, then left-to-right, to approximate
  reading order for mixed Arabic/English government forms.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a digital PDF, preserving block/line structure.
    """
    doc = fitz.open(pdf_path)
    all_pages = []

    for page in doc:
        # blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
        # block_type: 0 = text, 1 = image
        blocks = page.get_text("blocks")

        # Sort by vertical position, then horizontal (approximate reading order)
        blocks.sort(key=lambda b: (round(b[1]), b[0]))

        page_parts = []
        for b in blocks:
            if b[6] == 0:  # text block
                block_text = b[4].strip()
                if block_text:
                    page_parts.append(block_text)

        all_pages.append("\n".join(page_parts))

    doc.close()
    return "\n\n".join(all_pages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py path/to/file.pdf")
        sys.exit(1)

    path = sys.argv[1]
    extracted = extract_text_from_pdf(path)
    print(extracted[:1500])
    print(f"\n\n[Total characters extracted: {len(extracted)}]")
