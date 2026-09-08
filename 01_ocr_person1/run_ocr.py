"""
run_ocr.py
THIS IS THE FILE YOU RUN. It ties everything together:

  PDF/Image in -> detect digital vs scanned -> extract or OCR -> clean -> text out

Usage:
    python run_ocr.py path/to/some_file.pdf
    python run_ocr.py path/to/some_image.png
"""

import sys
from document_detector import is_digital_pdf
from pdf_extractor import extract_text_from_pdf
from trocr_engine import ocr_image, ocr_pdf_as_images
from cleaning import clean_text


def process_document(file_path: str) -> str:
    """
    This is your module's single entry point -- the function the rest
    of the team will eventually call. Give it a file path, get clean text back.
    """
    is_pdf = file_path.lower().endswith(".pdf")

    if is_pdf:
        if is_digital_pdf(file_path):
            print("[INFO] Detected: digital PDF -> using PyMuPDF direct extraction")
            raw_text = extract_text_from_pdf(file_path)
        else:
            print("[INFO] Detected: scanned PDF -> using TrOCR")
            raw_text = ocr_pdf_as_images(file_path)
    else:
        print("[INFO] Detected: image file -> using TrOCR")
        raw_text = ocr_image(file_path)

    print(f"[INFO] Raw text length: {len(raw_text)} characters")

    cleaned = clean_text(raw_text)
    print(f"[INFO] Cleaned text length: {len(cleaned)} characters")

    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py path/to/file.pdf")
        sys.exit(1)

    input_path = sys.argv[1]
    result = process_document(input_path)

    print("\n--- FINAL OUTPUT ---\n")
    print(result)
