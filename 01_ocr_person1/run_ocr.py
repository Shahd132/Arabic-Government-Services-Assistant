"""
run_ocr.py
Main entry point: PDF/Image → detect → extract/OCR → clean → text out.

FIXES for Problems #2, #6, #7:
- MIN_CHARS_FOR_CONFIDENCE: 15 → 30 (reject short garbled output).
- MIN_ARABIC_CHAR_RATIO: 0.3 → 0.15 (accept mixed Arabic/English documents).
- Arabic regex extended to cover:
    * Arabic Supplement (\u0750-\u077F)
    * Arabic Extended-A (\u08A0-\u08FF)
    * Arabic Presentation Forms-A (\uFB50-\uFDFF)
    * Arabic Presentation Forms-B (\uFE70-\uFEFF)
- Also uses clean_text(..., for_search=False) to keep original letters.
"""

import re
import sys
from document_detector import is_digital_pdf
from pdf_extractor import extract_text_from_pdf
from trocr_engine import ocr_image, ocr_pdf_as_images
from cleaning import clean_text

# Extended Arabic character ranges
_ARABIC_CHAR = re.compile(
    r"[\u0600-\u06FF"       # Arabic
    r"\u0750-\u077F"        # Arabic Supplement
    r"\u08A0-\u08FF"        # Arabic Extended-A
    r"\uFB50-\uFDFF"        # Arabic Presentation Forms-A
    r"\uFE70-\uFEFF]"       # Arabic Presentation Forms-B
)

MIN_CHARS_FOR_CONFIDENCE = 30
MIN_ARABIC_CHAR_RATIO = 0.15


def _looks_low_confidence(text: str) -> bool:
    """
    Runtime safety net to reject OCR output that is either:
    - Too short (< 30 chars)
    - Mostly non-Arabic (< 15% Arabic letters)
    """
    stripped = text.strip()

    if len(stripped) < MIN_CHARS_FOR_CONFIDENCE:
        return True

    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return True

    arabic_ratio = sum(1 for c in letters if _ARABIC_CHAR.match(c)) / len(letters)
    return arabic_ratio < MIN_ARABIC_CHAR_RATIO


def process_document(file_path: str) -> str:
    """
    Single entry point. Returns cleaned text or "" if unreliable.
    """
    is_pdf = file_path.lower().endswith(".pdf")

    if is_pdf:
        if is_digital_pdf(file_path):
            print("[INFO] Detected: digital PDF -> using PyMuPDF direct extraction")
            raw_text = extract_text_from_pdf(file_path)
        else:
            print("[INFO] Detected: scanned PDF -> using EasyOCR")
            raw_text = ocr_pdf_as_images(file_path)
    else:
        print("[INFO] Detected: image file -> using EasyOCR")
        raw_text = ocr_image(file_path)

    print(f"[INFO] Raw text length: {len(raw_text)} characters")

    # for_search=False → keep original letters (أحمد stays أحمد)
    cleaned = clean_text(raw_text, arabic=True, for_search=False)
    print(f"[INFO] Cleaned text length: {len(cleaned)} characters")

    if _looks_low_confidence(cleaned):
        print("[WARN] OCR output looks low-confidence -- discarding")
        return ""

    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py path/to/file.pdf")
        sys.exit(1)

    input_path = sys.argv[1]
    result = process_document(input_path)

    print("\n--- FINAL OUTPUT ---\n")
    print(result)
