"""
run_ocr.py
THIS IS THE FILE YOU RUN. It ties everything together:
 
  PDF/Image in -> detect digital vs scanned -> extract or OCR -> clean -> text out
 
Usage:
    python run_ocr.py path/to/some_file.pdf
    python run_ocr.py path/to/some_image.png
"""
 
import re
import sys
from document_detector import is_digital_pdf
from pdf_extractor import extract_text_from_pdf
from trocr_engine import ocr_image, ocr_pdf_as_images
from cleaning import clean_text
 
# Arabic block, so we can tell "real Arabic text" apart from OCR noise
# (mis-recognized glyphs tend to produce Latin/symbol garbage or near-empty
# output on Arabic government forms).
_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")
 
MIN_CHARS_FOR_CONFIDENCE = 15          # anything shorter is treated as "OCR basically failed"
MIN_ARABIC_CHAR_RATIO = 0.3            # below this, treat output as garbled/noise
 
 
def _looks_low_confidence(text: str) -> bool:
    """Heuristic gate: catches the two most common OCR-failure signatures
    (near-empty output, or output that isn't mostly Arabic characters)
    before this text ever reaches the router/retrieval/generation stages
    downstream. This is NOT a substitute for the CER/WER numbers in
    evaluate_ocr.py -- it's a cheap runtime safety net for the cases those
    offline metrics can't catch (a single bad scan in production)."""
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
    This is your module's single entry point -- the function the rest
    of the team will eventually call. Give it a file path, get clean text back.
 
    Returns "" if the extracted/OCR'd text looks unreliable, rather than
    silently handing garbled text downstream as if it were trustworthy
    document content -- graph.py already treats empty ocr_text as "no
    useful document text available".
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
 
    if _looks_low_confidence(cleaned):
        print("[WARN] OCR output looks low-confidence (too short or not mostly Arabic) -- discarding")
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

