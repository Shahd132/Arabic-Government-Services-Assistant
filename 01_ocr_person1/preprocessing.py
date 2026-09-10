"""
preprocessing.py
Image cleanup applied BEFORE OCR — deskew, denoise, contrast normalization.

This was the gap flagged in trocr_engine's own docstring ("would improve
further with: image preprocessing (deskew, denoise, contrast
normalization)"). Phone photos of a form/card degrade OCR accuracy fast
without this; clean scans are more forgiving but still benefit from it.

Usage:
    from preprocessing import preprocess_image
    processed = preprocess_image("photo.jpg")   # -> np.ndarray, ready for EasyOCR
"""

from __future__ import annotations
import cv2
import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Removes speckle/sensor noise common in phone photos without
    blurring text edges too aggressively."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE (adaptive histogram equalization) — evens out uneven lighting
    across a photographed page/card better than a global contrast stretch."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Detects and corrects page/text rotation using the minimum-area
    bounding box of foreground (text) pixels. Small skew (a few degrees,
    typical of handheld photos) is what this targets -- it is not a
    general-purpose rotation detector for severely misaligned scans."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        # Not enough foreground pixels to estimate a reliable angle
        # (near-blank image) -- skip rather than risk a bogus rotation.
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Don't "correct" tiny angles that are just detector noise, and don't
    # apply a huge rotation from a bad estimate -- clamp to a sane range.
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(image_path_or_array) -> np.ndarray:
    """
    Full pipeline: load -> grayscale -> denoise -> contrast -> deskew.
    Accepts either a file path (str) or an already-loaded np.ndarray (e.g.
    a PDF page rendered by PyMuPDF), so it can slot into both ocr_image()
    and ocr_pdf_as_images() without an extra disk round-trip.

    Returns a single-channel (grayscale) np.ndarray, which EasyOCR accepts
    directly via reader.readtext(array, ...).
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
    else:
        img = image_path_or_array

    gray = _to_grayscale(img)
    gray = denoise(gray)
    gray = normalize_contrast(gray)
    gray = deskew(gray)
    return gray


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py path/to/image.jpg")
        sys.exit(1)
    out = preprocess_image(sys.argv[1])
    cv2.imwrite("preprocessed_debug.png", out)
    print(f"Wrote preprocessed_debug.png ({out.shape[1]}x{out.shape[0]})")
