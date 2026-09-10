"""
preprocessing.py
Image cleanup BEFORE OCR.

FIX for Problem #3:
- `deskew` is now OPTIONAL and DISABLED by default.
  Government forms are usually already straight.
  Deskew was over-correcting and hurting OCR on well-aligned documents.
- `denoise` and `contrast` remain always on (they always help).
"""

from __future__ import annotations
import cv2
import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Removes speckle noise without blurring text edges."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE — evens out uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Corrects page rotation. Only applies for angles between
    0.5 and 10 degrees. Disabled by default.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 100:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if angle is clearly skewed
    if abs(angle) < 0.5 or abs(angle) > 10:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(image_path_or_array, enable_deskew: bool = False) -> np.ndarray:
    """
    Full pipeline: load → grayscale → denoise → contrast → (deskew).

    Args:
        image_path_or_array: file path or numpy array.
        enable_deskew: default False (see note above).

    Returns:
        Grayscale np.ndarray.
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

    if enable_deskew:
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
