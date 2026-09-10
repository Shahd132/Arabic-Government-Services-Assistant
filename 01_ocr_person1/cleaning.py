"""
cleaning.py
Cleans up raw text from OCR or digital extraction.

FIX for Problem #1:
- Previously `normalize_arabic` changed أ إ آ → ا and ى → ي,
  which broke retrieval ("أحمد" stored as "احمد" and not matching).
- Now we distinguish between:
  * `for_search=True` → aggressive normalization (unify alef/ya forms)
  * `for_search=False` (default) → keep original letters, only remove diacritics
"""

import re


def remove_unnecessary_symbols(text: str) -> str:
    """Strips invisible/formatting characters and collapses spaces."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200e\u200f\ufeff]", "", text)
    return text


def normalize_spaces(text: str) -> str:
    """Collapses blank lines and trims each line."""
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def remove_diacritics(text: str) -> str:
    """Removes Arabic tashkeel (fatha, damma, kasra, etc.)."""
    return re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)


def normalize_arabic_for_search(text: str) -> str:
    """
    Aggressive normalization ONLY for search/retrieval queries.
    Unifies:
      - Alef: أ إ آ ٱ → ا
      - Yeh: ى → ي
      - Teh Marbuta: ة → ه
    """
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text


def normalize_arabic_for_display(text: str) -> str:
    """
    Light normalization for display and storage.
    Only removes diacritics; keeps original letter forms.
    """
    return remove_diacritics(text)


def clean_text(raw_text: str, arabic: bool = True, for_search: bool = False) -> str:
    """
    Full cleaning pipeline.

    Args:
        raw_text: text to clean.
        arabic: apply Arabic-specific handling.
        for_search: if True → aggressive normalization (for queries).
                    if False (default) → keep original letters (for storage).

    Returns:
        Cleaned text.
    """
    text = remove_unnecessary_symbols(raw_text)
    text = normalize_spaces(text)

    if arabic:
        if for_search:
            text = normalize_arabic_for_search(text)
        else:
            text = normalize_arabic_for_display(text)

    return text


if __name__ == "__main__":
    sample = "   إيصال   سداد   رسوم  \n\n مصلحة الأحوال المدنية  \u200b "
    print("BEFORE: ", repr(sample))
    print("DISPLAY:", repr(clean_text(sample, for_search=False)))
    print("SEARCH: ", repr(clean_text(sample, for_search=True)))
