"""
cleaning.py
Cleans up raw text coming out of OCR (or even digital extraction --
digital PDFs can also have weird spacing/symbols).

Usage:
    from cleaning import clean_text
    clean = clean_text(raw_text)
"""

import re


def remove_unnecessary_symbols(text: str) -> str:
    """Strips characters that aren't real content: control chars, weird bullets, etc."""
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse repeated spaces/tabs (keep newlines)
    text = re.sub(r"[\u200b\u200c\u200e\u200f\ufeff]", "", text)  # invisible/formatting chars
    return text


def normalize_spaces(text: str) -> str:
    """Collapses multiple blank lines and trims each line."""
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


def normalize_arabic(text: str) -> str:
    """
    Basic Arabic normalization:
    - unify different forms of alef (أ إ آ) -> ا
    - unify ة -> ه is a common choice, but risky for meaning; we do NOT do that here
    - remove diacritics (tashkeel)
    - unify ى -> ي
    """
    # Remove Arabic diacritics (tashkeel)
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = arabic_diacritics.sub("", text)

    # Unify alef forms
    text = re.sub(r"[إأآا]", "ا", text)

    # Unify ya forms
    text = re.sub(r"ى", "ي", text)

    return text


def clean_text(raw_text: str, arabic: bool = True) -> str:
    """
    Full cleaning pipeline. Set arabic=False if you're processing
    non-Arabic text and want to skip Arabic-specific normalization.
    """
    text = remove_unnecessary_symbols(raw_text)
    text = normalize_spaces(text)
    if arabic:
        text = normalize_arabic(text)
    return text


if __name__ == "__main__":
    sample = "   إيصال   سداد   رسوم  \n\n مصلحة الأحوال المدنية  \u200b "
    print("BEFORE:")
    print(repr(sample))
    print("\nAFTER:")
    print(repr(clean_text(sample)))
