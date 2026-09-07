"""
Arabic text normalization utilities.
Unifies different letter forms so the same word is represented
consistently (important for search/retrieval quality later).
This module only exports functions - it doesn't run a pipeline by itself.
"""

import re

# Arabic diacritics (tashkeel) - e.g. fatha, damma, kasra, sukun, etc.
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')


def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text:
    - Unifies alef forms: أ، إ، آ -> ا
    - Unifies yeh forms: ى -> ي
    - Unifies teh marbuta: ة -> ه (at the end of words)
    - Removes diacritics (tashkeel) and the tatweel elongation character
    """
    if not text:
        return ""

    text = ARABIC_DIACRITICS.sub('', text)

    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)

    return text


def normalize_records(records: list) -> list:
    """
    Applies normalize_arabic() to the "law_name" and "text" fields
    of every record. Keeps the original "categories" field untouched
    so the raw category names are still readable for humans reviewing
    the mapping in metadata_tagger.py.
    """
    normalized = []
    for record in records:
        normalized.append({
            "law_name": normalize_arabic(record["law_name"]),
            "law_name_original": record["law_name"],
            "categories": record["categories"],
            "text": normalize_arabic(record["text"]),
            "tokens": record["tokens"],
        })
    return normalized
