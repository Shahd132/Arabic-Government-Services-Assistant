"""
Text cleaning utilities.
Removes stray characters, extra whitespace, and empty records.
This module only exports functions - it doesn't run a pipeline by itself.
"""

import re


def clean_text(text: str) -> str:
    """
    Cleans a single text string:
    - Removes characters that aren't Arabic, English, digits, or
      common punctuation
    - Collapses multiple whitespace characters into a single space
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""

    # Keep only Arabic, English, digits, and common punctuation
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F0-9a-zA-Z\s.,:;!؟?()\-/]', ' ', text)

    # Collapse whitespace (tabs, newlines, multiple spaces) into one space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_records(records: list) -> list:
    """
    Applies clean_text() to the "law_name" and "text" fields of every
    record in the list. Drops records that become empty after cleaning.
    """
    cleaned = []
    for record in records:
        cleaned_record = {
            "law_name": clean_text(record.get("law_name", "")),
            "categories": record.get("categories", []),
            "text": clean_text(record.get("text", "")),
            "tokens": record.get("tokens", 0),
        }
        if cleaned_record["text"]:
            cleaned.append(cleaned_record)
    return cleaned
