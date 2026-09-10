"""
Processes a single full-law raw text file (e.g. from the HazemShoaib/Egypt_Laws_raw
HuggingFace dataset) through the SAME clean -> normalize -> chunk -> metadata
pipeline used in build_kb.py, and merges the resulting chunks into an existing
department chunks.json.

Uses the EXACT same metadata shape as the rest of the corpus - no extra
fields (department, law_name, categories, chunk_index, source only).

Expected input format (plain .txt):
    قانون رقم: <number>
    العنوان: <title>
    التاريخ: ...
    الحالة: ...
    جهة الإصدار: ...

    ================================================================================

    <full law body text - chapters, articles, etc.>

Usage:
    python process_full_law.py \
        --input ../data/additional_laws/civil_affairs/law_143_1994_civil_status.txt \
        --department civil_affairs \
        --categories "قانون الأحوال المدنية" \
        --source "Egypt Laws Dataset (HazemShoaib, HuggingFace)" \
        --merge-into ../output/chunks/civil_affairs/chunks.json
"""

import json
import os
import re
import argparse

from cleaner import clean_text
from normalizer import normalize_arabic
from chunker import chunk_text

SEPARATOR = "=" * 80


def parse_law_file(path: str):
    """Splits the raw file into (title, body). The header block (before the
    '====' separator) is only used to pull out the law's title - it isn't
    included in the chunked body text."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if SEPARATOR in raw:
        header, body = raw.split(SEPARATOR, 1)
    else:
        header, body = "", raw

    title_match = re.search(r"العنوان:\s*(.+)", header)
    title = title_match.group(1).strip() if title_match else os.path.splitext(os.path.basename(path))[0]

    return title, body


def process_law(title: str, body: str, department: str, categories: list, source: str) -> list:
    title_clean = clean_text(title)
    body_clean = clean_text(body)

    title_norm = normalize_arabic(title_clean)
    body_norm = normalize_arabic(body_clean)

    text_chunks = chunk_text(body_norm)

    result = []
    for i, chunk in enumerate(text_chunks):
        result.append({
            "id": f"{title_clean}_{i}",
            "text": chunk,
            "metadata": {
                "department": department,
                "law_name": title_clean,
                "categories": categories,
                "chunk_index": i,
                "source": source,
            },
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw law .txt file")
    parser.add_argument("--department", required=True, choices=["civil_affairs", "tax", "traffic"])
    parser.add_argument("--categories", nargs="+", required=True,
                         help="one or more category labels, e.g. --categories \"قانون الأحوال المدنية\"")
    parser.add_argument("--source", required=True, help="human-readable source label")
    parser.add_argument("--merge-into", required=True,
                         help="existing department chunks.json to merge new chunks into")
    args = parser.parse_args()

    title, body = parse_law_file(args.input)
    print(f"Parsed law: {title}")
    print(f"Body length: {len(body.split())} words")

    new_chunks = process_law(title, body, args.department, args.categories, args.source)
    print(f"Produced {len(new_chunks)} new chunks.")

    existing_chunks = []
    if os.path.exists(args.merge_into):
        with open(args.merge_into, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    print(f"Existing chunks in {args.merge_into}: {len(existing_chunks)}")

    existing_ids = {c["id"] for c in existing_chunks}
    deduped_new = [c for c in new_chunks if c["id"] not in existing_ids]
    skipped = len(new_chunks) - len(deduped_new)
    if skipped:
        print(f"Skipped {skipped} chunks with ids that already existed.")

    merged = existing_chunks + deduped_new

    os.makedirs(os.path.dirname(args.merge_into), exist_ok=True)
    with open(args.merge_into, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged. {args.merge_into} now has {len(merged)} total chunks "
          f"({len(existing_chunks)} original + {len(deduped_new)} new).")


if __name__ == "__main__":
    main()
