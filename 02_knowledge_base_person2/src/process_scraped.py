"""
Processes scraped documents (e.g. traffic_scraped_clean.json) through the
SAME clean -> normalize -> chunk -> metadata pipeline used in build_kb.py
for the main Egyptian Legal Corpus, so scraped chunks are indistinguishable
in shape from corpus-derived chunks.

Input doc shape (from clean_scraped_for_person2.py):
    {
      "title": "...",
      "department": "traffic",
      "categories": [...],
      "source_url": "...",
      "source": "traffic.moi.gov.eg (scraped)",
      "text": "..."
    }

Output chunk shape (identical to build_kb.py's output):
    {
      "id": "<title>_<chunk_index>",
      "text": "...",
      "metadata": {
          "department": ...,
          "law_name": ...,          # holds the doc title, for consistency
          "categories": [...],
          "chunk_index": ...,
          "source": ...,            # preserved from the input doc, NOT overwritten
          "source_url": ...         # extra field, kept for traceability
      }
    }

Usage:
    python process_scraped.py --input ../data/scraped/traffic_scraped_clean.json \
                               --merge-into ../output/chunks/traffic/chunks.json
"""

import json
import os
import argparse

from cleaner import clean_text
from normalizer import normalize_arabic
from chunker import chunk_text


def process_documents(docs: list) -> list:
    all_chunks = []

    for doc in docs:
        title_clean = clean_text(doc.get("title", ""))
        text_clean = clean_text(doc.get("text", ""))

        if not text_clean:
            continue

        title_norm = normalize_arabic(title_clean)
        text_norm = normalize_arabic(text_clean)

        text_chunks = chunk_text(text_norm)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{title_clean}_{i}",
                "text": chunk,
                "metadata": {
                    "department": doc.get("department"),
                    "law_name": title_clean,
                    "categories": doc.get("categories", []),
                    "chunk_index": i,
                    "source": doc.get("source", "scraped"),
                    "source_url": doc.get("source_url", ""),
                },
            })

    return all_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="cleaned scraped-docs JSON file")
    parser.add_argument("--merge-into", required=True,
                         help="existing department chunks.json to merge new chunks into")
    parser.add_argument("--out-report", default=None,
                         help="optional path to write a short summary report")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} scraped documents from {args.input}")

    new_chunks = process_documents(docs)
    print(f"Produced {len(new_chunks)} new chunks.")

    existing_chunks = []
    if os.path.exists(args.merge_into):
        with open(args.merge_into, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    print(f"Existing chunks in {args.merge_into}: {len(existing_chunks)}")

    # Avoid id collisions with existing chunks (shouldn't happen with distinct
    # titles, but guard anyway)
    existing_ids = {c["id"] for c in existing_chunks}
    deduped_new = [c for c in new_chunks if c["id"] not in existing_ids]
    skipped = len(new_chunks) - len(deduped_new)
    if skipped:
        print(f"Skipped {skipped} chunks with ids that already existed.")

    merged = existing_chunks + deduped_new

    with open(args.merge_into, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged. {args.merge_into} now has {len(merged)} total chunks "
          f"({len(existing_chunks)} original + {len(deduped_new)} new).")


if __name__ == "__main__":
    main()
