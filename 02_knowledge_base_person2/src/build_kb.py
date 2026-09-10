"""
Main pipeline: runs the full Knowledge Base build process end to end.

Steps:
1. Load the raw dataset (download it first if it's missing)
2. Clean the text (cleaner.py)
3. Normalize the Arabic text (normalizer.py)
4. Classify each law into a department (metadata_tagger.py)
5. Split each law's text into chunks (chunker.py)
6. Save the final chunks into output/chunks/<department>/chunks.json

Run this file to build the whole knowledge base in one go:
    python build_kb.py
"""

import json
import os

from download_data import download_dataset, RAW_OUTPUT_PATH
from cleaner import clean_records
from normalizer import normalize_records
from chunker import chunk_text
from metadata_tagger import get_department, build_metadata

CLEANED_OUTPUT_PATH = "../data/egyptian_legal_corpus/cleaned/egypt_legal_corpus_cleaned.json"
CHUNKS_OUTPUT_DIR = "../output/chunks"

DEPARTMENTS = ["civil_affairs", "tax", "traffic"]


def load_raw_data():
    if not os.path.exists(RAW_OUTPUT_PATH):
        print("Raw data not found - downloading it first.")
        download_dataset()

    with open(RAW_OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 1. Load raw data
    raw_data = load_raw_data()
    print(f"Loaded {len(raw_data)} raw records.")

    # 2. Clean
    cleaned_data = clean_records(raw_data)
    print(f"{len(cleaned_data)} records remain after cleaning.")

    # Save the cleaned data too, in case another script needs it directly
    os.makedirs(os.path.dirname(CLEANED_OUTPUT_PATH), exist_ok=True)
    with open(CLEANED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    # 3. Normalize
    normalized_data = normalize_records(cleaned_data)

    # 4 & 5. Classify department, then chunk each matching law
    chunks_by_department = {dept: [] for dept in DEPARTMENTS}
    department_law_counts = {dept: 0 for dept in DEPARTMENTS}

    for record in normalized_data:
        department = get_department(record["law_name"], record["categories"])

        if department is None:
            continue  # law doesn't belong to our 3 target departments

        department_law_counts[department] += 1

        text_chunks = chunk_text(record["text"])
        for i, chunk in enumerate(text_chunks):
            chunks_by_department[department].append({
                "id": f"{record['law_name_original']}_{i}",
                "text": chunk,
                "metadata": build_metadata(
                    record["law_name_original"],
                    record["categories"],
                    department,
                    i,
                ),
            })

    print(f"\nLaws matched per department: {department_law_counts}")

    # 6. Save chunks - one file per department, matching the agreed folder structure
    for department in DEPARTMENTS:
        dept_dir = os.path.join(CHUNKS_OUTPUT_DIR, department)
        os.makedirs(dept_dir, exist_ok=True)

        output_path = os.path.join(dept_dir, "chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_by_department[department], f, ensure_ascii=False, indent=2)

        print(f"Saved {len(chunks_by_department[department])} chunks to: {output_path}")


if __name__ == "__main__":
    main()
