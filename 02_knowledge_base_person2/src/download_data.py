"""
Downloads the Egyptian Legal Corpus dataset from HuggingFace and saves
it as a raw JSON file. This is the entry point of the pipeline - run
this first, before cleaner.py / chunker.py / build_kb.py.
"""

from datasets import load_dataset
import json
import os

RAW_OUTPUT_PATH = "../data/egyptian_legal_corpus/raw/egypt_legal_corpus.json"


def download_dataset():
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("dataflare/egypt-legal-corpus")
    data_split = dataset["train"]

    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([row for row in data_split], f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data_split)} records to: {RAW_OUTPUT_PATH}")
    return RAW_OUTPUT_PATH


if __name__ == "__main__":
    download_dataset()
