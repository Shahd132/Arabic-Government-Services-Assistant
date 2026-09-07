"""
Exploration helper: lists every unique category found in the cleaned
dataset, with a count of how many laws use it. Use this to review and
extend CATEGORY_TO_DEPARTMENT in metadata_tagger.py.

Run this AFTER build_kb.py has produced the cleaned data file.
Result is written to a text file (not printed) because Arabic text
can render incorrectly in some terminals.
"""

import json
from collections import Counter

CLEANED_PATH = "../data/egyptian_legal_corpus/cleaned/egypt_legal_corpus_cleaned.json"
OUTPUT_PATH = "category_report.txt"


def main():
    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_counter = Counter()
    for record in data:
        for category in record["categories"]:
            category_counter[category] += 1

    sorted_categories = category_counter.most_common()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Total unique categories found: {len(sorted_categories)}\n\n")
        f.write("Category name  ->  number of laws\n")
        f.write("-" * 40 + "\n")
        for category, count in sorted_categories:
            f.write(f"{category}  ->  {count}\n")

    print(f"Done. Open '{OUTPUT_PATH}' in VS Code to see the full list.")


if __name__ == "__main__":
    main()
