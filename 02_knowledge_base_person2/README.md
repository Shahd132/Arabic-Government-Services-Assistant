# 02 - Knowledge Base (Person 2)

Prepares the Egyptian Legal Corpus for the RAG system: downloads it,
cleans it, normalizes the Arabic text, classifies each law into a
department (civil_affairs / tax / traffic), and splits it into chunks
with metadata.

## Setup

```
pip install datasets
```

## How to run

From inside the `src/` folder, run the full pipeline with one command:

```
cd src
python build_kb.py
```

This will:
1. Download the dataset (only the first time - it's cached after that
   in `data/egyptian_legal_corpus/raw/`)
2. Clean and normalize the text
3. Classify each law into a department
4. Split matching laws into chunks
5. Save the results into `output/chunks/<department>/chunks.json`

## Reviewing the department mapping

The department classification in `src/metadata_tagger.py` starts from
a small, manually curated list of categories. Before treating the
output as final, run:

```
python explore_categories.py
```

This creates `category_report.txt` listing every category in the
dataset and how many laws use it. Review it and add any missing
categories to `CATEGORY_TO_DEPARTMENT` in `metadata_tagger.py`, then
re-run `build_kb.py`.

## Output format handed to Person 3 (Embeddings + RAG)

Each chunk in `output/chunks/<department>/chunks.json` looks like:

```json
{
  "id": "law_name_0",
  "text": "chunk text here...",
  "metadata": {
    "department": "traffic",
    "law_name": "...",
    "categories": ["..."],
    "chunk_index": 0,
    "source": "Egyptian Legal Corpus"
  }
}
```

This should be confirmed against `shared/schemas.py` once the team
agrees on the final `Chunk` schema.

## Folder structure

```
02_knowledge_base_person2/
├── data/
│   └── egyptian_legal_corpus/
│       ├── raw/          <- downloaded dataset
│       └── cleaned/      <- after cleaning
├── src/
│   ├── download_data.py
│   ├── cleaner.py
│   ├── normalizer.py
│   ├── chunker.py
│   ├── metadata_tagger.py
│   ├── explore_categories.py
│   └── build_kb.py       <- run this to build everything
├── output/
│   └── chunks/
│       ├── civil_affairs/
│       ├── tax/
│       └── traffic/
└── notebooks/
```
