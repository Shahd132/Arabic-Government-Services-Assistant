from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from schemas import Chunk  # noqa: E402
from embedder import Embedder  # noqa: E402
from dense_index import DenseIndexRegistry  # noqa: E402
from sparse_index import SparseIndexRegistry  # noqa: E402

logger = logging.getLogger(__name__)


def load_chunks(chunks_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []

    for root, _dirs, files in os.walk(chunks_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            path = os.path.join(root, fname)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            for d in data:
                chunks.append(Chunk.from_dict(d))

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_dir)
    return chunks

def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    seen_texts = set()
    unique = []
    dropped = 0
    for c in chunks:
        key = c.text.strip()
        if key in seen_texts:
            dropped += 1
            continue
        seen_texts.add(key)
        unique.append(c)
    logger.info("Deduplication: kept %d, dropped %d duplicate chunks", len(unique), dropped)
    return unique

def main():
    parser = argparse.ArgumentParser(
        description="Build hybrid retrieval indices"
    )

    parser.add_argument(
        "--chunks_dir",
        default="02_knowledge_base_person2/output/chunks"
    )

    parser.add_argument(
        "--out_dir",
        default="03_retrieval_person3/vector_store"
    )

    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    chunks = load_chunks(args.chunks_dir)
    chunks = deduplicate_chunks(chunks) 
    if not chunks:
        raise SystemExit(
            f"No chunks found under {args.chunks_dir}"
        )

    # Real embedding model
    embedder = Embedder(model_name=args.model)

    logger.info("Building dense (FAISS) index...")
    dense_registry = DenseIndexRegistry().build_from_chunks(
        chunks,
        embedder
    )

    logger.info("Building sparse (BM25) index...")
    sparse_registry = SparseIndexRegistry().build_from_chunks(
        chunks
    )

    # Save indexes
    dense_registry.save(
        os.path.join(args.out_dir, "dense")
    )

    sparse_registry.save(
        os.path.join(args.out_dir, "sparse")
    )

    # Save build information
    os.makedirs(args.out_dir, exist_ok=True)

    with open(
        os.path.join(args.out_dir, "build_info.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "num_chunks": len(chunks),
                "departments": sorted(
                    {
                        c.department
                        for c in chunks
                        if c.department
                    }
                ),
                "embedder_model": args.model,
                "dense_index": "FAISS IndexFlatIP",
                "sparse_index": "BM25",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Done. Indices saved under %s",
        args.out_dir
    )


if __name__ == "__main__":
    main()