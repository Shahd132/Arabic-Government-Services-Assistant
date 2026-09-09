from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import List, Optional
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from schemas import Chunk  # noqa: E402

from embedder import Embedder  # noqa: E402
from dense_index import DenseIndexRegistry  # noqa: E402
from sparse_index import SparseIndexRegistry  # noqa: E402
from hybrid_search import HybridSearch  # noqa: E402

VECTOR_STORE_DIR = os.environ.get(
    "RETRIEVAL_VECTOR_STORE",
    os.path.join(os.path.dirname(__file__), "..", "vector_store"),
)
EMBEDDING_MODEL = os.environ.get("RETRIEVAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


@lru_cache(maxsize=1)
def _get_hybrid_search() -> HybridSearch:
    dense_registry = DenseIndexRegistry.load(os.path.join(VECTOR_STORE_DIR, "dense"))
    sparse_registry = SparseIndexRegistry.load(os.path.join(VECTOR_STORE_DIR, "sparse"))
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    return HybridSearch(embedder, dense_registry, sparse_registry, fusion="rrf")


def retrieve(
    query: str,
    top_k: int = 2,
    department: Optional[str] = None,
) -> List[RetrievedChunk]:
    hybrid = _get_hybrid_search()
    result = hybrid.retrieve(query, top_k=top_k, department=department)
    return result.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--department", default=None)
    args = parser.parse_args()

    for rc in retrieve(args.query, top_k=args.top_k, department=args.department):
        print(f"[{rc.score:.4f}] {rc.chunk.law_name} :: {rc.chunk.text[:100]}...")
