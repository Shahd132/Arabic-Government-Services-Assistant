from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Literal, Optional
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from schemas import Chunk, RetrievedChunk, RetrievalResult  # noqa: E402

from embedder import Embedder  # noqa: E402
from dense_index import DenseIndexRegistry  # noqa: E402
from sparse_index import SparseIndexRegistry  # noqa: E402

logger = logging.getLogger(__name__)

FusionMethod = Literal["rrf", "weighted"]


class HybridSearch:
    def __init__(
        self,
        embedder: Embedder,
        dense_registry: DenseIndexRegistry,
        sparse_registry: SparseIndexRegistry,
        fusion: FusionMethod = "rrf",
        alpha: float = 0.5,          # weight on dense score, used only if fusion="weighted"
        rrf_k: int = 60,
        candidate_pool: int = 20,     # how many candidates to pull from EACH retriever before fusing
    ):
        self.embedder = embedder
        self.dense_registry = dense_registry
        self.sparse_registry = sparse_registry
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        department: Optional[str] = None,
    ) -> RetrievalResult:
        dense_index = self.dense_registry.get(department)
        sparse_index = self.sparse_registry.get(department)

        pool = max(self.candidate_pool, top_k)

        query_vec = self.embedder.embed_query(query)
        dense_hits = dense_index.search(query_vec, top_k=pool)          # [(Chunk, score), ...]
        sparse_hits = sparse_index.search(query, top_k=pool)            # [(Chunk, score), ...]

        if self.fusion == "rrf":
            fused = self._fuse_rrf(dense_hits, sparse_hits)
        else:
            fused = self._fuse_weighted(dense_hits, sparse_hits)

        fused_sorted = sorted(fused.values(), key=lambda rc: rc.score, reverse=True)[:top_k]
        return RetrievalResult(query=query, results=fused_sorted)

    def _fuse_rrf(self, dense_hits, sparse_hits) -> Dict[str, RetrievedChunk]:
        fused: Dict[str, RetrievedChunk] = {}
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            rc = fused.setdefault(chunk.id, RetrievedChunk(chunk=chunk, score=0.0))
            rc.dense_score = score
            rc.dense_rank = rank
            rc.score += 1.0 / (self.rrf_k + rank)

        for rank, (chunk, score) in enumerate(sparse_hits, start=1):
            rc = fused.setdefault(chunk.id, RetrievedChunk(chunk=chunk, score=0.0))
            rc.sparse_score = score
            rc.sparse_rank = rank
            rc.score += 1.0 / (self.rrf_k + rank)

        return fused

    def _fuse_weighted(self, dense_hits, sparse_hits) -> Dict[str, RetrievedChunk]:
        fused: Dict[str, RetrievedChunk] = {}

        dense_scores = [s for _, s in dense_hits]
        sparse_scores = [s for _, s in sparse_hits]
        d_min, d_max = (min(dense_scores), max(dense_scores)) if dense_scores else (0, 1)
        s_min, s_max = (min(sparse_scores), max(sparse_scores)) if sparse_scores else (0, 1)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.0

        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            rc = fused.setdefault(chunk.id, RetrievedChunk(chunk=chunk, score=0.0))
            rc.dense_score = score
            rc.dense_rank = rank
            rc.score += self.alpha * norm(score, d_min, d_max)

        for rank, (chunk, score) in enumerate(sparse_hits, start=1):
            rc = fused.setdefault(chunk.id, RetrievedChunk(chunk=chunk, score=0.0))
            rc.sparse_score = score
            rc.sparse_rank = rank
            rc.score += (1 - self.alpha) * norm(score, s_min, s_max)

        return fused

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        embedder: Optional[Embedder] = None,
        **kwargs,
    ) -> "HybridSearch":
        """Convenience constructor: builds dense + sparse registries from
        a flat list of chunks in one call."""
        embedder = embedder or Embedder()
        dense_registry = DenseIndexRegistry().build_from_chunks(chunks, embedder)
        sparse_registry = SparseIndexRegistry().build_from_chunks(chunks)
        return cls(embedder, dense_registry, sparse_registry, **kwargs)
