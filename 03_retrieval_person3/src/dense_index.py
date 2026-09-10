from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from schemas import Chunk  # noqa: E402


logger = logging.getLogger(__name__)
class DenseIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> "DenseIndex":
        assert len(chunks) == embeddings.shape[0], "chunks/embeddings length mismatch"
        assert embeddings.shape[1] == self.dimension, "embedding dimension mismatch"
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        logger.info("DenseIndex: added %d vectors (total=%d)", len(chunks), self.index.ntotal)
        return self

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        self.build(chunks, embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype("float32")
        top_k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "count": len(self.chunks)}, f)
        logger.info("DenseIndex saved to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str) -> "DenseIndex":
        with open(os.path.join(dir_path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(dimension=meta["dimension"])
        obj.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        obj.chunks = []
        with open(os.path.join(dir_path, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj.chunks.append(Chunk.from_dict(json.loads(line)))
        logger.info("DenseIndex loaded from %s (%d vectors)", dir_path, obj.index.ntotal)
        return obj


class DenseIndexRegistry:
    ALL = "__all__"

    def __init__(self):
        self.indices: Dict[str, DenseIndex] = {}

    def build_from_chunks(self, chunks: List[Chunk], embedder) -> "DenseIndexRegistry":
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_passages(texts)
        dim = embeddings.shape[1]

        all_index = DenseIndex(dim).build(chunks, embeddings)
        self.indices[self.ALL] = all_index

        # per-department indices
        by_dept: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            dept = c.department or "unknown"
            by_dept.setdefault(dept, []).append(i)

        for dept, idxs in by_dept.items():
            dept_index = DenseIndex(dim)
            dept_index.build([chunks[i] for i in idxs], embeddings[idxs])
            self.indices[dept] = dept_index

        return self

    def get(self, department: Optional[str] = None) -> DenseIndex:
        key = department if department else self.ALL
        if key not in self.indices:
            raise KeyError(
                f"No dense index for department='{department}'. "
                f"Available: {list(self.indices.keys())}"
            )
        return self.indices[key]

    def save(self, root_dir: str) -> None:
        for key, idx in self.indices.items():
            idx.save(os.path.join(root_dir, key.replace(" ", "_")))

    @classmethod
    def load(cls, root_dir: str) -> "DenseIndexRegistry":
        reg = cls()
        for name in os.listdir(root_dir):
            sub = os.path.join(root_dir, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "meta.json")):
                reg.indices[name] = DenseIndex.load(sub)
        return reg
