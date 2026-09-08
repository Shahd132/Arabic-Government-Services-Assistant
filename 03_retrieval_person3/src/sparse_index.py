from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from schemas import Chunk  # noqa: E402

logger = logging.getLogger(__name__)

_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_ALEF_VARIANTS = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
    }
)


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_arabic(text.lower())
    tokens = text.split(" ")
    out = []
    for tok in tokens:
        if not tok:
            continue
        # light stemming: drop leading definite article "ال"
        if tok.startswith("ال") and len(tok) > 3:
            tok = tok[2:]
        out.append(tok)
    return out


class SparseIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]) -> "SparseIndex":
        self.chunks = list(chunks)
        self._tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("SparseIndex: built BM25 over %d chunks", len(self.chunks))
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k = min(top_k, len(self.chunks))
        top_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        logger.info("SparseIndex saved to %s (rebuild BM25 on load)", dir_path)

    @classmethod
    def load(cls, dir_path: str) -> "SparseIndex":
        chunks = []
        with open(os.path.join(dir_path, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.from_dict(json.loads(line)))
        return cls().build(chunks)


class SparseIndexRegistry:
    ALL = "__all__"

    def __init__(self):
        self.indices = {}

    def build_from_chunks(self, chunks: List[Chunk]) -> "SparseIndexRegistry":
        self.indices[self.ALL] = SparseIndex().build(chunks)

        by_dept = {}
        for c in chunks:
            dept = c.department or "unknown"
            by_dept.setdefault(dept, []).append(c)

        for dept, dept_chunks in by_dept.items():
            self.indices[dept] = SparseIndex().build(dept_chunks)

        return self

    def get(self, department: Optional[str] = None) -> SparseIndex:
        key = department if department else self.ALL
        if key not in self.indices:
            raise KeyError(
                f"No sparse index for department='{department}'. "
                f"Available: {list(self.indices.keys())}"
            )
        return self.indices[key]

    def save(self, root_dir: str) -> None:
        for key, idx in self.indices.items():
            idx.save(os.path.join(root_dir, key.replace(" ", "_")))

    @classmethod
    def load(cls, root_dir: str) -> "SparseIndexRegistry":
        reg = cls()
        for name in os.listdir(root_dir):
            sub = os.path.join(root_dir, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "chunks.jsonl")):
                reg.indices[name] = SparseIndex.load(sub)
        return reg
