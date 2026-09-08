from __future__ import annotations
import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 16,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # lazy-loaded so importing this module is cheap
        self._device = device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype("float32")


class MockEmbedder(Embedder):
    def __init__(self, dim: int = 256, ngram: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
        self._ngram = ngram


    @property
    def dimension(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def _encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype="float32")
        for i, t in enumerate(texts):
            t = t.strip()
            for j in range(len(t) - self._ngram + 1):
                gram = t[j : j + self._ngram]
                h = hash(gram) % self._dim
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
