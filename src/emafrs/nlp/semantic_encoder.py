"""Contextual Semantic Encoder (CSE) -- PDF Section 6.4.4: dense embedding of a
communication's topical/contextual content, used as the c_t component of the affective
feature vector (PDF Eq. 1)."""
from __future__ import annotations

import numpy as np


class ContextualSemanticEncoder:
    def __init__(self, hf_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.hf_id = hf_id
        self._model = SentenceTransformer(hf_id)
        # `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension` in
        # newer sentence-transformers; support both installed versions.
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns an (n_texts, dim) array of L2-normalized embeddings."""
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
