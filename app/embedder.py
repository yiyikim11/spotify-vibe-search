"""
Low-memory text embedder for Render's 512MB free tier.

Uses FastEmbed (ONNX Runtime) instead of PyTorch/Sentence-Transformers.
Same MiniLM model family → 384-D vectors; we L2-normalize so FAISS IP == cosine.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


class FastEmbedEncoder:
    """ONNX MiniLM encoder — much lighter RAM than torch + transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding

        # threads=1 reduces peak RSS on tiny instances
        self.model = TextEmbedding(model_name=model_name, threads=1)
        self.model_name = model_name
        self.dim = 384

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self.model.embed(texts, batch_size=batch_size))
        return _l2_normalize(np.stack(vectors, axis=0))


@lru_cache(maxsize=1)
def get_encoder(model_name: str = DEFAULT_MODEL) -> FastEmbedEncoder:
    """Singleton so we only pay ONNX load cost once per process."""
    # Prefer HF cache dir on ephemeral disks
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return FastEmbedEncoder(model_name=model_name)
