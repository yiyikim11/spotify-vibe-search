"""
Query / document text embedders.

Backends:
  - remote  → Hugging Face Inference API (no model in Render RAM; needs HF_TOKEN)
  - fastembed → local ONNX MiniLM (for building the index on your laptop)

Lyrics are embedded OFFLINE into FAISS. Render only embeds the short user query.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Official HF Inference feature-extraction URL (sentence embeddings / token feats)
_HF_FEATURE_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "{model}/pipeline/feature-extraction"
)
# Fallback older host
_HF_FEATURE_URL_LEGACY = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _pool_hf_output(data: object) -> np.ndarray:
    """
    HF feature-extraction may return:
      - [d]                  sentence vector
      - [tokens, d]          token vectors → mean-pool
      - [[d], ...]           batch of sentence vectors
      - [[tokens, d], ...]   batch of token matrices
    Always returns shape (1, dim) or (batch, dim).
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        # Single input usually returns (seq_len, hidden) — mean-pool to one vector
        if arr.shape[0] == 1:
            return arr
        return arr.mean(axis=0, keepdims=True)
    if arr.ndim == 3:
        return arr.mean(axis=1)
    raise ValueError(f"Unexpected HF embedding shape: {arr.shape}")


class Embedder(Protocol):
    model_name: str
    dim: int
    backend: str

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...


class RemoteHFEncoder:
    """
    Call Hugging Face Inference API for MiniLM embeddings.
    Render holds FAISS only — query vectors come from the API (tiny RAM).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, token: str | None = None):
        import httpx

        self.model_name = model_name
        self.dim = 384
        self.backend = "remote-hf"
        self._token = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()
        if not self._token:
            raise RuntimeError(
                "EMBED_BACKEND=remote requires HF_TOKEN. "
                "Create a free token at https://huggingface.co/settings/tokens "
                "and set it in the Render Environment tab."
            )
        self._http = httpx.Client(timeout=60.0)
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _post(self, url: str, payload: dict) -> object:
        r = self._http.post(url, headers=self._headers, json=payload)
        if r.status_code == 503:
            # Model loading on HF side — retry once with wait
            r = self._http.post(
                url,
                headers=self._headers,
                json={**payload, "options": {"wait_for_model": True}},
            )
        r.raise_for_status()
        return r.json()

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        urls = [
            _HF_FEATURE_URL.format(model=self.model_name),
            _HF_FEATURE_URL_LEGACY.format(model=self.model_name),
        ]
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Single-string input is most reliable for sentence models
            vecs: list[np.ndarray] = []
            last_err: Exception | None = None
            for text in batch:
                payload = {"inputs": text, "options": {"wait_for_model": True}}
                data = None
                for url in urls:
                    try:
                        data = self._post(url, payload)
                        break
                    except Exception as exc:  # noqa: BLE001 — try fallback host
                        last_err = exc
                        continue
                if data is None:
                    raise RuntimeError(f"HF embedding failed: {last_err}") from last_err
                pooled = _pool_hf_output(data)
                if pooled.shape[0] != 1:
                    pooled = pooled.mean(axis=0, keepdims=True)
                vecs.append(pooled.reshape(-1))
            out.append(np.stack(vecs, axis=0))

        return _l2_normalize(np.vstack(out))


class FastEmbedEncoder:
    """Local ONNX MiniLM — use on your laptop to BUILD the FAISS index."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, threads=1)
        self.model_name = model_name
        self.dim = 384
        self.backend = "fastembed-onnx"

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self.model.embed(texts, batch_size=batch_size))
        return _l2_normalize(np.stack(vectors, axis=0))


def resolve_backend() -> str:
    """
    EMBED_BACKEND=remote|fastembed|auto
    auto → remote when HF_TOKEN is set (Render), else fastembed (local).
    """
    raw = (os.environ.get("EMBED_BACKEND") or "auto").strip().lower()
    if raw in {"remote", "hf", "huggingface"}:
        return "remote"
    if raw in {"fastembed", "onnx", "local"}:
        return "fastembed"
    # auto
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        return "remote"
    return "fastembed"


@lru_cache(maxsize=1)
def get_encoder(model_name: str = DEFAULT_MODEL) -> Embedder:
    backend = resolve_backend()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if backend == "remote":
        return RemoteHFEncoder(model_name=model_name)
    return FastEmbedEncoder(model_name=model_name)
