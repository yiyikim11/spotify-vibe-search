"""
Online search engine for FastAPI.

Pipeline for one user query:
  1) embed vibe text with ONNX MiniLM (fastembed) — low RAM for Render free tier
  2) FAISS Flat IP → nearest lyric chunks (cosine after L2-normalize)
  3) max-pool chunk hits by song_id
  4) enrich top songs with cover / YouTube (display only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from app.embedder import DEFAULT_MODEL, get_encoder
from app.media import MediaEnricher

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@dataclass
class SearchHit:
    song_id: int
    artist: str
    song: str
    link: str
    score: float
    excerpt: str
    image_url: str | None = None
    youtube_url: str | None = None


class VibeSearchEngine:
    """Loads FAISS + chunk metadata; encoder is a shared ONNX singleton."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        self.artifacts_dir = Path(artifacts_dir)
        config_path = self.artifacts_dir / "config.json"
        index_path = self.artifacts_dir / "faiss.index"
        chunks_path = self.artifacts_dir / "chunks.parquet"

        if not index_path.exists() or not chunks_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts in {self.artifacts_dir}. "
                "Run: python scripts/prepare_and_index.py"
            )

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = self.config.get("model_name", DEFAULT_MODEL)

        # ONNX encoder (not PyTorch) — required to fit ~512MB Render free RAM
        self.encoder = get_encoder(model_name)

        self.index = faiss.read_index(str(index_path))
        # Only columns needed at search time (smaller RSS than full frame use)
        self.chunks = pd.read_parquet(
            chunks_path,
            columns=["song_id", "artist", "song", "link", "chunk_text"],
        )
        self.media = MediaEnricher()

        expected = int(self.config["embedding_dim"])
        if self.index.d != expected:
            raise ValueError(f"Index dim {self.index.d} != config dim {expected}")

    @property
    def embedding_dim(self) -> int:
        return int(self.config["embedding_dim"])

    def embed_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query], batch_size=1)

    def search(self, query: str, top_k: int = 10, candidate_chunks: int = 80) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []

        q = self.embed_query(query)
        n_retrieve = min(max(candidate_chunks, top_k), self.index.ntotal)
        scores, indices = self.index.search(q, n_retrieve)

        best: dict[int, SearchHit] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.chunks.iloc[int(idx)]
            song_id = int(row["song_id"])
            hit = SearchHit(
                song_id=song_id,
                artist=str(row["artist"]),
                song=str(row["song"]),
                link=str(row.get("link", "")),
                score=float(score),
                excerpt=str(row["chunk_text"]),
            )
            if song_id not in best or hit.score > best[song_id].score:
                best[song_id] = hit

        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]

        for hit in ranked:
            image_url, youtube_url = self.media.enrich(hit.artist, hit.song)
            hit.image_url = image_url
            hit.youtube_url = youtube_url

        return ranked
