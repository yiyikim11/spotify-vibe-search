"""
Online search engine (loaded by FastAPI at startup).

Pipeline for one user query:
  1) embed the vibe text with the SAME MiniLM model used offline
  2) FAISS Flat IP → nearest lyric chunks  (cosine, because vectors are L2-normalized)
  3) max-pool chunk hits by song_id
  4) enrich top songs with cover art + YouTube listen link (display only; no re-chunk)
  5) return top-k songs with score + matching lyric excerpt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.media import MediaEnricher

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@dataclass
class SearchHit:
    """One ranked song result shown in the UI."""

    song_id: int
    artist: str
    song: str
    link: str
    score: float  # cosine similarity in [-1, 1]; higher = closer vibe
    excerpt: str  # best-matching lyric chunk for this song
    image_url: str | None = None  # cover art from iTunes (optional)
    youtube_url: str | None = None  # Listen on YouTube target


class VibeSearchEngine:
    """Loads offline artifacts once, then answers vibe queries."""

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

        # config.json stores model name, dim, metric, index type (for UI + evaluation)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        # MUST be the same model as offline indexing so query/lyrics share one latent space
        self.model = SentenceTransformer(self.config["model_name"])

        # Pre-built Flat IP index over all lyric-chunk vectors
        self.index = faiss.read_index(str(index_path))

        # Metadata aligned with FAISS row ids (chunk_id == vector row)
        self.chunks = pd.read_parquet(chunks_path)

        # Cover art + YouTube links (cached; does not touch FAISS / embeddings)
        self.media = MediaEnricher()

        expected = int(self.config["embedding_dim"])
        if self.index.d != expected:
            raise ValueError(f"Index dim {self.index.d} != config dim {expected}")

    @property
    def embedding_dim(self) -> int:
        return int(self.config["embedding_dim"])

    def embed_query(self, query: str) -> np.ndarray:
        """
        Map the user's vibe description into the same 384-D space as lyric chunks.
        normalize_embeddings=True so FAISS inner product == cosine similarity.
        """
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32)

    def search(self, query: str, top_k: int = 10, candidate_chunks: int = 80) -> list[SearchHit]:
        """
        Retrieve top_k songs for a natural-language vibe query.

        candidate_chunks:
          how many chunk neighbors to pull from FAISS before pooling to songs.
          Needs to be > top_k because several chunks may belong to the same song.
        """
        query = (query or "").strip()
        if not query:
            return []

        # 1) Query vector in shared embedding space
        q = self.embed_query(query)

        # 2) Exact nearest-neighbor search over all chunk vectors
        n_retrieve = min(max(candidate_chunks, top_k), self.index.ntotal)
        scores, indices = self.index.search(q, n_retrieve)
        # scores[0][i] = cosine similarity of query vs chunk indices[0][i]

        # 3) Max-pool: one score per song = best matching chunk for that song
        best: dict[int, SearchHit] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS padding when asking for more neighbors than exist
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
            # Keep only the strongest lyric match for each track
            if song_id not in best or hit.score > best[song_id].score:
                best[song_id] = hit

        # 4) Rank songs by pooled score
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]

        # 5) Attach cover + YouTube listen link (metadata only; no re-index)
        for hit in ranked:
            image_url, youtube_url = self.media.enrich(hit.artist, hit.song)
            hit.image_url = image_url
            hit.youtube_url = youtube_url

        return ranked
