"""
Offline pipeline (run once before starting the web app):
  1) load lyrics CSV
  2) split lyrics into sentence / line chunks
  3) embed chunks with Sentence Transformers (384-D)
  4) L2-normalize + build FAISS Flat IP index  (IP == cosine after normalize)
  5) save index + metadata into artifacts/ for the web app to load

Why offline?
  Embedding tens/hundreds of thousands of lyric chunks is slow.
  Do it once, then each user search only embeds the short query.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "raw" / "spotify_millsongdata.csv"
DEFAULT_OUT = ROOT / "artifacts"

# Dense retrieval model: maps text -> 384-D semantic vector
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # fixed output size of MiniLM

# Split lyrics on sentence endings (. ! ?) OR newline breaks (common in lyric text)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_lyric(text: str) -> str:
    """Normalize line breaks / spaces so chunking is consistent."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")  # some CSV rows store literal \n
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_lyrics(text: str, min_chars: int = 25, max_chars: int = 400) -> list[str]:
    """
    Split one song's lyrics into short semantic chunks.

    Why chunk instead of embedding the whole lyric?
      - Transformers have length limits
      - One song vector would average many moods together
      - Vibe often lives in a few local lines

    Steps:
      1. clean text
      2. split into sentences / lines
      3. merge fragments shorter than min_chars (e.g. "Oh baby")
      4. cap each chunk at max_chars
      5. dedupe repeated chorus lines
    """
    text = clean_lyric(text)
    raw_parts = [p.strip(" -|\t") for p in SENTENCE_SPLIT.split(text) if p.strip()]

    chunks: list[str] = []
    buf = ""  # holds short pieces until they become long enough
    for part in raw_parts:
        # Too short alone → accumulate in buffer
        if len(part) < min_chars:
            buf = f"{buf} {part}".strip() if buf else part
            if len(buf) >= min_chars:
                chunks.append(buf[:max_chars])
                buf = ""
            continue

        # Flush any leftover short buffer into this part
        if buf:
            part = f"{buf} {part}".strip()
            buf = ""
        chunks.append(part[:max_chars])

    # Don't lose a trailing short buffer if it finally became long enough
    if buf and len(buf) >= min_chars:
        chunks.append(buf[:max_chars])

    # Choruses repeat; keep first occurrence only (order preserved)
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def build_chunk_table(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    """
    Expand songs → one row per chunk.
    Each row keeps song_id so search can map a chunk hit back to a track.
    """
    rows: list[dict] = []
    for song_id, row in tqdm(df.iterrows(), total=len(df), desc="Chunking lyrics"):
        for chunk_text in chunk_lyrics(row["text"], min_chars=min_chars):
            rows.append(
                {
                    "song_id": int(song_id),
                    "artist": row["artist"],
                    "song": row["song"],
                    "link": row.get("link", ""),
                    "chunk_text": chunk_text,
                }
            )
    chunks = pd.DataFrame(rows)
    # chunk_id aligns with FAISS vector row index (0 .. N-1)
    chunks["chunk_id"] = np.arange(len(chunks), dtype=np.int32)
    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    """
    Convert text chunks → float32 embedding matrix of shape (N, 384).

    batch_size: how many chunks per model forward pass (speed/memory only; not quality).
    normalize_embeddings=True: each vector has length 1, so later
      inner_product(a, b) == cosine_similarity(a, b)
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a Flat (brute-force / exact) FAISS index.

    IndexFlatIP = Inner Product search over all vectors.
    With L2-normalized embeddings, this is exact cosine nearest-neighbor search.
    Flat is simple and high-recall; good fit for ~tens of thousands of songs.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # row i in FAISS == chunk_id i in chunks.parquet
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Spotify vibe-search FAISS index")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to spotify_millsongdata.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Where to write artifacts/")
    p.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Optional cap for a fast first run (e.g. 5000). Omit to use all songs.",
    )
    p.add_argument(
        "--sample",
        choices=["head", "random"],
        default="random",
        help="How to pick songs when --max-songs is set (default: random).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed when --sample random")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks per embedding batch (larger = faster, more RAM)",
    )
    p.add_argument("--min-chunk-chars", type=int, default=25, help="Drop/merge chunks shorter than this")
    p.add_argument("--model", type=str, default=MODEL_NAME, help="Sentence Transformer model name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Load + optional subsample ---
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["text", "artist", "song"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    if args.max_songs is not None:
        n = min(args.max_songs, len(df))
        if args.sample == "random":
            # Random sample avoids alphabet bias (CSV starts with ABBA, etc.)
            df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
            print(f"Using random sample of {len(df)} songs (seed={args.seed})")
        else:
            df = df.head(n).copy()
            print(f"Using first {len(df)} songs (--max-songs head)")

    # --- 2) Save song-level metadata ---
    songs = df[["artist", "song", "link"]].copy()
    songs.insert(0, "song_id", songs.index.astype(np.int32))
    songs_path = args.out_dir / "songs.parquet"
    songs.to_parquet(songs_path, index=False)
    print(f"Saved {len(songs)} songs -> {songs_path}")

    # --- 3) Chunk every lyric ---
    chunks = build_chunk_table(df, min_chars=args.min_chunk_chars)
    if chunks.empty:
        print("No chunks produced. Check CSV / min-chunk-chars.", file=sys.stderr)
        return 1

    chunks_path = args.out_dir / "chunks.parquet"
    chunks.to_parquet(chunks_path, index=False)
    print(f"Saved {len(chunks)} chunks -> {chunks_path}")

    # --- 4) Load embedding model ---
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if dim != EMBED_DIM and args.model == MODEL_NAME:
        print(f"Warning: expected dim {EMBED_DIM}, got {dim}")

    # --- 5) Embed all chunks (slowest step) ---
    print("Embedding chunks ...")
    embeddings = embed_texts(model, chunks["chunk_text"].tolist(), batch_size=args.batch_size)
    emb_path = args.out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} -> {emb_path}")

    # --- 6) Build searchable FAISS index ---
    print("Building FAISS IndexFlatIP ...")
    index = build_faiss_index(embeddings)
    index_path = args.out_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index ({index.ntotal} vectors) -> {index_path}")

    # --- 7) Save design choices for UI / evaluation ---
    config = {
        "model_name": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "similarity": "cosine",
        "implementation": "L2-normalize embeddings + FAISS IndexFlatIP (dot product == cosine)",
        "index_type": "IndexFlatIP",
        "num_songs": int(len(songs)),
        "num_chunks": int(len(chunks)),
        "max_songs_used": args.max_songs,
        "min_chunk_chars": args.min_chunk_chars,
        "chunking": "sentence/line split + short-fragment merge + chorus dedupe",
        "song_aggregation": "max chunk score per song",
    }
    config_path = args.out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved config -> {config_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
