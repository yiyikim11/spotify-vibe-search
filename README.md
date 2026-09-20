# Spotify Semantic / Vibe Search (Week 2)

Search songs by natural-language mood descriptions using dense lyric embeddings + FAISS.

## Demo UI

![Spotify Vibe Search UI](docs/ui-screenshot.png)

## Design choices (evaluation)

| Question | Answer |
|---|---|
| Embedding dimension | **384** (`all-MiniLM-L6-v2`) |
| Representation | Dense MiniLM embeddings; same model family for queries and lyric chunks |
| Long lyrics | Sentence/line **chunking**, then **max score per song** |
| Similarity | **Cosine** (L2-normalize + FAISS `IndexFlatIP`) |
| Index | **Brute-force Flat** — exact search |

## Architecture (local build vs Render)

```text
Laptop (once):
  CSV → chunk lyrics → embed with fastembed ONNX → save faiss.index + chunks.parquet

Render (runtime, low RAM):
  user query → Hugging Face Inference API (embed query only)
            → FAISS search on prebuilt index
            → ranked songs + YouTube / cover
```

Lyrics are **not** re-embedded on Render. Only the short query is embedded remotely.

## Setup (local)

```bash
cd week2-exercise
pip install -r requirements-local.txt
```

Place CSV at `data/raw/spotify_millsongdata.csv`.

### Build index (on your machine)

```bash
python scripts/prepare_and_index.py --max-songs 800
# or larger for local demos:
# python scripts/prepare_and_index.py --max-songs 2500
```

### Run web app locally

```bash
# optional: use local ONNX for queries (default without HF_TOKEN)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# or test the Render path (remote query embeddings):
# set HF_TOKEN=hf_xxx
# set EMBED_BACKEND=remote
```

Open http://127.0.0.1:8000

## Deploy on Render (free / 512MB)

1. Push repo to GitHub (include `artifacts/faiss.index`, `chunks.parquet`, `config.json`).
2. Create free HF token: https://huggingface.co/settings/tokens  
3. Render → Blueprint / Web Service with `render.yaml`.
4. In **Environment**, set secret **`HF_TOKEN`** = your token (`EMBED_BACKEND=remote` is already in `render.yaml`).
5. Deploy. Health check: `/health` (does not call HF / load a model).

**Why this fits 512MB:** Render never loads MiniLM weights — only FAISS (~tens of MB) + pandas metadata + FastAPI. Query vectors come from Hugging Face’s API.

## Pipeline

1. Offline: chunk → embed lyrics → FAISS Flat IP  
2. Online: embed query (remote or local) → search → max-pool songs → UI  

## Project layout

```
week2-exercise/
  app/main.py
  app/search.py
  app/embedder.py      # remote HF + local fastembed
  app/media.py
  app/templates/index.html
  scripts/prepare_and_index.py
  artifacts/           # prebuilt index (commit for Render)
  requirements.txt         # Render (no onnx/torch)
  requirements-local.txt   # laptop index builds
  render.yaml
```
