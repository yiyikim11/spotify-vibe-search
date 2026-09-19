# Spotify Semantic / Vibe Search (Week 2)

Search songs by natural-language mood descriptions using Sentence Transformers + FAISS.

## Demo UI

![Spotify Vibe Search UI](docs/ui-screenshot.png)

## Design choices (evaluation)

| Question | Answer |
|---|---|
| Embedding dimension | **384** (`all-MiniLM-L6-v2`) |
| Representation | Sentence Transformers dense embeddings; same model for queries and lyric chunks |
| Long lyrics | Sentence/line **chunking**, then **max score per song** |
| Similarity | **Cosine** (L2-normalize vectors, search with FAISS `IndexFlatIP` so dot product == cosine) |
| Index | **Brute-force Flat** — exact search, high recall, fine for ~57k songs |

## Setup (local)

```bash
cd week2-exercise
pip install -r requirements.txt
```

Place CSV at `data/raw/spotify_millsongdata.csv`.

### Build index

```bash
# Deploy-sized sample (fits Render free / GitHub)
python scripts/prepare_and_index.py --max-songs 2500

# Full dataset (local only; large artifacts)
python scripts/prepare_and_index.py
```

### Run web app

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Deploy on Render (free)

Render free instances only have **512MB RAM**. PyTorch + Sentence-Transformers will OOM.
This project uses **fastembed (ONNX MiniLM)** instead, plus a smaller committed index (~800 songs).

1. Push this repo to GitHub (include updated `artifacts/`).
2. Render → **Manual Deploy** / Blueprint with `render.yaml`.
3. Health check path: `/health` (does not load the model).
4. First real page load downloads the ONNX model once (cold start).

If it still OOMs: rebuild even smaller locally and redeploy:

```bash
python scripts/prepare_and_index.py --max-songs 500
```

## Pipeline

1. Load CSV → chunk lyrics → embed (384-D) → FAISS Flat IP  
2. Query → embed → search chunks → max-pool to songs → UI (+ cover / YouTube)

## Project layout

```
week2-exercise/
  app/main.py
  app/search.py
  app/media.py
  app/templates/index.html
  scripts/prepare_and_index.py
  artifacts/          # faiss.index, chunks.parquet, config.json
  docs/ui-screenshot.png
  render.yaml
  requirements.txt
```
