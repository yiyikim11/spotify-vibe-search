"""
FastAPI web UI for Spotify vibe / semantic lyric search.

Memory notes (Render free 512MB):
  - Lyrics are pre-embedded offline into FAISS (not recomputed on Render)
  - Query embedding uses Hugging Face API (EMBED_BACKEND=remote + HF_TOKEN)
  - No local MiniLM / PyTorch / ONNX weights loaded in the web process
  - /health does not load FAISS or call HF
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.search import VibeSearchEngine

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

app = FastAPI(title="Spotify Vibe Search", version="1.0.0")

# Lazy singleton — loaded on first real page/search, not during cold health checks
_engine: VibeSearchEngine | None = None
_engine_error: str | None = None


def get_engine() -> VibeSearchEngine:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise RuntimeError(_engine_error)
    try:
        _engine = VibeSearchEngine()
        return _engine
    except Exception as exc:  # noqa: BLE001
        _engine_error = str(exc)
        raise


@app.get("/health")
def health() -> JSONResponse:
    """Lightweight probe for Render — does not load MiniLM/FAISS."""
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    cfg = {}
    error = None
    try:
        cfg = get_engine().config
    except Exception as exc:  # noqa: BLE001
        error = f"Engine failed to load (often RAM on free tier): {exc}"

    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "results": None,
            "query": "",
            "error": error,
            "config": cfg,
            "top_k": 5,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = Form(...),
    top_k: int = Form(5),
) -> HTMLResponse:
    query = q.strip()
    error = None
    results = []
    cfg = {}
    try:
        engine = get_engine()
        cfg = engine.config
        top_k = max(1, min(int(top_k), 50))
        results = engine.search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "results": results,
            "query": query,
            "error": error,
            "config": cfg,
            "top_k": top_k,
        },
    )
