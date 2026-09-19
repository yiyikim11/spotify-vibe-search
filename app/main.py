"""
FastAPI web UI for Spotify vibe / semantic lyric search.

Routes:
  GET  /        → search form (empty results)
  POST /search  → run VibeSearchEngine and render ranked songs

The heavy model + FAISS index are loaded once at startup (not per request).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.search import VibeSearchEngine

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

app = FastAPI(title="Spotify Vibe Search", version="1.0.0")

# Shared engine instance (model + FAISS index). Filled in on startup.
engine: VibeSearchEngine | None = None


@app.on_event("startup")
def load_engine() -> None:
    """Load MiniLM + FAISS artifacts once when the server starts."""
    global engine
    engine = VibeSearchEngine()


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Show the landing page with the vibe query text box."""
    cfg = engine.config if engine else {}
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "results": None,
            "query": "",
            "error": None,
            "config": cfg,  # shows dim / metric / index type in the page footer
            "top_k": 5,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = Form(...),  # vibe text from the HTML form
    top_k: int = Form(10),  # how many songs to display
) -> HTMLResponse:
    """
    Handle search form submit:
      embed query → FAISS chunk search → max-pool to songs → render HTML.
    """
    assert engine is not None
    query = q.strip()
    error = None
    results = []
    try:
        top_k = max(1, min(int(top_k), 50))
        results = engine.search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - show in UI for coursework demo
        error = str(exc)

    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "results": results,
            "query": query,
            "error": error,
            "config": engine.config,
            "top_k": top_k,
        },
    )
