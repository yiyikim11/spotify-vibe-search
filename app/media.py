"""
Enrich search hits with cover art + YouTube listen links.

Does NOT re-chunk or re-embed. Media is display metadata only:
  - Cover image: iTunes Search API (free, no key) matched by artist + title
  - YouTube: best matching video via Invidious public API when available,
    otherwise a precise YouTube search URL for that artist + song
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "artifacts" / "media_cache.json"

# Public Invidious instances (tried in order) for resolving a real watch URL
INVIDIOUS_SEARCH = [
    "https://inv.nadeko.net/api/v1/search",
    "https://yewtu.be/api/v1/search",
    "https://invidious.fdn.fr/api/v1/search",
]

_UA = {"User-Agent": "Mozilla/5.0 (compatible; SpotifyVibeSearch/1.0)"}


def _get_json(url: str, timeout: float = 4.0) -> object | None:
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def youtube_search_url(artist: str, song: str) -> str:
    """Always-valid Listen link: opens YouTube search for this exact track."""
    q = f'{artist} {song}'
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)


def _youtube_watch_url(artist: str, song: str) -> str | None:
    """Try to resolve a direct youtube.com/watch?v=... URL via Invidious search."""
    q = urllib.parse.quote_plus(f"{artist} {song}")
    for base in INVIDIOUS_SEARCH:
        data = _get_json(f"{base}?q={q}&type=video")
        if not isinstance(data, list) or not data:
            continue
        video_id = data[0].get("videoId")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return None


def _itunes_artwork(artist: str, song: str) -> str | None:
    """Fetch best-matching cover art from iTunes (upgrade 100px → 600px)."""
    term = urllib.parse.quote_plus(f"{artist} {song}")
    data = _get_json(
        f"https://itunes.apple.com/search?term={term}&media=music&entity=song&limit=5"
    )
    if not isinstance(data, dict):
        return None
    results = data.get("results") or []
    if not results:
        return None

    want_artist = _norm(artist)
    want_song = _norm(song)
    best = None
    best_score = -1
    for item in results:
        a = _norm(str(item.get("artistName", "")))
        t = _norm(str(item.get("trackName", "")))
        score = 0
        if want_artist and want_artist in a:
            score += 2
        if a and a in want_artist:
            score += 1
        if want_song and (want_song in t or t in want_song):
            score += 2
        if score > best_score:
            best_score = score
            best = item

    if not best or best_score <= 0:
        best = results[0]

    art = best.get("artworkUrl100") or best.get("artworkUrl60")
    if not art:
        return None
    # iTunes serves larger art by rewriting the size token in the URL
    return re.sub(r"\d+x\d+bb", "600x600bb", str(art))


class MediaEnricher:
    """Cached lookup of image_url + youtube_url per artist/song."""

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, str | None]] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _key(self, artist: str, song: str) -> str:
        return f"{_norm(artist)}|{_norm(song)}"

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def enrich(self, artist: str, song: str) -> tuple[str | None, str]:
        """
        Returns (image_url, youtube_url).
        youtube_url always set (direct watch if found, else YouTube search).
        """
        key = self._key(artist, song)
        if key in self._cache:
            row = self._cache[key]
            yt = row.get("youtube_url") or youtube_search_url(artist, song)
            return row.get("image_url"), yt

        image_url = _itunes_artwork(artist, song)
        youtube_url = _youtube_watch_url(artist, song) or youtube_search_url(artist, song)

        self._cache[key] = {"image_url": image_url, "youtube_url": youtube_url}
        self._save()
        return image_url, youtube_url
