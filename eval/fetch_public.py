"""eval/fetch_public.py — fetch public-domain test images from Wikimedia Commons.

Each image is picked by a descriptive search term, so the title itself is the
ground truth anchor for the manifest (e.g. a file titled "Baseball game at
Wrigley Field.jpg" must yield a ``baseball`` sports signal).

Idempotent: already-downloaded images are skipped. Writes a provenance map to
``eval/images/public/.sources.json`` (slug -> title/url) so manifest entries
can cite where each expectation came from.

Usage:
    python eval/fetch_public.py [--terms baseball,hockey,...]

No auth, no tokens. Requires network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "DeepSightEval/0.2 (test fixture fetcher for the deepsight repo)"
OUT_DIR = Path(__file__).resolve().parent / "images" / "public"

DEFAULT_TERMS = [
    ("baseball", "baseball pitcher game"),
    ("hockey", "ice hockey game arena"),
    ("tennis", "tennis player court"),
    ("surfing", "surfer wave"),
    ("soccer", "soccer match stadium"),
    ("stop-sign", "stop sign road intersection"),
    ("eiffel", "Eiffel Tower Paris"),
    ("pizza", "pizza on table"),
    ("cat", "cat close up"),
    ("street", "New York City street"),
]

MIN_WIDTH = 480
MIN_HEIGHT = 360


def _get(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 3.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(delay * (attempt + 1) * 1.5)
                continue
            raise
    raise RuntimeError("unreachable")


def search_top_image(query: str) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": "10",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "800",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    data = json.loads(_get(url))
    pages = (data.get("query") or {}).get("pages") or {}
    for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
            continue
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        ext = "jpg" if mime == "image/jpeg" else "png"
        title = page.get("title", "?")
        license = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "?")
        # Prefer the cached thumbnail (gentler on the API); fall back to full-res.
        url = info.get("thumburl") or info["url"]
        return {"url": url, "title": title, "ext": ext, "license": license}
    return None


def slugify(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", title.rsplit(".", 1)[0]).strip("-").lower()
    return base[:80] or "image"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terms", default=",".join(t for t, _ in DEFAULT_TERMS))
    args = ap.parse_args()

    wanted = dict(DEFAULT_TERMS)
    if args.terms != ",".join(t for t, _ in DEFAULT_TERMS):
        wanted = {t.strip(): t.strip() for t in args.terms.split(",") if t.strip()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources_path = OUT_DIR / ".sources.json"
    sources: dict = json.loads(sources_path.read_text()) if sources_path.exists() else {}

    for slug, query in wanted.items():
        dest = OUT_DIR / f"{slug}.jpg"
        if dest.exists():
            print(f"  cached  {slug}")
            continue
        try:
            hit = search_top_image(query)
        except urllib.error.HTTPError as exc:
            print(f"  SKIP    {slug}  (HTTP {exc.code})")
            continue
        if not hit:
            print(f"  MISS    {slug}  (no bitmap >= {MIN_WIDTH}x{MIN_HEIGHT})")
            continue
        try:
            data = _get(hit["url"])
        except Exception as exc:  # noqa: BLE001 — network fallthrough
            print(f"  FAIL    {slug}  ({exc})")
            continue
        dest.write_bytes(data)
        sources[slug] = {"query": query, "title": hit["title"], "url": hit["url"],
                         "license": hit["license"]}
        print(f"  fetched {slug}  <- {hit['title']}  [{hit['license']}]")
        time.sleep(3.0)  # be gentle with the Commons API

    sources_path.write_text(json.dumps(sources, indent=2))
    print(f"\n  provenance: {sources_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
