"""
Web pages: their own site/blog (first-hand) and articles about them.

Pages are fetched politely (robots.txt respected) and reduced to readable
text. Optional web search uses the Brave Search API (BRAVE_API_KEY).
"""
from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

from kickvoice.research.net import Blocked, NetError
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import ABOUT, SELF, Document

_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form",
              "svg", "button", "iframe", "template"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6",
               "blockquote", "pre", "br", "tr", "main"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.title = ""
        self._in_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            if key and attrs.get("content"):
                self.meta[key.lower()] = attrs["content"]
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str, dict]:
    """(title, readable text, meta tags) — good enough for articles and blogs."""
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    # Drop menu/boilerplate fragments: keep paragraphs that read like prose.
    paragraphs = [p for p in paragraphs if len(p) >= 40 or p.endswith((".", "?", "!"))]
    title = (parser.meta.get("og:title") or parser.title).strip()
    return title, "\n\n".join(paragraphs), parser.meta


def collect_url(ctx: SourceContext, url: str, first_hand: bool) -> CollectStats:
    """One page. `first_hand` = it's their own site (they wrote it)."""
    stats = CollectStats()
    try:
        response = ctx.fetcher.get(url, respect_robots=True)
    except Blocked:
        stats.errors.append(f"robots.txt disallows {url}")
        return stats
    except NetError as e:
        stats.errors.append(f"{url}: {e}")
        return stats
    if response.status != 200:
        stats.errors.append(f"{url}: HTTP {response.status}")
        return stats
    if "html" not in response.headers.get("content-type", "text/html"):
        stats.errors.append(f"{url}: not an HTML page")
        return stats
    title, text, meta = html_to_text(response.text)
    if len(text) < 80:
        stats.errors.append(f"{url}: no readable article text")
        return stats
    ctx.save(Document(
        person_id=ctx.person.id, source="web", kind="article", external_id=url,
        text=text, authorship=SELF if first_hand else ABOUT, url=url, title=title,
        published_at=meta.get("article:published_time") or meta.get("date"),
        meta={"site": urlparse(url).netloc},
    ), None, stats)
    ctx.progress(f"web: {title[:70] or url}")
    return stats


def search_web(ctx: SourceContext, query: str, count: int = 10) -> list[dict]:
    """Brave Search results [{title, url, description}] (needs BRAVE_API_KEY)."""
    rcfg = ctx.config.research
    key = os.environ.get(rcfg.brave_api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"Web search: set the {rcfg.brave_api_key_env} environment variable "
                           "(https://brave.com/search/api/).")
    data = ctx.fetcher.get_json(rcfg.brave_search_api, {"q": query, "count": count},
                                headers={"X-Subscription-Token": key, "Accept": "application/json"})
    return [{"title": r.get("title", ""), "url": r["url"], "description": r.get("description", "")}
            for r in data.get("web", {}).get("results", []) if r.get("url")]
