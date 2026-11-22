"""
Wikipedia (background, reported positions) and Wikidata (official accounts).

Wikidata is the identity backbone: editors record the accounts a person
publicly claims (X username, YouTube channel, Twitch, Reddit, website…).
Linking accounts through it — rather than guessing from similar usernames —
keeps the research about the right person and never unmasks alt accounts.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from kickvoice.research.net import Fetcher
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import ABOUT, Account, Chunk, Document

# Wikidata properties for public accounts → (platform, url template)
ACCOUNT_PROPERTIES = {
    "P2002": ("x", "https://x.com/{}"),
    "P2397": ("youtube", "https://www.youtube.com/channel/{}"),
    "P5797": ("twitch", "https://www.twitch.tv/{}"),
    "P4265": ("reddit", "https://www.reddit.com/user/{}"),
    "P2003": ("instagram", "https://www.instagram.com/{}"),
    "P7085": ("tiktok", "https://www.tiktok.com/@{}"),
    "P856": ("website", "{}"),
}
PLATFORM_PROPERTY = {platform: prop for prop, (platform, _) in ACCOUNT_PROPERTIES.items()}


@dataclass
class WikidataPerson:
    qid: str
    label: str
    description: str
    wikipedia_title: str | None
    accounts: list[tuple[str, str, str]]      # (platform, handle, url)


# ── Wikidata ─────────────────────────────────────────────────────

def search_people(fetcher: Fetcher, api: str, name: str, limit: int = 5) -> list[dict]:
    """Wikidata entities matching a name: [{id, label, description}]."""
    data = fetcher.get_json(api, {
        "action": "wbsearchentities", "search": name, "language": "en",
        "type": "item", "limit": limit, "format": "json"})
    return [{"id": r["id"], "label": r.get("label", ""), "description": r.get("description", "")}
            for r in data.get("search", [])]


def find_by_account(fetcher: Fetcher, api: str, platform: str, handle: str) -> list[str]:
    """Wikidata items that list this account (exact statement match)."""
    prop = PLATFORM_PROPERTY.get(platform)
    if not prop:
        return []
    found: list[str] = []
    for value in dict.fromkeys([handle, handle.lower()]):    # stored case varies
        data = fetcher.get_json(api, {
            "action": "query", "list": "search", "format": "json",
            "srsearch": f"haswbstatement:{prop}={value}", "srlimit": 5})
        for hit in data.get("query", {}).get("search", []):
            if hit["title"] not in found:
                found.append(hit["title"])
    return found


def get_person(fetcher: Fetcher, api: str, qid: str) -> WikidataPerson:
    data = fetcher.get_json(api, {
        "action": "wbgetentities", "ids": qid, "format": "json",
        "props": "labels|descriptions|claims|sitelinks", "languages": "en"})
    entity = data.get("entities", {}).get(qid)
    if not entity or "missing" in entity:
        raise ValueError(f"Wikidata item {qid} not found")
    accounts = []
    for prop, (platform, template) in ACCOUNT_PROPERTIES.items():
        for statement in entity.get("claims", {}).get(prop, []):
            if statement.get("rank") == "deprecated":
                continue        # e.g. an old, abandoned handle
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(value, str) and value:
                accounts.append((platform, value, template.format(value)))
    return WikidataPerson(
        qid=qid,
        label=entity.get("labels", {}).get("en", {}).get("value", qid),
        description=entity.get("descriptions", {}).get("en", {}).get("value", ""),
        wikipedia_title=entity.get("sitelinks", {}).get("enwiki", {}).get("title"),
        accounts=accounts,
    )


# ── Wikipedia ────────────────────────────────────────────────────

def wikipedia_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def collect_wikipedia(ctx: SourceContext, account: Account) -> CollectStats:
    """Fetch the article as plain text. It's background *about* them."""
    stats = CollectStats()
    api = ctx.config.research.wikipedia_api
    data = ctx.fetcher.get_json(api, {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|revisions|pageprops", "explaintext": 1, "rvprop": "timestamp",
        "titles": account.handle})
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        stats.errors.append(f"Wikipedia article not found: {account.handle}")
        return stats
    page = pages[0]
    text = page.get("extract", "")
    revisions = page.get("revisions") or [{}]
    doc = Document(
        person_id=ctx.person.id, source="wikipedia", kind="article",
        external_id=page["title"], text=text, authorship=ABOUT,
        url=wikipedia_url(page["title"]), title=page["title"],
        published_at=revisions[0].get("timestamp"),
        meta={"wikidata": page.get("pageprops", {}).get("wikibase_item")},
    )
    ctx.save(doc, _sections(text), stats)
    ctx.progress(f"wikipedia: {page['title']} ({len(text):,} chars)")
    return stats


def _sections(text: str, max_chars: int = 1500) -> list[Chunk]:
    """Chunk by '== Section ==' headings, prefixing each chunk with its section."""
    from kickvoice.research.chunking import chunk_text

    chunks: list[Chunk] = []
    section, buf = "Introduction", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            for c in chunk_text(body, max_chars):
                chunks.append(Chunk(f"[{section}] {c.text}"))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("==") and stripped.endswith("=="):
            flush()
            section, buf = stripped.strip("= ").strip() or section, []
        else:
            buf.append(line)
    flush()
    return chunks
