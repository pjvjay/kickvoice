"""
Pre-debate brief: their positions by topic, verbatim quotes, possible
contradictions, arguments to expect, and prepared counters.

Model-written text (summaries, counters) is labelled as such; every quote
and date comes from the database, and a contradiction must pair two of
their OWN sincere statements.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from kickvoice.config import Config
from kickvoice.research.llm import LLMClient, complete_json
from kickvoice.research.store import (
    AUTHORSHIP_LABELS,
    FIRST_HAND,
    Claim,
    Person,
    ResearchStore,
)

logger = logging.getLogger(__name__)

MAX_CLAIMS_PER_TOPIC = 30

SYSTEM = """You help a debater prepare to debate {name}. You get {name}'s statements on one
topic, each with an id, date, and whether it's their own words or reported about them.

Produce:
- "summary": {name}'s position on the topic in 1-2 neutral sentences.
- "contradictions": pairs of THEIR OWN statements (ids "a" and "b") that genuinely conflict
  (e.g. a changed position). Note if it may simply be a change of mind over time. Don't stretch.
- "expected_arguments": up to 4 arguments {name} is likely to make on this topic.
- "counters": up to 4 strong, fair counter-arguments, citing statement ids where relevant.
Only use ids from the list. Never invent quotes, dates or statistics.

Reply with ONLY JSON: {{"summary": "...", "contradictions": [{{"a": "C1", "b": "C7",
"explanation": "..."}}], "expected_arguments": ["..."], "counters": [{{"text": "...",
"evidence": ["C3"]}}]}}"""


@dataclass
class TopicBrief:
    topic: str
    claims: list[Claim]
    summary: str = ""
    contradictions: list[tuple[Claim, Claim, str]] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    counters: list[tuple[str, list[Claim]]] = field(default_factory=list)
    error: str = ""


def brief_topics(store: ResearchStore, person: Person, topic: str | None, max_topics: int) -> list[str]:
    topics = [t for t, _ in store.topics(person.id)]
    if topic:
        wanted = topic.lower()
        topics = [t for t in topics if wanted in t.lower()] or [topic]
    return topics[:max_topics]


def build_topic(store, person, llm, config, topic: str) -> TopicBrief:
    claims = store.claims(person.id, topic=topic, sincere_only=True)
    # Their own words first, newest first within each group (stable two-pass sort).
    claims.sort(key=lambda c: c.published_at or "", reverse=True)
    claims.sort(key=lambda c: c.authorship not in FIRST_HAND)
    claims = claims[:MAX_CLAIMS_PER_TOPIC]
    tb = TopicBrief(topic=topic, claims=claims)
    if not claims:
        tb.error = "no claims on this topic yet"
        return tb

    by_ref = {f"C{c.id}": c for c in claims}
    lines = [f"Topic: {topic}", "", f"{person.name}'s statements:"]
    for ref, c in by_ref.items():
        when = (c.published_at or "undated")[:10]
        lines.append(
            f"[{ref}] ({AUTHORSHIP_LABELS.get(c.authorship, c.authorship)}, {when}) "
            f'{c.claim} — quote: "{c.quote}"'
        )
    try:
        data, _ = complete_json(
            llm,
            SYSTEM.format(name=person.name),
            "\n".join(lines),
            config.llm.analysis_model,
            validate=_validate,
            max_tokens=2000,
        )
    except Exception as e:
        tb.error = str(e)
        return tb

    tb.summary = data["summary"]
    tb.expected = data["expected_arguments"][:4]
    for item in data["contradictions"]:
        a, b = by_ref.get(item["a"]), by_ref.get(item["b"])
        if a and b and a.id != b.id and a.authorship in FIRST_HAND and b.authorship in FIRST_HAND:
            first, second = sorted((a, b), key=lambda c: c.published_at or "")
            tb.contradictions.append((first, second, item["explanation"]))
    for item in data["counters"]:
        cited = [by_ref[r] for r in item["evidence"] if r in by_ref]
        if len(cited) == len(item["evidence"]) and len(tb.counters) < 4:  # unknown ids → dropped
            tb.counters.append((item["text"], cited))
    return tb


def build_brief(
    store: ResearchStore,
    person: Person,
    llm: LLMClient,
    config: Config,
    topic: str | None = None,
    max_topics: int = 8,
    progress: Callable[[str], None] = lambda _m: None,
) -> str:
    topics = brief_topics(store, person, topic, max_topics)
    sections = []
    for i, t in enumerate(topics, 1):
        progress(f"briefing topic {i}/{len(topics)}: {t}")
        sections.append(build_topic(store, person, llm, config, t))
    return render_markdown(store, person, sections)


def _cite(c: Claim) -> str:
    when = (c.published_at or "undated")[:10]
    link = f" — [source]({c.url})" if c.url else ""
    return f'"{c.quote}" ({when}, {AUTHORSHIP_LABELS.get(c.authorship, c.authorship)}){link}'


def render_markdown(store: ResearchStore, person: Person, sections: list[TopicBrief]) -> str:
    stats = store.stats(person.id)
    sources = ", ".join(f"{k} {v}" for k, v in sorted(stats["by_source"].items()))
    out = [
        f"# Debate brief: {person.name}",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from {stats['documents']} documents "
        f"({sources or 'none'}) and {stats['claims']} extracted claims._  ",
        "_Quotes are verbatim and were checked against their sources. Summaries and "
        "counters are model-written — verify before relying on them._",
        "",
    ]
    if not sections:
        out.append("No topics yet — run `research collect` and `research analyze` first.")
    for tb in sections:
        out += [f"## {tb.topic.title()}", ""]
        if tb.error and not tb.claims:
            out += [f"_{tb.error}_", ""]
            continue
        if tb.summary:
            out += [f"**Their position** _(model summary)_: {tb.summary}", ""]
        out.append("**Key quotes**")
        out += [f"- {_cite(c)}" for c in tb.claims[:6]]
        out.append("")
        if tb.contradictions:
            out.append("**Possible contradictions** _(check context before using)_")
            for a, b, why in tb.contradictions:
                out += [f"- ⚠ {why}", f"  - {_cite(a)}", f"  - {_cite(b)}"]
            out.append("")
        if tb.expected:
            out.append("**Arguments to expect** _(model)_")
            out += [f"- {e}" for e in tb.expected]
            out.append("")
        if tb.counters:
            out.append("**Counters** _(model)_")
            for text, cited in tb.counters:
                refs = "; ".join(f'"{c.quote[:80]}" ({(c.published_at or "undated")[:10]})' for c in cited)
                out.append(f"- {text}" + (f"  \n  ↳ {refs}" if refs else ""))
            out.append("")
        if tb.error:
            out += [f"_Analysis failed for this topic: {tb.error}_", ""]
    return "\n".join(out).rstrip() + "\n"


def _validate(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")

    def refs(v):
        return [str(x).strip("[]") for x in (v if isinstance(v, list) else [v]) if x]

    return {
        "summary": str(data.get("summary") or "").strip(),
        "contradictions": [
            {
                "a": str(c.get("a", "")).strip("[]"),
                "b": str(c.get("b", "")).strip("[]"),
                "explanation": str(c.get("explanation") or "").strip(),
            }
            for c in (data.get("contradictions") or [])
            if isinstance(c, dict)
        ],
        "expected_arguments": [
            str(x).strip() for x in (data.get("expected_arguments") or []) if str(x).strip()
        ],
        "counters": [
            {"text": str(c["text"]).strip(), "evidence": refs(c.get("evidence") or [])}
            for c in (data.get("counters") or [])
            if isinstance(c, dict) and str(c.get("text") or "").strip()
        ],
    }
