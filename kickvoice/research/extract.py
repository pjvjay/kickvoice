"""
Claim extraction: turn collected text into citable positions and arguments.

The model proposes claims; the code decides what's kept:
  - every claim must cite a chunk from the batch it was given
  - its quote must actually appear in that chunk (tolerating small
    punctuation/case differences); the *source's* words are stored, not the
    model's rendering of them — so a quote shown live is always verbatim
  - jokes, sarcasm and role-play are kept but flagged insincere, so they
    aren't thrown back at someone as a serious position
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from kickvoice.config import Config
from kickvoice.research.llm import LLMClient, complete_json
from kickvoice.research.store import (
    ABOUT, AUTHORSHIP_LABELS, Chunk, Claim, Document, Person, ResearchStore, deep_link,
)

logger = logging.getLogger(__name__)

CLAIM_KINDS = ("position", "argument", "factual_claim", "prediction")

SYSTEM = """You are a meticulous research assistant preparing a debater for a public debate.
You extract the publicly stated positions and arguments of ONE public figure (the subject)
from source excerpts, so they can be cited accurately. Accuracy matters more than volume:
a misattributed or paraphrased "quote" used live would be embarrassing and unfair.

Rules:
- Only extract what the SUBJECT asserts. In transcripts and threads other people may speak;
  never attribute their words to the subject.
- For excerpts marked "reported about them", extract only positions the text explicitly
  attributes to the subject.
- "quote" must be copied VERBATIM from the excerpt (a contiguous span, 5-60 words).
- "claim" is a neutral one-sentence paraphrase of what they assert.
- "topic" is a short lowercase noun phrase (e.g. "gun control", "minimum wage").
- "kind" is one of: position, argument, factual_claim, prediction.
- "sincere" is false for jokes, sarcasm, trolling, role-play, or when they are
  describing someone else's view rather than their own.
- Skip small talk, game commentary and anything about private life (family, health,
  relationships, location). Only issues they argue about publicly.
- If an excerpt contains no such claims, return nothing for it.

Reply with ONLY JSON: {"claims": [{"chunk": "K12", "topic": "...", "claim": "...",
"quote": "...", "kind": "position", "sincere": true}]}"""


@dataclass
class ExtractionStats:
    chunks: int = 0
    batches: int = 0
    claims: int = 0
    rejected_quotes: int = 0
    failed_batches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


def estimate_tokens(items: list[tuple[Chunk, Document]], config: Config) -> tuple[int, int]:
    """Rough (input, output) token estimate for a run (≈4 chars per token)."""
    chars = sum(len(c.text) + 40 for c, _ in items)
    batches = max(1, -(-chars // config.research.extraction_batch_chars))
    return chars // 4 + batches * len(SYSTEM) // 4, batches * 800


def make_batches(items: list[tuple[Chunk, Document]], max_chars: int) -> list[list[tuple[Chunk, Document]]]:
    batches, current, size = [], [], 0
    for chunk, doc in items:
        n = len(chunk.text) + 80
        if current and size + n > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append((chunk, doc))
        size += n
    if current:
        batches.append(current)
    return batches


def render_batch(person: Person, batch: list[tuple[Chunk, Document]]) -> str:
    parts = [f"Subject: {person.name}" + (f" ({person.description})" if person.description else ""),
             ""]
    for chunk, doc in batch:
        header = (f"[K{chunk.id}] source={doc.source} kind={doc.kind} "
                  f"({AUTHORSHIP_LABELS.get(doc.authorship, doc.authorship)})"
                  + (f" date={doc.published_at[:10]}" if doc.published_at else "")
                  + (f" title={doc.title[:80]!r}" if doc.title else ""))
        parts += [header, chunk.text, ""]
    return "\n".join(parts)


# ── quote verification ───────────────────────────────────────────

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _norm(word: str) -> str:
    return word.lower().replace("’", "'").replace("‘", "'").strip("'")


def find_quote(quote: str, source: str, min_ratio: float = 0.85) -> str | None:
    """
    Locate `quote` in `source` and return the source's own words for it, or
    None if it isn't really there. Tolerates case, punctuation and a word or
    two of drift (models often "tidy" quotes); rejects paraphrases.
    """
    q = [_norm(w) for w in _WORD.findall(quote)]
    if len(q) < 3:
        return None
    spans = [(m.start(), m.end(), _norm(m.group())) for m in _WORD.finditer(source)]
    words = [w for _, _, w in spans]
    if len(words) < len(q) - 2:
        return None

    best_ratio, best = 0.0, None
    lengths = {len(q), len(q) - 1, len(q) + 1, len(q) - 2, len(q) + 2}
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(q)
    for n in sorted(x for x in lengths if x >= 3):
        for i in range(0, len(words) - n + 1):
            matcher.set_seq1(words[i:i + n])
            # Upper bounds first: skip windows that can't beat the best so far.
            if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (i, i + n)
    if best is None or best_ratio < min_ratio:
        return None
    start, end = spans[best[0]][0], spans[best[1] - 1][1]
    return source[start:end]


# ── extraction ───────────────────────────────────────────────────

def extract_claims(
    store: ResearchStore,
    person: Person,
    llm: LLMClient,
    config: Config,
    limit_chunks: int | None = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> ExtractionStats:
    stats = ExtractionStats()
    items = store.chunks_to_analyze(person.id, limit=limit_chunks)
    stats.chunks = len(items)
    model = config.llm.analysis_model
    batches = make_batches(items, config.research.extraction_batch_chars)
    for n, batch in enumerate(batches, 1):
        progress(f"analysing batch {n}/{len(batches)} ({len(batch)} excerpt(s))")
        by_ref = {f"K{c.id}": (c, d) for c, d in batch}
        try:
            raw, reply = complete_json(
                llm, SYSTEM, render_batch(person, batch), model,
                validate=_validate_shape, max_tokens=4000)
        except Exception as e:                            # keep going; retry later
            stats.failed_batches += 1
            stats.errors.append(f"batch {n}: {e}")
            logger.warning("Extraction batch %d failed: %s", n, e)
            continue
        stats.batches += 1
        stats.input_tokens += reply.input_tokens
        stats.output_tokens += reply.output_tokens

        claims = []
        for item in raw:
            ref = by_ref.get(str(item["chunk"]).strip("[]"))
            if ref is None:
                stats.rejected_quotes += 1
                continue
            chunk, doc = ref
            verbatim = find_quote(item["quote"], chunk.text)
            if verbatim is None:
                stats.rejected_quotes += 1
                logger.debug("Rejected unverifiable quote: %r", item["quote"][:80])
                continue
            claims.append(Claim(
                person_id=person.id, document_id=doc.id, chunk_id=chunk.id,
                topic=item["topic"].strip().lower()[:80], claim=item["claim"].strip(),
                quote=verbatim, kind=item["kind"], sincere=item["sincere"],
                authorship=doc.authorship, published_at=doc.published_at,
                url=deep_link(doc.url, chunk.start_s), model=reply.model))
        store.add_claims(claims)
        store.mark_analyzed([c.id for c, _ in batch], model)
        stats.claims += len(claims)
    return stats


def _validate_shape(data) -> list[dict]:
    """Structural check only; per-item problems are dropped, not retried."""
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError('expected an object like {"claims": [...]}')
    clean = []
    for item in data["claims"]:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(k), str) and item.get(k).strip()
                   for k in ("chunk", "topic", "claim", "quote")):
            continue
        kind = item.get("kind") if item.get("kind") in CLAIM_KINDS else "position"
        clean.append({**item, "kind": kind, "sincere": item.get("sincere") is not False})
    return clean


__all__ = ["extract_claims", "find_quote", "estimate_tokens", "ABOUT"]
