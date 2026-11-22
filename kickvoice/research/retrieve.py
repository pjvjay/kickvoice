"""
Find the parts of someone's record relevant to what they just said.

Hybrid retrieval:
  - SQLite full-text search (BM25) over extracted claims and raw chunks
  - optional embedding similarity over claims (if an embedding model is set)
Results are merged with reciprocal-rank fusion. Claims rank ahead of raw
chunks at equal relevance because their quotes are verified and dated.
"""
from __future__ import annotations

import re

import numpy as np

from kickvoice.research.llm import EmbeddingClient
from kickvoice.research.store import Evidence, Person, ResearchStore, fts_or_query

RRF_K = 60


class Retriever:
    def __init__(self, store: ResearchStore, person: Person,
                 embedder: EmbeddingClient | None = None):
        self.store = store
        self.person = person
        self.embedder = embedder
        self._ids: list[int] = []
        self._vectors: np.ndarray | None = None
        if embedder is not None:
            self._ids, vectors = store.claim_vectors(person.id)
            self._vectors = vectors if len(self._ids) else None

    def search(self, text: str, k: int = 12, include_chunks: bool = True) -> list[Evidence]:
        rankings: list[list[Evidence]] = [self.store.search_claims(self.person.id, text, k * 2)]
        if self._vectors is not None and self.embedder is not None and text.strip():
            query = self.embedder.embed([text])[0]
            sims = self._vectors @ query
            top = [self._ids[i] for i in np.argsort(-sims)[: k * 2]]
            rankings.append(self.store.claim_evidence(top))
        if include_chunks:
            rankings.append(self.store.search_chunks(self.person.id, text, k))

        fused: dict[tuple[str, int], tuple[float, Evidence]] = {}
        for ranking in rankings:
            for rank, ev in enumerate(ranking):
                bonus = 1.0 if ev.kind == "claim" else 0.8
                score = bonus / (RRF_K + rank)
                key = (ev.kind, ev.id)
                prev = fused.get(key)
                fused[key] = ((prev[0] if prev else 0.0) + score, ev)
        ranked = sorted(fused.values(), key=lambda x: -x[0])
        out = []
        for score, ev in ranked[:k]:
            ev.score = score
            if ev.kind == "chunk":
                ev.quote = snippet(ev.text, text)
            out.append(ev)
        return out


def snippet(text: str, query: str, width: int = 320) -> str:
    """The part of a long chunk that best matches the query words."""
    if len(text) <= width:
        return text
    terms = [t.strip('"') for t in fts_or_query(query).split(" OR ") if t]
    if not terms:
        return text[:width].rsplit(" ", 1)[0] + " …"
    lowered = text.lower()
    best_start, best_hits = 0, -1
    step = max(40, width // 4)
    for start in range(0, max(1, len(text) - width + step), step):
        window = lowered[start:start + width]
        hits = sum(len(re.findall(rf"\b{re.escape(t)}\b", window)) for t in terms)
        if hits > best_hits:
            best_start, best_hits = start, hits
    # Expand to word boundaries.
    start = text.rfind(" ", 0, best_start) + 1 if best_start else 0
    end = text.find(" ", start + width)
    end = len(text) if end == -1 else end
    return ("… " if start else "") + text[start:end].strip() + (" …" if end < len(text) else "")
