"""Shared collector plumbing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kickvoice.config import Config
from kickvoice.research.chunking import chunk_text, chunk_timed
from kickvoice.research.net import Fetcher
from kickvoice.research.store import Chunk, Document, Person, ResearchStore


@dataclass
class CollectStats:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, outcome: str):
        setattr(self, outcome, getattr(self, outcome) + 1)

    def merge(self, other: "CollectStats"):
        self.new += other.new
        self.updated += other.updated
        self.unchanged += other.unchanged
        self.errors += other.errors

    @property
    def total(self) -> int:
        return self.new + self.updated + self.unchanged


@dataclass
class SourceContext:
    config: Config
    fetcher: Fetcher
    store: ResearchStore
    person: Person
    progress: Callable[[str], None] = lambda _msg: None

    def save(self, doc: Document, chunks: list[Chunk] | None = None,
             stats: CollectStats | None = None) -> str:
        """Store a document, chunking its text if no chunks are given."""
        if chunks is None:
            chunks = chunk_text(doc.text)
        _, outcome = self.store.upsert_document(doc, chunks)
        if stats is not None:
            stats.record(outcome)
        return outcome


__all__ = ["CollectStats", "SourceContext", "chunk_text", "chunk_timed"]
