"""
Research database (SQLite): people, their accounts, what they've said, and
the claims extracted from it.

    persons    a researched public figure (optionally linked to a voice profile)
    accounts   their public accounts — "candidate" until you confirm them
    documents  one collected item: a tweet, a video transcript, an article…
    chunks     searchable pieces of documents (with video timestamps)
    claims     positions/arguments extracted from chunks, each with a
               verbatim quote that was checked against the source text

Every document and claim records its *authorship*, because "they tweeted
this" and "Wikipedia says this about them" must never be confused when you
quote someone live.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Authorship — how sure we are the subject said it themselves.
SELF = "self"  # written by them (their tweet, post, comment)
SELF_VOICE = "self_voice"  # their speech, confirmed by voice match
SELF_CHANNEL = "self_channel"  # captions from their own channel (speaker unverified)
ABOUT = "about"  # third party writing about them (Wikipedia, articles)

AUTHORSHIP_LABELS = {
    SELF: "their own words",
    SELF_VOICE: "their voice (verified)",
    SELF_CHANNEL: "their channel (speaker unverified)",
    ABOUT: "reported about them",
}
FIRST_HAND = (SELF, SELF_VOICE, SELF_CHANNEL)

CONFIRMED, CANDIDATE, REJECTED = "confirmed", "candidate", "rejected"

# Documents collected from an account you've since rejected are ignored
# everywhere (search, analysis, briefs, live), even before they're purged.
ACTIVE_DOC = (
    "(d.account_id IS NULL OR d.account_id NOT IN (SELECT id FROM accounts WHERE status = 'rejected'))"
)


@dataclass
class Person:
    id: int
    name: str
    voice_profile: str | None = None
    wikidata_id: str | None = None
    description: str = ""
    created_at: str = ""


@dataclass
class Account:
    id: int
    person_id: int
    platform: str  # wikipedia / youtube / x / reddit / twitch / website / …
    handle: str  # username, channel id, article title or URL
    url: str = ""
    status: str = CANDIDATE
    found_via: str = ""  # where the link came from (e.g. "wikidata Q123 P2002")


@dataclass
class Document:
    person_id: int
    source: str  # platform it came from
    kind: str  # tweet / video / article / comment / post / stream / file
    external_id: str  # stable id within the source (tweet id, video id, URL…)
    text: str
    authorship: str
    url: str = ""
    title: str = ""
    published_at: str | None = None
    meta: dict = field(default_factory=dict)
    id: int | None = None
    account_id: int | None = None  # which linked account it was collected from


@dataclass
class Chunk:
    text: str
    start_s: float | None = None  # for timed media
    end_s: float | None = None
    id: int | None = None
    document_id: int | None = None
    seq: int = 0


@dataclass
class Claim:
    person_id: int
    document_id: int
    chunk_id: int
    topic: str
    claim: str  # one-sentence paraphrase of what they assert
    quote: str  # verbatim excerpt from the chunk (verified)
    kind: str = "position"  # position / factual_claim / prediction / argument
    sincere: bool = True  # False for jokes, sarcasm, role-play, quoting others
    authorship: str = SELF
    published_at: str | None = None
    url: str = ""  # deep link (with timestamp for videos)
    model: str = ""
    id: int | None = None


@dataclass
class Evidence:
    """A retrievable item (claim or raw chunk) with everything needed to cite it."""

    kind: str  # "claim" or "chunk"
    id: int
    text: str  # the claim paraphrase, or the chunk text
    quote: str  # verbatim words to show
    authorship: str
    url: str
    published_at: str | None
    source: str
    title: str
    topic: str = ""
    score: float = 0.0

    @property
    def ref(self) -> str:
        """Short handle for prompts: C12 = claim 12, K7 = chunk 7."""
        return f"{'C' if self.kind == 'claim' else 'K'}{self.id}"


class ResearchStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _create_schema(self):
        self._migrate()
        with self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    voice_profile TEXT, wikidata_id TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    platform TEXT NOT NULL, handle TEXT NOT NULL, url TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL, found_via TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL,
                    UNIQUE (person_id, platform, handle));
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    source TEXT NOT NULL, kind TEXT NOT NULL, external_id TEXT NOT NULL,
                    url TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '',
                    published_at TEXT, authorship TEXT NOT NULL, text TEXT NOT NULL,
                    content_hash TEXT NOT NULL, meta TEXT, fetched_at TEXT NOT NULL,
                    account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
                    UNIQUE (person_id, source, external_id));
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL, text TEXT NOT NULL,
                    start_s REAL, end_s REAL, embedding BLOB);
                CREATE TABLE IF NOT EXISTS chunk_analysis (
                    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, analyzed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    topic TEXT NOT NULL, claim TEXT NOT NULL, quote TEXT NOT NULL,
                    kind TEXT NOT NULL, sincere INTEGER NOT NULL, authorship TEXT NOT NULL,
                    published_at TEXT, url TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, embedding BLOB);
                CREATE INDEX IF NOT EXISTS idx_docs_person ON documents(person_id, source);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, seq);
                CREATE INDEX IF NOT EXISTS idx_claims_person ON claims(person_id, topic);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, content='chunks', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
                CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts
                    USING fts5(topic, claim, quote, content='claims', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
                    INSERT INTO claims_fts(rowid, topic, claim, quote)
                        VALUES (new.id, new.topic, new.claim, new.quote); END;
                CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
                    INSERT INTO claims_fts(claims_fts, rowid, topic, claim, quote)
                        VALUES ('delete', old.id, old.topic, old.claim, old.quote); END;
            """)

    def _migrate(self):
        """Bring databases created by earlier versions up to date."""
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(documents)")}
        if cols and "account_id" not in cols:
            with self._db:
                self._db.execute(
                    "ALTER TABLE documents ADD COLUMN account_id INTEGER "
                    "REFERENCES accounts(id) ON DELETE SET NULL"
                )

    # ── persons ──────────────────────────────────────────────────

    def add_person(
        self,
        name: str,
        voice_profile: str | None = None,
        wikidata_id: str | None = None,
        description: str = "",
    ) -> Person:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO persons (name, voice_profile, wikidata_id, description, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (name, voice_profile, wikidata_id, description, _now()),
            )
        return self.get_person_by_id(cur.lastrowid)

    def get_person(self, name: str) -> Person | None:
        row = self._db.execute("SELECT * FROM persons WHERE name = ?", (name,)).fetchone()
        return Person(**dict(row)) if row else None

    def get_person_by_id(self, person_id: int) -> Person | None:
        row = self._db.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
        return Person(**dict(row)) if row else None

    def list_persons(self) -> list[Person]:
        return [Person(**dict(r)) for r in self._db.execute("SELECT * FROM persons ORDER BY name")]

    def update_person(self, person: Person):
        with self._db:
            self._db.execute(
                "UPDATE persons SET voice_profile = ?, wikidata_id = ?, description = ? WHERE id = ?",
                (person.voice_profile, person.wikidata_id, person.description, person.id),
            )

    def remove_person(self, person_id: int) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return cur.rowcount > 0

    # ── accounts ─────────────────────────────────────────────────

    def add_account(
        self,
        person_id: int,
        platform: str,
        handle: str,
        url: str = "",
        status: str = CANDIDATE,
        found_via: str = "",
    ) -> Account:
        """Insert, or return the existing row (a confirmed status is never downgraded)."""
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO accounts (person_id, platform, handle, url, status,"
                " found_via, added_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (person_id, platform, handle, url, status, found_via, _now()),
            )
            if status == CONFIRMED:
                self._db.execute(
                    "UPDATE accounts SET status = ? WHERE person_id = ? AND platform = ? AND handle = ?",
                    (CONFIRMED, person_id, platform, handle),
                )
        row = self._db.execute(
            "SELECT * FROM accounts WHERE person_id = ? AND platform = ? AND handle = ?",
            (person_id, platform, handle),
        ).fetchone()
        return _account(row)

    def set_account_status(self, account_id: int, status: str):
        with self._db:
            self._db.execute("UPDATE accounts SET status = ? WHERE id = ?", (status, account_id))

    def get_account(self, account_id: int) -> Account | None:
        row = self._db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return _account(row) if row else None

    def documents_from_account(self, account: Account) -> list[int]:
        """
        Ids of documents collected from `account`. Rows collected before
        provenance was recorded (account_id NULL) are matched by URL/title.
        """
        ids = [
            r["id"] for r in self._db.execute("SELECT id FROM documents WHERE account_id = ?", (account.id,))
        ]
        legacy = self._db.execute(
            "SELECT id, source, external_id, url, meta FROM documents"
            " WHERE person_id = ? AND account_id IS NULL",
            (account.person_id,),
        ).fetchall()
        for row in legacy:
            if _legacy_match(account, row):
                ids.append(row["id"])
        return ids

    def delete_documents(self, document_ids: Iterable[int]) -> int:
        """Delete documents with their chunks and claims (cascades)."""
        ids = list(document_ids)
        with self._db:
            self._db.executemany("DELETE FROM documents WHERE id = ?", [(i,) for i in ids])
        return len(ids)

    def accounts(
        self, person_id: int, status: str | None = None, platform: str | None = None
    ) -> list[Account]:
        sql, params = "SELECT * FROM accounts WHERE person_id = ?", [person_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        if platform:
            sql += " AND platform = ?"
            params.append(platform)
        return [_account(r) for r in self._db.execute(sql + " ORDER BY platform, handle", params)]

    # ── documents & chunks ───────────────────────────────────────

    def upsert_document(self, doc: Document, chunks: list[Chunk]) -> tuple[int, str]:
        """
        Store a document and its chunks. Returns (document_id, outcome) where
        outcome is "new", "updated" (text changed: old chunks/claims replaced)
        or "unchanged".
        """
        digest = hashlib.sha256(doc.text.encode()).hexdigest()
        existing = self._db.execute(
            "SELECT id, content_hash FROM documents WHERE person_id = ? AND source = ? AND external_id = ?",
            (doc.person_id, doc.source, doc.external_id),
        ).fetchone()
        with self._db:
            if existing and existing["content_hash"] == digest:
                doc.id = existing["id"]
                if doc.account_id is not None:
                    self._db.execute(
                        "UPDATE documents SET account_id = ? WHERE id = ? AND account_id IS NULL",
                        (doc.account_id, doc.id),
                    )
                return doc.id, "unchanged"
            if existing:
                self._db.execute("DELETE FROM chunks WHERE document_id = ?", (existing["id"],))
                self._db.execute(
                    "UPDATE documents SET url = ?, title = ?, published_at = ?, authorship = ?,"
                    " text = ?, content_hash = ?, meta = ?, fetched_at = ?, kind = ?,"
                    " account_id = COALESCE(?, account_id) WHERE id = ?",
                    (
                        doc.url,
                        doc.title,
                        doc.published_at,
                        doc.authorship,
                        doc.text,
                        digest,
                        json.dumps(doc.meta),
                        _now(),
                        doc.kind,
                        doc.account_id,
                        existing["id"],
                    ),
                )
                doc.id, outcome = existing["id"], "updated"
            else:
                cur = self._db.execute(
                    "INSERT INTO documents (person_id, source, kind, external_id, url, title,"
                    " published_at, authorship, text, content_hash, meta, fetched_at, account_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc.person_id,
                        doc.source,
                        doc.kind,
                        doc.external_id,
                        doc.url,
                        doc.title,
                        doc.published_at,
                        doc.authorship,
                        doc.text,
                        digest,
                        json.dumps(doc.meta),
                        _now(),
                        doc.account_id,
                    ),
                )
                doc.id, outcome = cur.lastrowid, "new"
            for seq, chunk in enumerate(chunks):
                cur = self._db.execute(
                    "INSERT INTO chunks (document_id, seq, text, start_s, end_s) VALUES (?, ?, ?, ?, ?)",
                    (doc.id, seq, chunk.text, chunk.start_s, chunk.end_s),
                )
                chunk.id, chunk.document_id, chunk.seq = cur.lastrowid, doc.id, seq
        return doc.id, outcome

    def get_document(self, document_id: int) -> Document | None:
        row = self._db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _document(row) if row else None

    def documents(self, person_id: int, source: str | None = None) -> list[Document]:
        sql, params = "SELECT * FROM documents WHERE person_id = ?", [person_id]
        if source:
            sql += " AND source = ?"
            params.append(source)
        return [_document(r) for r in self._db.execute(sql + " ORDER BY published_at DESC", params)]

    def chunks_to_analyze(self, person_id: int, limit: int | None = None) -> list[tuple[Chunk, Document]]:
        """Chunks of first-hand documents that haven't been through claim extraction."""
        sql = f"""
            SELECT c.*, d.id AS d_id FROM chunks c JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunk_analysis a ON a.chunk_id = c.id
            WHERE d.person_id = ? AND a.chunk_id IS NULL AND {ACTIVE_DOC}
            ORDER BY d.published_at DESC, c.document_id, c.seq
            {"LIMIT " + str(int(limit)) if limit else ""}"""
        rows = self._db.execute(sql, (person_id,)).fetchall()
        docs: dict[int, Document] = {}
        out = []
        for r in rows:
            if r["d_id"] not in docs:
                docs[r["d_id"]] = self.get_document(r["d_id"])
            out.append((_chunk(r), docs[r["d_id"]]))
        return out

    def mark_analyzed(self, chunk_ids: Iterable[int], model: str):
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO chunk_analysis (chunk_id, model, analyzed_at) VALUES (?, ?, ?)",
                [(cid, model, _now()) for cid in chunk_ids],
            )

    # ── claims ───────────────────────────────────────────────────

    def add_claims(self, claims: Iterable[Claim]) -> list[int]:
        ids = []
        with self._db:
            for c in claims:
                cur = self._db.execute(
                    "INSERT INTO claims (person_id, document_id, chunk_id, topic, claim, quote, kind,"
                    " sincere, authorship, published_at, url, model, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c.person_id,
                        c.document_id,
                        c.chunk_id,
                        c.topic,
                        c.claim,
                        c.quote,
                        c.kind,
                        int(c.sincere),
                        c.authorship,
                        c.published_at,
                        c.url,
                        c.model,
                        _now(),
                    ),
                )
                c.id = cur.lastrowid
                ids.append(c.id)
        return ids

    def claims(self, person_id: int, topic: str | None = None, sincere_only: bool = False) -> list[Claim]:
        sql = (
            "SELECT c.* FROM claims c JOIN documents d ON d.id = c.document_id"
            f" WHERE c.person_id = ? AND {ACTIVE_DOC}"
        )
        params = [person_id]
        if topic:
            sql += " AND topic LIKE ?"
            params.append(f"%{topic}%")
        if sincere_only:
            sql += " AND sincere = 1"
        return [_claim(r) for r in self._db.execute(sql + " ORDER BY c.topic, c.published_at", params)]

    def get_claim(self, claim_id: int) -> Claim | None:
        row = self._db.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return _claim(row) if row else None

    def topics(self, person_id: int) -> list[tuple[str, int]]:
        return [
            (r["topic"], r["n"])
            for r in self._db.execute(
                "SELECT topic, COUNT(*) n FROM claims c JOIN documents d ON d.id = c.document_id"
                f" WHERE c.person_id = ? AND sincere = 1 AND {ACTIVE_DOC}"
                " GROUP BY topic COLLATE NOCASE ORDER BY n DESC",
                (person_id,),
            )
        ]

    # ── search ───────────────────────────────────────────────────

    def search_claims(self, person_id: int, query: str, limit: int = 20) -> list[Evidence]:
        fts = fts_or_query(query)
        if not fts:
            return []
        rows = self._db.execute(
            f"""
            SELECT c.*, d.source, d.title, bm25(claims_fts) AS score
            FROM claims_fts f JOIN claims c ON c.id = f.rowid JOIN documents d ON d.id = c.document_id
            WHERE claims_fts MATCH ? AND c.person_id = ? AND {ACTIVE_DOC}
            ORDER BY score LIMIT ?""",
            (fts, person_id, limit),
        ).fetchall()
        return [
            Evidence(
                "claim",
                r["id"],
                r["claim"],
                r["quote"],
                r["authorship"],
                r["url"],
                r["published_at"],
                r["source"],
                r["title"],
                r["topic"],
                -r["score"],
            )
            for r in rows
        ]

    def search_chunks(self, person_id: int, query: str, limit: int = 20) -> list[Evidence]:
        fts = fts_or_query(query)
        if not fts:
            return []
        rows = self._db.execute(
            f"""
            SELECT k.*, d.source, d.title, d.url AS d_url, d.published_at, d.authorship,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f JOIN chunks k ON k.id = f.rowid JOIN documents d ON d.id = k.document_id
            WHERE chunks_fts MATCH ? AND d.person_id = ? AND {ACTIVE_DOC}
            ORDER BY score LIMIT ?""",
            (fts, person_id, limit),
        ).fetchall()
        return [
            Evidence(
                "chunk",
                r["id"],
                r["text"],
                r["text"],
                r["authorship"],
                deep_link(r["d_url"], r["start_s"]),
                r["published_at"],
                r["source"],
                r["title"],
                "",
                -r["score"],
            )
            for r in rows
        ]

    # ── embeddings ───────────────────────────────────────────────

    def claims_without_embedding(self, person_id: int) -> list[Claim]:
        return [
            _claim(r)
            for r in self._db.execute(
                "SELECT * FROM claims WHERE person_id = ? AND embedding IS NULL", (person_id,)
            )
        ]

    def set_claim_embeddings(self, pairs: Iterable[tuple[int, np.ndarray]]):
        with self._db:
            self._db.executemany(
                "UPDATE claims SET embedding = ? WHERE id = ?",
                [(np.asarray(v, dtype="<f4").tobytes(), i) for i, v in pairs],
            )

    def claim_vectors(self, person_id: int) -> tuple[list[int], np.ndarray]:
        rows = self._db.execute(
            "SELECT c.id, c.embedding FROM claims c JOIN documents d ON d.id = c.document_id"
            f" WHERE c.person_id = ? AND c.embedding IS NOT NULL AND {ACTIVE_DOC}",
            (person_id,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ([r["id"] for r in rows], np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in rows]))

    def claim_evidence(self, claim_ids: list[int]) -> list[Evidence]:
        if not claim_ids:
            return []
        rows = self._db.execute(
            f"""
            SELECT c.*, d.source, d.title FROM claims c JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({",".join("?" * len(claim_ids))})""",
            claim_ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            Evidence(
                "claim",
                r["id"],
                r["claim"],
                r["quote"],
                r["authorship"],
                r["url"],
                r["published_at"],
                r["source"],
                r["title"],
                r["topic"],
            )
            for r in (by_id[i] for i in claim_ids if i in by_id)
        ]

    def chunk_evidence(self, chunk_ids: list[int]) -> list[Evidence]:
        if not chunk_ids:
            return []
        rows = self._db.execute(
            f"""
            SELECT k.*, d.source, d.title, d.url AS d_url, d.published_at, d.authorship
            FROM chunks k JOIN documents d ON d.id = k.document_id
            WHERE k.id IN ({",".join("?" * len(chunk_ids))}) AND {ACTIVE_DOC}""",
            chunk_ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            Evidence(
                "chunk",
                r["id"],
                r["text"],
                r["text"],
                r["authorship"],
                deep_link(r["d_url"], r["start_s"]),
                r["published_at"],
                r["source"],
                r["title"],
            )
            for r in (by_id[i] for i in chunk_ids if i in by_id)
        ]

    def evidence_by_refs(self, person_id: int, refs: list[str]) -> list[Evidence]:
        """Resolve "C12" (claim) / "K7" (chunk) handles, limited to this person."""
        claim_ids, chunk_ids = [], []
        for ref in refs:
            ref = str(ref).strip().strip("[]").upper()
            if len(ref) > 1 and ref[1:].isdigit():
                (claim_ids if ref[0] == "C" else chunk_ids if ref[0] == "K" else []).append(int(ref[1:]))
        own_claims = {c.id for c in self.claims(person_id)} if claim_ids else set()
        evidence = self.claim_evidence([i for i in claim_ids if i in own_claims])
        for ev in self.chunk_evidence(chunk_ids):
            doc_person = self._db.execute(
                "SELECT d.person_id FROM chunks k JOIN documents d ON d.id = k.document_id WHERE k.id = ?",
                (ev.id,),
            ).fetchone()
            if doc_person and doc_person["person_id"] == person_id:
                evidence.append(ev)
        return evidence

    def insincere_refs(self, evidence: list[Evidence]) -> set[str]:
        """
        Refs ("C12", "K7") of evidence that shouldn't be quoted as a serious
        position: claims flagged insincere, and raw excerpts whose extracted
        claims were *all* flagged insincere (jokes, sarcasm, role-play).
        """
        out: set[str] = set()
        claim_ids = [e.id for e in evidence if e.kind == "claim"]
        chunk_ids = [e.id for e in evidence if e.kind == "chunk"]
        if claim_ids:
            for r in self._db.execute(
                f"SELECT id FROM claims WHERE sincere = 0 AND id IN ({','.join('?' * len(claim_ids))})",
                claim_ids,
            ):
                out.add(f"C{r['id']}")
        if chunk_ids:
            for r in self._db.execute(
                f"SELECT chunk_id FROM claims WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})"
                " GROUP BY chunk_id HAVING MAX(sincere) = 0",
                chunk_ids,
            ):
                out.add(f"K{r['chunk_id']}")
        return out

    # ── stats ────────────────────────────────────────────────────

    def stats(self, person_id: int) -> dict:
        def one(sql: str) -> int:
            return self._db.execute(sql, (person_id,)).fetchone()[0]

        by_source = {
            r["source"]: r["n"]
            for r in self._db.execute(
                "SELECT source, COUNT(*) n FROM documents WHERE person_id = ? GROUP BY source", (person_id,)
            )
        }
        return {
            "documents": sum(by_source.values()),
            "by_source": by_source,
            "chunks": one(
                "SELECT COUNT(*) FROM chunks k JOIN documents d ON d.id = k.document_id WHERE d.person_id = ?"
            ),
            "analyzed": one(
                "SELECT COUNT(*) FROM chunk_analysis a JOIN chunks k ON k.id = a.chunk_id"
                " JOIN documents d ON d.id = k.document_id WHERE d.person_id = ?"
            ),
            "claims": one("SELECT COUNT(*) FROM claims WHERE person_id = ?"),
        }


# ── helpers ──────────────────────────────────────────────────────

_STOPWORDS = set(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "let",
        "like",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "really",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "yeah",
        "okay",
        "ok",
        "um",
        "uh",
        "gonna",
        "wanna",
        "gotta",
        "thing",
        "things",
        "think",
        "know",
        "mean",
        "right",
        "going",
        "get",
        "got",
        "say",
        "said",
    ]
)


def fts_or_query(text: str, max_terms: int = 24) -> str:
    """Turn free text into an FTS5 query that matches ANY of its content words."""
    words = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'+-]*", text.lower()):
        w = w.strip("'-")
        if len(w) >= 3 and w not in _STOPWORDS and w not in words:
            words.append(w)
    return " OR ".join(f'"{w}"' for w in words[:max_terms])


def deep_link(url: str, start_s: float | None) -> str:
    """Link straight to the moment in a video when we know it."""
    if not url or start_s is None:
        return url
    sep = "&" if "?" in url else "?"
    if "youtube.com/watch" in url or "youtu.be/" in url:
        return f"{url}{sep}t={int(start_s)}s"
    if "twitch.tv/videos/" in url:
        h, rem = divmod(int(start_s), 3600)
        m, sec = divmod(rem, 60)
        return f"{url}{sep}t={h}h{m}m{sec}s"
    return url


def _legacy_match(account: Account, row) -> bool:
    """Does a pre-provenance document row come from this account?"""
    handle = account.handle.strip().lstrip("@").lower()
    url = (row["url"] or "").lower()
    if account.platform == "x" and row["source"] == "x":
        return url.startswith((f"https://x.com/{handle}/status/", f"https://twitter.com/{handle}/status/"))
    if account.platform == "wikipedia" and row["source"] == "wikipedia":
        return row["external_id"].replace("_", " ").lower() == account.handle.replace("_", " ").lower()
    if account.platform == "youtube" and row["source"] == "youtube":
        meta = json.loads(row["meta"] or "{}")
        return meta.get("own_channel", True) and str(meta.get("channel_id") or "").lower() == handle
    if account.platform == "website" and row["source"] == "web":
        site = (account.url or account.handle).lower().rstrip("/")
        return bool(site) and row["external_id"].lower().startswith(site)
    return False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _account(r) -> Account:
    return Account(
        id=r["id"],
        person_id=r["person_id"],
        platform=r["platform"],
        handle=r["handle"],
        url=r["url"],
        status=r["status"],
        found_via=r["found_via"],
    )


def _document(r) -> Document:
    return Document(
        id=r["id"],
        person_id=r["person_id"],
        source=r["source"],
        kind=r["kind"],
        external_id=r["external_id"],
        text=r["text"],
        authorship=r["authorship"],
        url=r["url"],
        title=r["title"],
        published_at=r["published_at"],
        meta=json.loads(r["meta"] or "{}"),
        account_id=r["account_id"],
    )


def _chunk(r) -> Chunk:
    return Chunk(
        id=r["id"],
        document_id=r["document_id"],
        seq=r["seq"],
        text=r["text"],
        start_s=r["start_s"],
        end_s=r["end_s"],
    )


def _claim(r) -> Claim:
    return Claim(
        id=r["id"],
        person_id=r["person_id"],
        document_id=r["document_id"],
        chunk_id=r["chunk_id"],
        topic=r["topic"],
        claim=r["claim"],
        quote=r["quote"],
        kind=r["kind"],
        sincere=bool(r["sincere"]),
        authorship=r["authorship"],
        published_at=r["published_at"],
        url=r["url"],
        model=r["model"],
    )
