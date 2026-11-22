"""
Speaker-attributed transcript storage (SQLite, stdlib only).

    ~/.kickvoice/transcripts.db
        sessions      one row per transcribed stream/file
        utterances    one row per speaker turn: who, when, what, how sure
        utterances_fts  full-text index over utterance text (if FTS5 exists)

This is the hand-off point for later analysis: everything downstream reads
utterances from here, filtered by speaker and attribution level.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from kickvoice.transcripts.attribution import CONFIDENT, PROBABLE, UNKNOWN

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Which attribution levels each `min_attribution` filter lets through.
_LEVELS_AT_LEAST = {
    CONFIDENT: (CONFIDENT,),
    PROBABLE: (CONFIDENT, PROBABLE),
    UNKNOWN: (CONFIDENT, PROBABLE, UNKNOWN),
}


@dataclass
class TranscriptSession:
    id: str
    source: str
    started_at: str
    asr_model: str = ""
    embedding_model: str = ""
    ended_at: str | None = None
    captured_seconds: float = 0.0
    settings: dict = field(default_factory=dict)
    # Filled by list queries:
    n_utterances: int = 0
    speakers: list[str] = field(default_factory=list)


@dataclass
class Utterance:
    session_id: str
    start_s: float                # offset from the start of the capture
    end_s: float
    text: str
    speaker: str | None = None    # profile name, None = unrecognised voice
    similarity: float = 0.0
    attribution: str = UNKNOWN    # confident / probable / unknown
    asr_confidence: float = 0.0
    contested: bool = False
    # Which distinct voice in the session this is (1 = most talkative).
    # Lets analysis tell two unrecognised people apart.
    voice_group: int | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)
    id: int | None = None

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def wall_clock(self, session: TranscriptSession) -> datetime:
        """When this was said (meaningful for live streams)."""
        return datetime.fromisoformat(session.started_at) + timedelta(seconds=self.start_s)

    def to_dict(self, session: TranscriptSession | None = None) -> dict:
        data = {
            "id": self.id,
            "session_id": self.session_id,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "speaker": self.speaker,
            "voice_group": self.voice_group,
            "attribution": self.attribution,
            "similarity": round(self.similarity, 4),
            "asr_confidence": round(self.asr_confidence, 4),
            "text": self.text,
        }
        if session is not None:
            data["source"] = session.source
            data["said_at"] = self.wall_clock(session).isoformat(timespec="seconds")
        return data


class TranscriptStore:
    """Read/write speaker-attributed transcripts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")   # readers don't block the writer
        self.has_fts = False
        self._create_schema()

    # ── lifecycle ────────────────────────────────────────────────

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _create_schema(self):
        with self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS sessions (
                    id               TEXT PRIMARY KEY,
                    source           TEXT NOT NULL,
                    started_at       TEXT NOT NULL,
                    ended_at         TEXT,
                    captured_seconds REAL NOT NULL DEFAULT 0,
                    asr_model        TEXT,
                    embedding_model  TEXT,
                    settings         TEXT
                );
                CREATE TABLE IF NOT EXISTS utterances (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    start_s        REAL NOT NULL,
                    end_s          REAL NOT NULL,
                    speaker        TEXT,
                    similarity     REAL NOT NULL DEFAULT 0,
                    attribution    TEXT NOT NULL,
                    asr_confidence REAL NOT NULL DEFAULT 0,
                    contested      INTEGER NOT NULL DEFAULT 0,
                    voice_group    INTEGER,
                    text           TEXT NOT NULL,
                    embedding      BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id, start_s);
                CREATE INDEX IF NOT EXISTS idx_utt_speaker ON utterances(speaker, attribution);
            """)
            self._db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                             (str(SCHEMA_VERSION),))
        try:
            with self._db:
                self._db.executescript("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts
                        USING fts5(text, content='utterances', content_rowid='id');
                    CREATE TRIGGER IF NOT EXISTS utt_ai AFTER INSERT ON utterances BEGIN
                        INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS utt_ad AFTER DELETE ON utterances BEGIN
                        INSERT INTO utterances_fts(utterances_fts, rowid, text)
                            VALUES ('delete', old.id, old.text);
                    END;
                """)
            self.has_fts = True
        except sqlite3.OperationalError:
            logger.info("SQLite FTS5 unavailable — search falls back to LIKE")

    # ── sessions ─────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        source: str,
        asr_model: str = "",
        embedding_model: str = "",
        settings: dict | None = None,
        started_at: datetime | None = None,
    ) -> TranscriptSession:
        session = TranscriptSession(
            id=self._unique_session_id(session_id),
            source=source,
            started_at=(started_at or datetime.now()).isoformat(timespec="seconds"),
            asr_model=asr_model,
            embedding_model=embedding_model,
            settings=settings or {},
        )
        with self._db:
            self._db.execute(
                "INSERT INTO sessions (id, source, started_at, asr_model, embedding_model, settings)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session.id, session.source, session.started_at, asr_model,
                 embedding_model, json.dumps(session.settings)),
            )
        return session

    def _unique_session_id(self, base: str) -> str:
        candidate, n = base, 2
        while self._db.execute("SELECT 1 FROM sessions WHERE id = ?", (candidate,)).fetchone():
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def finish_session(self, session_id: str, captured_seconds: float):
        with self._db:
            self._db.execute(
                "UPDATE sessions SET ended_at = ?, captured_seconds = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), captured_seconds, session_id),
            )

    def get_session(self, ref: str) -> TranscriptSession | None:
        """Exact id, or a unique id prefix."""
        rows = self._db.execute(
            "SELECT * FROM sessions WHERE substr(id, 1, length(?)) = ? ORDER BY id",
            (ref, ref),
        ).fetchall()
        exact = [r for r in rows if r["id"] == ref]
        rows = exact or rows
        return self._session(rows[0]) if len(rows) == 1 else None

    def list_sessions(self) -> list[TranscriptSession]:
        sessions = []
        for row in self._db.execute("SELECT * FROM sessions ORDER BY started_at DESC, id DESC"):
            session = self._session(row)
            stats = self._db.execute(
                "SELECT COUNT(*) n, GROUP_CONCAT(DISTINCT speaker) speakers "
                "FROM utterances WHERE session_id = ?", (session.id,)).fetchone()
            session.n_utterances = stats["n"]
            session.speakers = sorted((stats["speakers"] or "").split(",")) if stats["speakers"] else []
            sessions.append(session)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0

    @staticmethod
    def _session(row: sqlite3.Row) -> TranscriptSession:
        return TranscriptSession(
            id=row["id"], source=row["source"], started_at=row["started_at"],
            ended_at=row["ended_at"], captured_seconds=row["captured_seconds"],
            asr_model=row["asr_model"] or "", embedding_model=row["embedding_model"] or "",
            settings=json.loads(row["settings"] or "{}"),
        )

    # ── utterances ───────────────────────────────────────────────

    def add_utterances(self, utterances: Iterable[Utterance]) -> list[int]:
        ids = []
        with self._db:
            for u in utterances:
                cur = self._db.execute(
                    "INSERT INTO utterances (session_id, start_s, end_s, speaker, similarity,"
                    " attribution, asr_confidence, contested, voice_group, text, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (u.session_id, u.start_s, u.end_s, u.speaker, u.similarity,
                     u.attribution, u.asr_confidence, int(u.contested), u.voice_group,
                     u.text, _to_blob(u.embedding)),
                )
                u.id = cur.lastrowid
                ids.append(u.id)
        return ids

    def utterances(
        self,
        session_id: str | None = None,
        speaker: str | None = None,
        min_attribution: str = UNKNOWN,
        limit: int | None = None,
    ) -> list[Utterance]:
        """Utterances in time order, filtered by session/speaker/trust level."""
        where, params = self._filters(session_id, speaker, min_attribution)
        sql = f"SELECT * FROM utterances {where} ORDER BY session_id, start_s"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [_utterance(r) for r in self._db.execute(sql, params)]

    def search(
        self,
        query: str,
        speaker: str | None = None,
        min_attribution: str = UNKNOWN,
        limit: int = 50,
    ) -> list[Utterance]:
        """Full-text search (FTS5 query syntax when available)."""
        where, params = self._filters(None, speaker, min_attribution, prefix="u.")
        if self.has_fts:
            where = (where + " AND" if where else "WHERE") + " utterances_fts MATCH ?"
            sql = (f"SELECT u.* FROM utterances u JOIN utterances_fts f ON f.rowid = u.id "
                   f"{where} ORDER BY rank LIMIT ?")
            params += [_fts_query(query), limit]
        else:
            where = (where + " AND" if where else "WHERE") + " u.text LIKE ?"
            sql = f"SELECT u.* FROM utterances u {where} ORDER BY u.session_id, u.start_s LIMIT ?"
            params += [f"%{query}%", limit]
        return [_utterance(r) for r in self._db.execute(sql, params)]

    def speaker_summary(self) -> list[dict]:
        """How much speech is on record per speaker and trust level."""
        rows = self._db.execute("""
            SELECT speaker,
                   COUNT(*) AS utterances,
                   SUM(end_s - start_s) AS seconds,
                   SUM(CASE WHEN attribution = 'confident' THEN end_s - start_s ELSE 0 END)
                       AS confident_seconds,
                   COUNT(DISTINCT session_id) AS sessions
            FROM utterances GROUP BY speaker ORDER BY confident_seconds DESC, seconds DESC
        """)
        return [dict(r) for r in rows]

    def delete_utterances(self, ids: Iterable[int]):
        with self._db:
            self._db.executemany("DELETE FROM utterances WHERE id = ?", [(i,) for i in ids])

    def update_attributions(
        self, updates: Iterable[tuple[int, str | None, float, str, int | None]],
    ):
        """Bulk-set (id, speaker, similarity, attribution, voice_group)."""
        with self._db:
            self._db.executemany(
                "UPDATE utterances SET speaker = ?, similarity = ?, attribution = ?,"
                " voice_group = ? WHERE id = ?",
                [(speaker, sim, level, group, uid) for uid, speaker, sim, level, group in updates])

    def iter_sessions_with_utterances(
        self, speaker: str | None = None, min_attribution: str = UNKNOWN,
    ) -> Iterator[tuple[TranscriptSession, list[Utterance]]]:
        for session in self.list_sessions()[::-1]:          # oldest first
            utts = self.utterances(session.id, speaker, min_attribution)
            if utts:
                yield session, utts

    @staticmethod
    def _filters(session_id, speaker, min_attribution, prefix=""):
        if min_attribution not in _LEVELS_AT_LEAST:
            raise ValueError(f"min_attribution must be one of {tuple(_LEVELS_AT_LEAST)}")
        clauses, params = [], []
        if session_id:
            clauses.append(f"{prefix}session_id = ?")
            params.append(session_id)
        if speaker:
            clauses.append(f"{prefix}speaker = ? COLLATE NOCASE")
            params.append(speaker)
        levels = _LEVELS_AT_LEAST[min_attribution]
        if len(levels) < 3:
            clauses.append(f"{prefix}attribution IN ({','.join('?' * len(levels))})")
            params += list(levels)
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def _to_blob(embedding: np.ndarray | None) -> bytes | None:
    if embedding is None:
        return None
    return np.asarray(embedding, dtype="<f4").reshape(-1).tobytes()


def _utterance(row: sqlite3.Row) -> Utterance:
    blob = row["embedding"]
    return Utterance(
        id=row["id"], session_id=row["session_id"], start_s=row["start_s"], end_s=row["end_s"],
        speaker=row["speaker"], similarity=row["similarity"], attribution=row["attribution"],
        asr_confidence=row["asr_confidence"], contested=bool(row["contested"]),
        voice_group=row["voice_group"], text=row["text"],
        embedding=np.frombuffer(blob, dtype="<f4").copy() if blob else None,
    )


def _fts_query(query: str) -> str:
    """
    Make plain user input safe for FTS5: each word becomes a quoted term, so
    punctuation like "don't" or "c++" can't cause syntax errors.
    """
    terms = [t.replace('"', '""') for t in query.split() if t.strip()]
    return " ".join(f'"{t}"' for t in terms) or '""'
