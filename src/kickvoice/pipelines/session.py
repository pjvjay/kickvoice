"""
Extraction sessions: the on-disk hand-off between discovery and approval.

    ~/.kickvoice/extractions/<session-id>/
        session.json       metadata + one entry per discovered speaker
        speaker_1.npy      candidate voiceprint (L2-normalised)
        speaker_1.wav      review clip (most typical ~20s of that voice)

Nothing becomes a profile until the user approves it, so a capture can be
reviewed right away, later (`kickvoice review`), or partially and resumed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from kickvoice.audio.ffmpeg import write_wav

PENDING = "pending"
CREATED = "created"  # became a new profile
MERGED = "merged"  # folded into an existing profile
SKIPPED = "skipped"  # user rejected


@dataclass
class Candidate:
    id: str  # "speaker_1", ... (1 = most speech)
    speech_seconds: float
    n_windows: int
    first_heard_s: float  # offset from start of capture
    cohesion: float  # how tight the cluster is (0..1)
    embedding_file: str
    sample_file: str
    # Closest known profile at extraction time (informational — review
    # re-matches against the live profile store).
    best_match: str | None = None
    best_similarity: float = 0.0
    matched: bool = False  # best_similarity >= match threshold
    status: str = PENDING
    profile_name: str | None = None
    reviewed_at: str | None = None


@dataclass
class ExtractionSession:
    id: str
    source: str  # stream URL or file path
    created_at: str
    embedding_model: str
    captured_seconds: float = 0.0
    speech_seconds: float = 0.0
    settings: dict = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    path: Path | None = field(default=None, repr=False, compare=False)

    FILE = "session.json"

    # ── construction / persistence ──────────────────────────────

    @classmethod
    def create(cls, root: Path, source: str, embedding_model: str) -> ExtractionSession:
        now = datetime.now()
        sid = f"{now:%Y%m%d-%H%M%S}_{source_slug(source)}"
        path = Path(root) / sid
        n = 2
        while path.exists():
            path = Path(root) / f"{sid}-{n}"
            n += 1
        path.mkdir(parents=True)
        return cls(
            id=path.name,
            source=source,
            created_at=now.isoformat(),
            embedding_model=embedding_model,
            path=path,
        )

    @classmethod
    def load(cls, path: Path) -> ExtractionSession:
        path = Path(path)
        data = json.loads((path / cls.FILE).read_text())
        known = {f.name for f in fields(Candidate)}
        cands = [Candidate(**{k: v for k, v in c.items() if k in known}) for c in data.pop("candidates", [])]
        own = {f.name for f in fields(cls)} - {"candidates", "path"}
        return cls(**{k: v for k, v in data.items() if k in own}, candidates=cands, path=path)

    def save(self):
        data = asdict(self)
        data.pop("path")
        tmp = self.path / (self.FILE + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path / self.FILE)  # atomic: safe to Ctrl+C

    # ── candidate helpers ────────────────────────────────────────

    def add_candidate(
        self,
        embedding: np.ndarray,
        sample_pcm: bytes,
        sample_rate: int,
        **meta,
    ) -> Candidate:
        cid = f"speaker_{len(self.candidates) + 1}"
        np.save(self.path / f"{cid}.npy", np.asarray(embedding))
        write_wav(self.path / f"{cid}.wav", sample_pcm, sample_rate)
        cand = Candidate(id=cid, embedding_file=f"{cid}.npy", sample_file=f"{cid}.wav", **meta)
        self.candidates.append(cand)
        return cand

    def embedding(self, cand: Candidate) -> np.ndarray:
        return np.load(self.path / cand.embedding_file)

    def sample_path(self, cand: Candidate) -> Path:
        return self.path / cand.sample_file

    @property
    def pending(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status == PENDING]


def list_sessions(root: Path) -> list[ExtractionSession]:
    """All sessions under `root`, newest first. Corrupt dirs are skipped."""
    root = Path(root)
    if not root.exists():
        return []
    sessions = []
    for d in root.iterdir():
        if (d / ExtractionSession.FILE).exists():
            try:
                sessions.append(ExtractionSession.load(d))
            except (ValueError, TypeError, KeyError):
                continue
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions


def find_session(root: Path, ref: str | None) -> ExtractionSession | None:
    """
    Resolve a session by path, id, or unique id prefix. With no ref, return
    the newest session that still has pending candidates.
    """
    if ref:
        p = Path(ref).expanduser()
        if (p / ExtractionSession.FILE).exists():
            return ExtractionSession.load(p)
        matches = [s for s in list_sessions(root) if s.id.startswith(ref)]
        return matches[0] if len(matches) == 1 else None
    for s in list_sessions(root):
        if s.pending:
            return s
    return None


def source_slug(source: str) -> str:
    """'https://kick.com/SomeChannel?x=1' -> 'somechannel'; files -> stem."""
    name = channel_name(source) or Path(source.rstrip("/")).stem or "capture"
    return re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")[:40] or "capture"


def channel_name(source: str) -> str | None:
    """Channel slug from a kick.com URL (the likely main speaker), else None."""
    m = re.match(r"^(?:https?://)?(?:www\.)?kick\.com/([^/?#]+)", source.strip(), re.I)
    if m and m.group(1).lower() not in {"video", "videos", "clips", "categories"}:
        return m.group(1)
    return None
