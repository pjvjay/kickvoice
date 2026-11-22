"""
Speaker profile storage: a JSON index plus one .npy voiceprint per speaker.

    ~/.kickvoice/profiles/
        index.json
        drdisrespect_1a2b3c4d.npy
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from kickvoice.config import Config

logger = logging.getLogger(__name__)


def normalise(embedding: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (cosine similarity then reduces to a dot product)."""
    vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@dataclass
class SpeakerProfile:
    """A registered speaker. The voiceprint itself lives in `embedding_file`."""
    name: str
    embedding_file: str
    enrolled_at: str = ""
    source_urls: list[str] = field(default_factory=list)
    n_segments: int = 0
    notes: str = ""
    # Seconds of speech behind the embedding (0 = unknown, e.g. legacy profile).
    speech_seconds: float = 0.0
    # Model that produced the embedding. Embeddings from different models are
    # not comparable ("" = legacy profile, assumed to be the configured model).
    embedding_model: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.enrolled_at:
            self.enrolled_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        """Tolerant loader: unknown keys are ignored (forward compatible)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class ProfileStore:
    """Create, read, update and delete speaker profiles on disk."""

    INDEX_FILE = "index.json"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.profiles_dir = Path(self.config.profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.profiles_dir / self.INDEX_FILE
        self._profiles: dict[str, SpeakerProfile] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_index()

    # ── persistence ──────────────────────────────────────────────

    def _load_index(self):
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text())
        except json.JSONDecodeError:
            logger.error("Profile index is corrupt: %s", self._index_path)
            raise
        for name, pdata in data.items():
            profile = SpeakerProfile.from_dict(pdata)
            emb_path = self.profiles_dir / profile.embedding_file
            if not emb_path.exists():
                logger.warning("Voiceprint file missing for '%s': %s", name, emb_path)
                continue
            self._profiles[name] = profile
            self._embeddings[name] = np.load(emb_path)
        logger.info("Loaded %d speaker profile(s)", len(self._profiles))

    def _save_index(self):
        data = {name: asdict(p) for name, p in self._profiles.items()}
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self._index_path)      # atomic: never a half-written index

    def _embedding_filename(self, name: str) -> str:
        """
        Existing profiles keep their file; new ones get a filesystem-safe slug
        plus a short hash of the exact name, so "xQc" and "XQC" (or names with
        a '/') can't overwrite each other.
        """
        existing = self._profiles.get(name)
        if existing is not None:
            return existing.embedding_file
        slug = re.sub(r"[^a-z0-9_.-]+", "_", name.lower()).strip("._") or "speaker"
        return f"{slug[:48]}_{hashlib.sha1(name.encode()).hexdigest()[:8]}.npy"

    # ── writes ───────────────────────────────────────────────────

    def add_profile(
        self,
        name: str,
        embedding: np.ndarray,
        source_urls: list[str] | None = None,
        n_segments: int = 0,
        notes: str = "",
        speech_seconds: float = 0.0,
        embedding_model: str | None = None,
    ) -> SpeakerProfile:
        """Add a speaker, or replace an existing profile of the same name."""
        embedding = normalise(embedding)
        filename = self._embedding_filename(name)
        np.save(self.profiles_dir / filename, embedding)

        profile = SpeakerProfile(
            name=name,
            embedding_file=filename,
            source_urls=list(source_urls or []),
            n_segments=int(n_segments),
            notes=notes,
            speech_seconds=round(float(speech_seconds), 2),
            embedding_model=embedding_model or self.config.embedding.model_source,
        )
        self._profiles[name] = profile
        self._embeddings[name] = embedding
        self._save_index()
        logger.info("Saved profile '%s' (%d segments)", name, n_segments)
        return profile

    def merge_into_profile(
        self,
        name: str,
        embedding: np.ndarray,
        speech_seconds: float = 0.0,
        n_segments: int = 0,
        source_urls: list[str] | None = None,
        note: str = "",
    ) -> SpeakerProfile:
        """
        Fold new audio of a known speaker into their profile.

        The voiceprint becomes a speech-time-weighted average, so a short clip
        nudges a well-established profile instead of replacing it. Legacy
        profiles with unknown speech time are weighted equally with the new
        audio.
        """
        profile = self._profiles.get(name)
        if profile is None:
            raise KeyError(f"Profile not found: {name}")

        old = self._embeddings[name]
        new = normalise(embedding)
        old_weight = profile.speech_seconds or speech_seconds or 1.0
        new_weight = speech_seconds or old_weight
        merged = normalise(old * old_weight + new * new_weight)

        np.save(self.profiles_dir / profile.embedding_file, merged)
        self._embeddings[name] = merged

        profile.n_segments += int(n_segments)
        if profile.speech_seconds:
            profile.speech_seconds = round(profile.speech_seconds + float(speech_seconds), 2)
        for url in source_urls or []:
            if url not in profile.source_urls:
                profile.source_urls.append(url)
        if note:
            profile.notes = f"{profile.notes}\n{note}".strip() if profile.notes else note
        profile.updated_at = datetime.now().isoformat()
        self._save_index()

        logger.info("Merged %.1fs of speech into profile '%s'", speech_seconds, name)
        return profile

    def remove_profile(self, name: str) -> bool:
        profile = self._profiles.pop(name, None)
        if profile is None:
            return False
        self._embeddings.pop(name, None)
        (self.profiles_dir / profile.embedding_file).unlink(missing_ok=True)
        self._save_index()
        logger.info("Removed profile '%s'", name)
        return True

    # ── reads ────────────────────────────────────────────────────

    def get_profile(self, name: str) -> SpeakerProfile | None:
        return self._profiles.get(name)

    def find_profile(self, name: str) -> SpeakerProfile | None:
        """Exact lookup first, then case-insensitive (for user-typed names)."""
        if name in self._profiles:
            return self._profiles[name]
        lowered = name.casefold()
        return next((p for n, p in self._profiles.items() if n.casefold() == lowered), None)

    def get_embedding(self, name: str) -> np.ndarray | None:
        return self._embeddings.get(name)

    def list_profiles(self) -> list[SpeakerProfile]:
        return list(self._profiles.values())

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        return dict(self._embeddings)

    def embedding_matrix(self) -> tuple[list[str], np.ndarray]:
        """(names, matrix) of normalised voiceprints for bulk comparison."""
        names = list(self._embeddings)
        if not names:
            return [], np.zeros((0, self.config.embedding.embedding_dim))
        return names, np.stack([normalise(self._embeddings[n]) for n in names])

    @property
    def count(self) -> int:
        return len(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)
