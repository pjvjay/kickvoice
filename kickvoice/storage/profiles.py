"""
Speaker profile storage.

Stores speaker embeddings as .npy files with metadata in a JSON index.
"""
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from kickvoice.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """A registered speaker with their voice embedding."""
    name: str
    embedding_file: str  # filename of .npy in profiles_dir
    enrolled_at: str = ""
    source_urls: list[str] = field(default_factory=list)
    n_segments: int = 0
    notes: str = ""
    # Seconds of speech behind the embedding (0 = unknown, e.g. legacy profile).
    speech_seconds: float = 0.0
    # Model that produced the embedding. Embeddings from different models
    # are not comparable ("" = legacy profile, assumed to be the configured model).
    embedding_model: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.enrolled_at:
            self.enrolled_at = datetime.now().isoformat()


class ProfileStore:
    """Manages speaker profiles on disk."""

    INDEX_FILE = "index.json"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.profiles_dir = self.config.profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.profiles_dir / self.INDEX_FILE
        self._profiles: dict[str, SpeakerProfile] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_index()

    def _load_index(self):
        """Load the profile index from disk."""
        if not self._index_path.exists():
            self._profiles = {}
            return

        data = json.loads(self._index_path.read_text())
        known = {f.name for f in fields(SpeakerProfile)}
        for name, pdata in data.items():
            # Ignore unknown keys so newer index files don't crash older code.
            profile = SpeakerProfile(**{k: v for k, v in pdata.items() if k in known})
            self._profiles[name] = profile
            # Load embedding
            emb_path = self.profiles_dir / profile.embedding_file
            if emb_path.exists():
                self._embeddings[name] = np.load(emb_path)
            else:
                logger.warning(
                    f"Embedding file missing for '{name}': {emb_path}"
                )

        logger.info(f"Loaded {len(self._profiles)} speaker profiles")

    def _save_index(self):
        """Persist the index to disk."""
        data = {}
        for name, profile in self._profiles.items():
            d = asdict(profile)
            data[name] = d
        self._index_path.write_text(json.dumps(data, indent=2))

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
        """
        Add or replace a speaker profile.

        Args:
            name: Speaker name (used as identifier)
            embedding: L2-normalized mean embedding (192,)
            source_urls: YouTube URLs used for enrollment
            n_segments: Number of audio segments used
            notes: Optional notes

        Returns:
            The created/updated SpeakerProfile
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        emb_filename = self._embedding_filename(name)
        emb_path = self.profiles_dir / emb_filename
        np.save(emb_path, embedding)

        profile = SpeakerProfile(
            name=name,
            embedding_file=emb_filename,
            source_urls=source_urls or [],
            n_segments=n_segments,
            notes=notes,
            speech_seconds=round(float(speech_seconds), 2),
            embedding_model=embedding_model or self.config.embedding.model_source,
        )
        self._profiles[name] = profile
        self._embeddings[name] = embedding
        self._save_index()

        logger.info(f"Saved profile for '{name}' ({n_segments} segments)")
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
        Fold new audio of an existing speaker into their profile.

        The stored voiceprint becomes a speech-time-weighted average of the old
        and new embeddings, so a short clip nudges a well-established profile
        rather than replacing it. Legacy profiles with unknown speech time are
        weighted equally with the new audio.
        """
        profile = self._profiles.get(name)
        old = self._embeddings.get(name)
        if profile is None or old is None:
            raise KeyError(f"Profile not found: {name}")

        new = np.asarray(embedding, dtype=np.float64)
        new = new / (np.linalg.norm(new) or 1.0)
        old_w = profile.speech_seconds or speech_seconds or 1.0
        new_w = speech_seconds or old_w
        merged = (old * old_w + new * new_w) / (old_w + new_w)
        merged = merged / (np.linalg.norm(merged) or 1.0)

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

        logger.info(
            f"Merged {speech_seconds:.1f}s of new speech into profile '{name}'"
        )
        return profile

    def _embedding_filename(self, name: str) -> str:
        """
        Filename for a profile's .npy. Existing profiles keep their file; new
        ones get a filesystem-safe slug plus a short hash of the exact name so
        e.g. "xQc" and "XQC" (or names with '/') can't clobber each other.
        """
        existing = self._profiles.get(name)
        if existing is not None:
            return existing.embedding_file
        slug = re.sub(r"[^a-z0-9_.-]+", "_", name.lower()).strip("._") or "speaker"
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        return f"{slug[:48]}_{digest}.npy"

    def find_profile(self, name: str) -> SpeakerProfile | None:
        """Exact lookup first, then case-insensitive (for typed names)."""
        if name in self._profiles:
            return self._profiles[name]
        lowered = name.casefold()
        for pname, profile in self._profiles.items():
            if pname.casefold() == lowered:
                return profile
        return None

    def remove_profile(self, name: str) -> bool:
        """Remove a speaker profile."""
        if name not in self._profiles:
            return False

        profile = self._profiles.pop(name)
        self._embeddings.pop(name, None)
        emb_path = self.profiles_dir / profile.embedding_file
        if emb_path.exists():
            emb_path.unlink()
        self._save_index()
        logger.info(f"Removed profile for '{name}'")
        return True

    def get_profile(self, name: str) -> SpeakerProfile | None:
        return self._profiles.get(name)

    def get_embedding(self, name: str) -> np.ndarray | None:
        return self._embeddings.get(name)

    def list_profiles(self) -> list[SpeakerProfile]:
        return list(self._profiles.values())

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return dict of name → embedding for all enrolled speakers."""
        return dict(self._embeddings)

    @property
    def count(self) -> int:
        return len(self._profiles)
