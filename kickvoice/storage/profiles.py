"""
Speaker profile storage.

Stores speaker embeddings as .npy files with metadata in a JSON index.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
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
        for name, pdata in data.items():
            profile = SpeakerProfile(**pdata)
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
    ) -> SpeakerProfile:
        """
        Add or update a speaker profile.

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

        emb_filename = f"{name.lower().replace(' ', '_')}.npy"
        emb_path = self.profiles_dir / emb_filename
        np.save(emb_path, embedding)

        profile = SpeakerProfile(
            name=name,
            embedding_file=emb_filename,
            source_urls=source_urls or [],
            n_segments=n_segments,
            notes=notes,
        )
        self._profiles[name] = profile
        self._embeddings[name] = embedding
        self._save_index()

        logger.info(f"Saved profile for '{name}' ({n_segments} segments)")
        return profile

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
