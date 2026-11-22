"""Compare a voice embedding against enrolled speaker profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kickvoice.config import Config
from kickvoice.speakers.profiles import ProfileStore, normalise

logger = logging.getLogger(__name__)


@dataclass
class IdentifyResult:
    """Outcome of one identification attempt."""

    speaker_name: str | None  # None when nothing passed the threshold
    similarity: float  # cosine similarity of the best candidate
    confidence: str  # "high" | "medium" | "low" | "none"
    candidates: list[tuple[str, float]]  # top-N (name, similarity), best first
    segment_duration_s: float = 0.0

    @property
    def matched(self) -> bool:
        return self.speaker_name is not None

    @property
    def best(self) -> tuple[str, float] | None:
        """Closest profile, even if it didn't pass the threshold."""
        return self.candidates[0] if self.candidates else None

    def __str__(self) -> str:
        if self.matched:
            return (
                f"[{self.confidence.upper()}] {self.speaker_name} "
                f"(sim={self.similarity:.3f}, dur={self.segment_duration_s:.1f}s)"
            )
        return f"[UNKNOWN] best_sim={self.similarity:.3f}"


class SpeakerIdentifier:
    """Cosine-similarity matching against the profile store."""

    def __init__(self, config: Config | None = None, store: ProfileStore | None = None):
        self.config = config or Config()
        self.id_cfg = self.config.identify
        # Pass a shared store to see profiles added/updated during a session.
        self.store = store if store is not None else ProfileStore(config=self.config)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(normalise(a) @ normalise(b))

    def identify(self, embedding: np.ndarray, segment_duration_s: float = 0.0) -> IdentifyResult:
        """Find the best-matching profile for one embedding."""
        names, matrix = self.store.embedding_matrix()
        if not names:
            logger.debug("No enrolled speakers — cannot identify")
            return IdentifyResult(None, 0.0, "none", [], segment_duration_s)

        sims = matrix @ normalise(embedding)
        order = np.argsort(-sims)[: max(1, self.id_cfg.top_n)]
        candidates = [(names[i], float(sims[i])) for i in order]
        best_name, best_sim = candidates[0]

        if best_sim >= self.id_cfg.high_confidence_threshold:
            confidence, speaker = "high", best_name
        elif best_sim >= self.id_cfg.match_threshold:
            confidence, speaker = "medium", best_name
        else:
            confidence, speaker = "low", None

        result = IdentifyResult(speaker, best_sim, confidence, candidates, segment_duration_s)
        logger.log(logging.INFO if result.matched else logging.DEBUG, str(result))
        return result

    def identify_batch(
        self,
        embeddings: list[np.ndarray],
        segment_durations: list[float] | None = None,
    ) -> list[IdentifyResult]:
        durations = segment_durations or [0.0] * len(embeddings)
        return [self.identify(e, d) for e, d in zip(embeddings, durations, strict=True)]

    def reload_profiles(self):
        """Re-read profiles from disk (e.g. after enrolling new speakers)."""
        self.store = ProfileStore(config=self.config)
