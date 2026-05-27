"""
Speaker identification: compare live embeddings against enrolled speaker profiles.
"""
import logging
from dataclasses import dataclass

import numpy as np

from kickvoice.config import Config, IdentifyConfig
from kickvoice.storage.profiles import ProfileStore

logger = logging.getLogger(__name__)


@dataclass
class IdentifyResult:
    """Result of a speaker identification attempt."""
    speaker_name: str | None   # None if no match above threshold
    similarity: float          # cosine similarity of best match
    confidence: str            # "high", "medium", "low", "none"
    candidates: list[tuple[str, float]]  # top-N (name, similarity) pairs
    segment_duration_s: float = 0.0

    @property
    def matched(self) -> bool:
        return self.speaker_name is not None

    def __str__(self) -> str:
        if self.matched:
            return (
                f"[{self.confidence.upper()}] {self.speaker_name} "
                f"(sim={self.similarity:.3f}, dur={self.segment_duration_s:.1f}s)"
            )
        return f"[UNKNOWN] best_sim={self.similarity:.3f}"


class SpeakerIdentifier:
    """Compares embeddings against enrolled speaker profiles."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.id_cfg = self.config.identify
        self.store = ProfileStore(config=self.config)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def identify(
        self,
        embedding: np.ndarray,
        segment_duration_s: float = 0.0,
    ) -> IdentifyResult:
        """
        Identify a speaker from their embedding.

        Args:
            embedding: Speaker embedding from the live stream (192,)
            segment_duration_s: Duration of the source audio segment

        Returns:
            IdentifyResult with the best match (or no match)
        """
        all_profiles = self.store.get_all_embeddings()

        if not all_profiles:
            logger.warning("No enrolled speakers — cannot identify")
            return IdentifyResult(
                speaker_name=None,
                similarity=0.0,
                confidence="none",
                candidates=[],
                segment_duration_s=segment_duration_s,
            )

        # Compute similarity against all enrolled speakers
        similarities = []
        for name, profile_emb in all_profiles.items():
            sim = self.cosine_similarity(embedding, profile_emb)
            similarities.append((name, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = similarities[: self.id_cfg.top_n]

        best_name, best_sim = candidates[0]

        # Determine match quality
        if best_sim >= self.id_cfg.high_confidence_threshold:
            confidence = "high"
            speaker_name = best_name
        elif best_sim >= self.id_cfg.match_threshold:
            confidence = "medium"
            speaker_name = best_name
        else:
            confidence = "low"
            speaker_name = None

        result = IdentifyResult(
            speaker_name=speaker_name,
            similarity=best_sim,
            confidence=confidence,
            candidates=candidates,
            segment_duration_s=segment_duration_s,
        )

        if result.matched:
            logger.info(str(result))
        else:
            logger.debug(str(result))

        return result

    def identify_batch(
        self,
        embeddings: list[np.ndarray],
        segment_durations: list[float] | None = None,
    ) -> list[IdentifyResult]:
        """Identify multiple segments in batch."""
        durations = segment_durations or [0.0] * len(embeddings)
        return [
            self.identify(emb, dur)
            for emb, dur in zip(embeddings, durations)
        ]

    def reload_profiles(self):
        """Reload profiles from disk (e.g., after enrolling new speakers)."""
        self.store = ProfileStore(config=self.config)
