"""
Speaker enrollment: build a voice profile from audio files and/or video URLs.

    source → (download if a URL) → decode to PCM → VAD → embed → average → store

Local files and remote URLs share one path; the only extra step for a URL is
the download.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from kickvoice.audio.download import download_audio
from kickvoice.audio.ffmpeg import decode_to_pcm
from kickvoice.audio.vad import SpeechSegment, get_vad
from kickvoice.config import Config
from kickvoice.speakers.encoder import SpeakerEncoder
from kickvoice.speakers.profiles import ProfileStore, SpeakerProfile

logger = logging.getLogger(__name__)


def is_url(source: str) -> bool:
    return "://" in str(source)


@dataclass
class EnrollResult:
    profile: SpeakerProfile
    n_segments: int
    speech_seconds: float
    sources: list[str]


class Enroller:
    """Create speaker profiles from audio sources."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        encoder: SpeakerEncoder | None = None,
        vad=None,
        store: ProfileStore | None = None,
        downloader: Callable[[str, Path], Path] | None = None,
    ):
        self.config = config or Config()
        self.encoder = encoder if encoder is not None else SpeakerEncoder(config=self.config)
        self.vad = vad if vad is not None else get_vad(
            use_silero=True, audio_cfg=self.config.audio)
        self.store = store if store is not None else ProfileStore(config=self.config)
        # Resolved here (not as a default argument) so the module-level
        # function can be swapped out.
        self.downloader = downloader if downloader is not None else download_audio

    def _segments_from_source(self, source: str | Path, workdir: Path) -> list[SpeechSegment]:
        """Download (if needed), decode and run VAD over one source."""
        if is_url(str(source)):
            # Exactly one download per URL.
            audio_path: str | Path = self.downloader(str(source), workdir)
        else:
            audio_path = source

        pcm = decode_to_pcm(audio_path, self.config.audio)
        logger.info("%s: %.1fs of audio", source,
                    len(pcm) / (self.config.audio.sample_rate * 2))
        segments = self.vad.detect_speech(pcm)
        logger.info("%s: %d speech segment(s)", source, len(segments))
        return segments

    def enroll(
        self,
        name: str,
        sources: Sequence[str | Path],
        notes: str = "",
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> EnrollResult:
        """
        Enroll `name` from any mix of local files and video URLs.

        Using several sources with different moods (calm, excited, shouting)
        makes the profile more robust.
        """
        if not sources:
            raise ValueError("No audio sources given")

        all_segments: list[SpeechSegment] = []
        with tempfile.TemporaryDirectory(prefix="kickvoice_") as tmp:
            for i, source in enumerate(sources):
                if on_progress:
                    on_progress(str(source), i + 1, len(sources))
                all_segments += self._segments_from_source(source, Path(tmp) / f"src_{i}")

        if not all_segments:
            raise ValueError(
                f"No speech detected in {len(sources)} source(s). "
                "Check the audio, or lower the VAD threshold."
            )

        embeddings = self.encoder.embed_segments(all_segments)
        voiceprint = self.encoder.mean_embedding(embeddings)
        speech_seconds = sum(s.duration_s for s in all_segments)

        profile = self.store.add_profile(
            name=name,
            embedding=voiceprint,
            source_urls=[str(s) for s in sources],
            n_segments=len(all_segments),
            notes=notes,
            speech_seconds=speech_seconds,
        )
        logger.info("Enrolled '%s': %d segments, %.1fs of speech",
                    name, len(all_segments), speech_seconds)
        return EnrollResult(profile, len(all_segments), speech_seconds,
                            [str(s) for s in sources])

    # ── convenience wrappers (pre-refactor API) ──────────────────

    def enroll_from_file(self, name: str, audio_path: str | Path, notes: str = "") -> int:
        return self.enroll(name, [audio_path], notes=notes).n_segments

    def enroll_from_youtube(self, name: str, urls: list[str], notes: str = "") -> int:
        return self.enroll(name, list(urls), notes=notes).n_segments
