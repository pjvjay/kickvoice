"""
Voice Activity Detection: keep speech, drop silence, music and game audio.

Silero VAD is used when torch is available, with a dependency-free
energy-based fallback.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

import numpy as np

from kickvoice.audio.ffmpeg import pcm_to_numpy, pcm_to_tensor
from kickvoice.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech region within an audio chunk."""

    start_sample: int
    end_sample: int
    pcm_bytes: bytes
    confidence: float = 0.0
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        return len(self.pcm_bytes) / (self.sample_rate * 2)

    def to_numpy(self) -> np.ndarray:
        return pcm_to_numpy(self.pcm_bytes)

    def to_float_tensor(self):
        return pcm_to_tensor(self.pcm_bytes)


class BaseVAD:
    def __init__(self, audio_cfg: AudioConfig | None = None):
        self.audio_cfg = audio_cfg or AudioConfig()

    def detect_speech(
        self,
        pcm_bytes: bytes,
        threshold: float | None = None,
        min_duration_s: float | None = None,
    ) -> list[SpeechSegment]:
        raise NotImplementedError

    def _segment(self, pcm_bytes: bytes, start: int, end: int, confidence: float) -> SpeechSegment:
        return SpeechSegment(
            start_sample=start,
            end_sample=end,
            pcm_bytes=pcm_bytes[start * 2 : end * 2],
            confidence=confidence,
            sample_rate=self.audio_cfg.sample_rate,
        )


class SileroVAD(BaseVAD):
    """
    Silero VAD wrapper.

    Prefers the `silero-vad` package (weights ship in the wheel); falls back to
    torch.hub, which downloads from GitHub on first use.
    """

    def __init__(self, audio_cfg: AudioConfig | None = None):
        super().__init__(audio_cfg)
        self._model = None
        self._get_speech_timestamps = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            logger.info("Loading Silero VAD (silero-vad package)")
            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        except ImportError:
            import torch

            logger.info("Loading Silero VAD (torch.hub)")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
        logger.info("Silero VAD loaded")

    def detect_speech(self, pcm_bytes, threshold=None, min_duration_s=None):
        import torch

        self._load_model()
        threshold = threshold if threshold is not None else self.audio_cfg.vad_threshold
        min_duration_s = (
            min_duration_s if min_duration_s is not None else self.audio_cfg.min_speech_duration_s
        )

        audio = torch.from_numpy(pcm_to_numpy(pcm_bytes))
        with torch.no_grad():
            timestamps = self._get_speech_timestamps(
                audio,
                self._model,
                sampling_rate=self.audio_cfg.sample_rate,
                threshold=threshold,
                min_speech_duration_ms=int(min_duration_s * 1000),
                return_seconds=False,
            )

        segments = [
            self._segment(pcm_bytes, ts["start"], ts["end"], threshold)
            for ts in timestamps
            if (ts["end"] - ts["start"]) / self.audio_cfg.sample_rate >= min_duration_s
        ]
        logger.debug("Silero VAD: %d speech segment(s) from %d detection(s)", len(segments), len(timestamps))
        return segments


class EnergyVAD(BaseVAD):
    """Simple RMS-energy VAD fallback. No ML dependencies."""

    DEFAULT_THRESHOLD = 0.02

    def detect_speech(self, pcm_bytes, threshold=None, min_duration_s=None):
        threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        min_duration_s = (
            min_duration_s if min_duration_s is not None else self.audio_cfg.min_speech_duration_s
        )

        samples = pcm_to_numpy(pcm_bytes)
        frame = int(self.audio_cfg.sample_rate * 0.03)  # 30ms frames
        n_frames = len(samples) // frame
        if n_frames == 0:
            return []

        frames = samples[: n_frames * frame].reshape(n_frames, frame)
        loud = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) > threshold

        segments = []
        padded = np.concatenate(([False], loud, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start_frame, end_frame in zip(edges[::2], edges[1::2], strict=True):
            if (end_frame - start_frame) * 0.03 < min_duration_s:
                continue
            start = int(start_frame * frame)
            end = min(int(end_frame * frame), len(samples))
            segments.append(self._segment(pcm_bytes, start, end, 0.5))

        logger.debug("EnergyVAD: %d speech segment(s)", len(segments))
        return segments


def get_vad(use_silero: bool = True, audio_cfg: AudioConfig | None = None) -> BaseVAD:
    """Silero VAD when torch is available, else the energy-based fallback."""
    if use_silero:
        if importlib.util.find_spec("torch") is not None:
            return SileroVAD(audio_cfg)
        logger.warning("torch not available — falling back to EnergyVAD")
    return EnergyVAD(audio_cfg)
