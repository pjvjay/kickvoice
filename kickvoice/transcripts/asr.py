"""
Speech-to-text.

The pipeline only depends on the small `SpeechRecognizer` interface below
(audio in, timestamped words out), so engines are swappable. The default is
faster-whisper (Whisper via CTranslate2: CPU-friendly, word timestamps).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from kickvoice.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """One recognised word, timed relative to the audio it came from."""
    start: float          # seconds
    end: float            # seconds
    text: str             # includes Whisper's leading space, e.g. " hello"
    probability: float = 1.0

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


class SpeechRecognizer(Protocol):
    """What the transcription pipeline needs from an ASR engine."""

    @property
    def model_name(self) -> str: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> list[Word]:
        """`audio` is mono float32 in [-1, 1]. Returns words in time order."""
        ...

    # Optional: `load()` — called before capture starts, so a missing model
    # fails fast (and a first-time download doesn't let a live stream run ahead).


class ASRUnavailable(RuntimeError):
    pass


class FasterWhisperRecognizer:
    """Whisper via faster-whisper. Model downloads on first use."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.cfg = self.config.transcribe
        self._model = None

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self.cfg.asr_model}"

    def load(self):
        """Load (downloading on first use) the Whisper model now."""
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRUnavailable(
                "Transcription needs faster-whisper: pip install -e '.[transcribe]' "
                "(or: pip install faster-whisper)"
            ) from e
        logger.info("Loading Whisper model '%s' (%s, %s)",
                    self.cfg.asr_model, self.cfg.asr_device, self.cfg.asr_compute_type)
        try:
            self._model = WhisperModel(
                self.cfg.asr_model,
                device=self.cfg.asr_device,
                compute_type=self.cfg.asr_compute_type,
                download_root=str(Path(self.config.data_dir) / "pretrained_models" / "whisper"),
            )
        except Exception as e:   # network, disk, CUDA… — all fatal, all need a clear message
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            raise ASRUnavailable(
                f"Couldn't load Whisper model '{self.cfg.asr_model}': {first_line}\n"
                "The model downloads on first use (needs internet), or pass a local "
                "model directory with --model."
            ) from e

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> list[Word]:
        if sample_rate != 16000:
            raise ValueError("Whisper expects 16kHz audio")
        if len(audio) == 0:
            return []
        self._load_model()
        segments, _info = self._model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            word_timestamps=True,
            # We run our own VAD and drop words outside speech, and we don't
            # carry text between blocks: both curb Whisper's tendency to
            # hallucinate over music/silence or repeat itself.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        for segment in segments:
            for w in segment.words or []:
                if w.word.strip():
                    words.append(Word(float(w.start), float(w.end), w.word,
                                      float(w.probability)))
        return words
