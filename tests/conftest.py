"""Shared fixtures: isolated config dirs + model-free fakes for the pipeline."""
import math
import struct

import numpy as np
import pytest

from kickvoice.config import Config
from kickvoice.speakers.encoder import SpeakerEncoder

SR = 16000
DIM = 192


@pytest.fixture
def config(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", profiles_dir=tmp_path / "data" / "profiles")
    cfg.extract.min_speaker_speech_s = 8.0
    cfg.extract.chunk_duration_s = 10.0
    return cfg


def voice_vector(freq: float) -> np.ndarray:
    """Stable pseudo-'voiceprint' for a synthetic voice (tone frequency)."""
    rng = np.random.default_rng(int(freq))
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def tone(freq: float, seconds: float, amp: float = 0.3) -> bytes:
    n = int(SR * seconds)
    return struct.pack(
        f"<{n}h", *(int(amp * 32767 * math.sin(2 * math.pi * freq * i / SR)) for i in range(n))
    )


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SR * seconds)


def conversation(turns: list[tuple[float, float]], gap: float = 0.6) -> bytes:
    """[(freq, seconds), ...] → PCM with silence between turns."""
    out = [silence(gap)]
    for freq, secs in turns:
        out += [tone(freq, secs), silence(gap)]
    return b"".join(out)


class FakeEncoder(SpeakerEncoder):
    """
    Real encoder API, no model: maps a window to its dominant frequency's
    voiceprint (+ a little noise). Subclassing keeps it honest — if the
    encoder API changes, the fake changes with it.
    """

    def __init__(self, noise: float = 0.15, seed: int = 0, voices: dict | None = None):
        super().__init__()
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.voices = voices or {}          # freq → custom voiceprint (sound-alikes)

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        spec = np.abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]
        bucket = round(freq / 10) * 10          # tolerate FFT bin jitter
        base = self.voices.get(bucket)
        v = (voice_vector(bucket) if base is None else base) + self.rng.normal(scale=self.noise / math.sqrt(DIM), size=DIM)
        return v / np.linalg.norm(v)


class FakeCapture:
    """Serves PCM in blocks like StreamCapture/FileCapture, tracking closure."""

    def __init__(self, pcm: bytes):
        self.pcm = pcm
        self.closed = False
        self.requested = None

    def capture_chunks(self, chunk_duration_s=None, chunk_stride_s=None):
        self.requested = (chunk_duration_s, chunk_stride_s)
        size = int(SR * 2 * chunk_duration_s)
        try:
            for off in range(0, len(self.pcm), size):
                yield self.pcm[off:off + size]
        finally:
            self.closed = True


class ScriptedInput:
    """Feeds canned answers to Reviewer prompts; records what was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class FakeRecognizer:
    """
    ASR stand-in with Whisper's output shape: ~0.5s "words" over loud audio,
    each named after the dominant pitch ("w220"), so tests can check which
    voice every word was attributed to. `hallucinate=True` also emits a word
    in the middle of every silence, like Whisper does over music/silence.
    """

    model_name = "fake-asr"

    def __init__(self, word_s: float = 0.5, hallucinate: bool = False):
        self.word_s = word_s
        self.hallucinate = hallucinate
        self.calls: list[float] = []          # duration of each audio passed in

    def transcribe(self, audio, sample_rate):
        from kickvoice.transcripts.asr import Word

        self.calls.append(len(audio) / sample_rate)
        frame = int(sample_rate * 0.03)
        n = len(audio) // frame
        loud = np.sqrt((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1)) > 0.02
        padded = np.concatenate(([False], loud, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        words, prev_end = [], 0.0
        for a, b in zip(edges[::2], edges[1::2]):
            start, end = a * 0.03, b * 0.03
            if self.hallucinate and start - prev_end > 0.5:
                mid = (prev_end + start) / 2
                words.append(Word(mid - 0.1, mid + 0.1, " thanks", 0.3))
            t = start
            while t < end - 0.05:
                w_end = min(t + self.word_s, end)
                seg = audio[int(t * sample_rate):int(w_end * sample_rate)]
                spec = np.abs(np.fft.rfft(seg))
                freq = round(np.fft.rfftfreq(len(seg), 1 / sample_rate)[spec.argmax()] / 10) * 10
                words.append(Word(t, w_end, f" w{int(freq)}", 0.9))
                t = w_end
            prev_end = end
        return words
