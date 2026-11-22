"""Shared fixtures: isolated config dirs + model-free fakes for the pipeline."""
import math
import struct

import numpy as np
import pytest

from kickvoice.config import Config

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


class FakeEncoder:
    """Maps a window to its dominant frequency's voiceprint (+ a little noise)."""

    def __init__(self, noise: float = 0.15, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        spec = np.abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]
        bucket = round(freq / 10) * 10          # tolerate FFT bin jitter
        v = voice_vector(bucket) + self.rng.normal(scale=self.noise / math.sqrt(DIM), size=DIM)
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
