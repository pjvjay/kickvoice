"""
All ffmpeg/PCM handling lives here.

Everything downstream (enrollment, monitoring, extraction) works on the same
audio representation: raw little-endian 16-bit PCM at the configured sample
rate and channel count.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from kickvoice.config import AudioConfig

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2


class AudioError(RuntimeError):
    """Audio decoding/capture failed (bad file, dead stream, missing tool)."""


def require_tool(name: str, install_hint: str = "") -> str:
    """Path to an external tool, or an AudioError explaining how to get it."""
    path = shutil.which(name)
    if not path:
        hint = install_hint or f"Install {name} and make sure it is on your PATH."
        raise AudioError(f"'{name}' not found. {hint}")
    return path


def decode_cmd(source: str | Path, audio_cfg: AudioConfig) -> list[str]:
    """ffmpeg command decoding any file/URL to raw PCM on stdout."""
    return [
        require_tool("ffmpeg", "macOS: brew install ffmpeg · Ubuntu: sudo apt install ffmpeg"),
        "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vn",                                  # drop video
        "-acodec", "pcm_s16le",                 # 16-bit signed little-endian
        "-ar", str(audio_cfg.sample_rate),
        "-ac", str(audio_cfg.channels),
        "-f", "s16le",                          # raw PCM, no header
        "pipe:1",
    ]


def decode_to_pcm(
    source: str | Path,
    audio_cfg: AudioConfig,
    timeout: float | None = 300,
) -> bytes:
    """
    Decode a local file (or URL) fully to PCM bytes.

    Replaces the old per-module ffmpeg calls, so every pipeline normalises
    audio the same way: no separate "normalize to wav then decode again" step.
    """
    if "://" not in str(source) and not Path(source).exists():
        raise AudioError(f"Audio file not found: {source}")
    try:
        result = subprocess.run(decode_cmd(source, audio_cfg),
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AudioError(f"ffmpeg timed out decoding {source}") from e
    if result.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {source}: "
                         f"{result.stderr.decode(errors='replace').strip()}")
    if not result.stdout:
        raise AudioError(f"No audio found in {source}")
    return result.stdout


def open_pcm_stream(source: str | Path, audio_cfg: AudioConfig) -> subprocess.Popen:
    """Start ffmpeg decoding `source` continuously; read PCM from .stdout."""
    cmd = decode_cmd(source, audio_cfg)
    logger.debug("Starting ffmpeg: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ── conversions ──────────────────────────────────────────────────

def pcm_to_numpy(pcm: bytes) -> np.ndarray:
    """PCM bytes → float32 samples in [-1, 1)."""
    # 'copy' so callers get a writable array independent of the buffer.
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def pcm_to_tensor(pcm: bytes):
    """PCM bytes → 1-D torch float tensor (torch imported lazily)."""
    import torch

    return torch.from_numpy(pcm_to_numpy(pcm))


def pcm_duration_s(pcm: bytes, audio_cfg: AudioConfig) -> float:
    return len(pcm) / (audio_cfg.sample_rate * audio_cfg.channels * BYTES_PER_SAMPLE)


def seconds_to_bytes(seconds: float, audio_cfg: AudioConfig) -> int:
    """Byte count for `seconds` of audio, aligned to whole samples."""
    frame = BYTES_PER_SAMPLE * audio_cfg.channels
    return int(seconds * audio_cfg.sample_rate * audio_cfg.channels * BYTES_PER_SAMPLE) // frame * frame


def write_wav(path: str | Path, pcm: bytes, sample_rate: int, channels: int = 1):
    """Write PCM bytes as a playable .wav file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
