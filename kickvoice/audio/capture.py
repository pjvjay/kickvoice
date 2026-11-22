"""
Audio capture from a live stream (streamlink + ffmpeg) or a local file.

Both sources expose the same interface — `capture_chunks()` yielding PCM
chunks — so pipelines don't care where the audio comes from:

    capture = open_capture(stream_url="https://kick.com/channel", config=cfg)
    for chunk in capture.capture_chunks():   # overlapping or not, your choice
        ...
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

from kickvoice.audio.ffmpeg import (
    AudioError, decode_to_pcm, open_pcm_stream, require_tool, seconds_to_bytes,
)
from kickvoice.config import AudioConfig, Config

logger = logging.getLogger(__name__)


def rechunk(
    blocks: Iterable[bytes],
    chunk_bytes: int,
    stride_bytes: int,
    min_tail_bytes: int = 0,
) -> Iterator[bytes]:
    """
    Turn a stream of arbitrary byte blocks into fixed-size chunks.

    `stride_bytes` < `chunk_bytes` gives overlapping windows (monitoring);
    equal values give non-overlapping blocks (extraction, so no audio is
    embedded twice). A trailing partial chunk is yielded if it is at least
    `min_tail_bytes` long.
    """
    buffer = b""
    for block in blocks:
        if not block:
            continue
        buffer += block
        while len(buffer) >= chunk_bytes:
            yield buffer[:chunk_bytes]
            buffer = buffer[stride_bytes:]
    if len(buffer) >= min_tail_bytes > 0:
        yield buffer


class BaseCapture:
    """Shared chunking; subclasses only provide raw PCM blocks."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.audio_cfg: AudioConfig = self.config.audio

    def _pcm_blocks(self, block_bytes: int) -> Iterator[bytes]:
        raise NotImplementedError

    def capture_chunks(
        self,
        chunk_duration_s: float | None = None,
        chunk_stride_s: float | None = None,
    ) -> Iterator[bytes]:
        """
        Yield PCM chunks of `chunk_duration_s` every `chunk_stride_s`
        (both default to the audio config).
        """
        cfg = self.audio_cfg
        chunk_bytes = seconds_to_bytes(
            chunk_duration_s if chunk_duration_s is not None else cfg.chunk_duration_s, cfg)
        stride_bytes = seconds_to_bytes(
            chunk_stride_s if chunk_stride_s is not None else cfg.chunk_stride_s, cfg)
        if chunk_bytes <= 0 or stride_bytes <= 0:
            raise ValueError("chunk duration and stride must be positive")
        min_tail = seconds_to_bytes(cfg.min_speech_duration_s, cfg)
        try:
            yield from rechunk(self._pcm_blocks(stride_bytes),
                               chunk_bytes, stride_bytes, min_tail)
        finally:
            self.stop()

    def stop(self):
        """Release any resources (no-op unless overridden)."""


class StreamCapture(BaseCapture):
    """Captures audio from a live stream via streamlink + ffmpeg."""

    def __init__(self, stream_url: str, config: Config | None = None):
        super().__init__(config)
        self.stream_url = stream_url
        self._process: subprocess.Popen | None = None

    def resolve_url(self) -> str:
        """Ask streamlink for the playable media URL behind the page URL."""
        cmd = [require_tool("streamlink", "pip install streamlink"),
               "--stream-url", self.stream_url, "best"]
        logger.info("Resolving stream URL: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as e:
            raise AudioError(f"streamlink timed out resolving {self.stream_url}") from e
        if result.returncode != 0:
            # With --stream-url, streamlink reports errors (e.g. "No playable
            # streams found" for an offline channel) on stdout, not stderr.
            detail = (result.stderr.strip() or result.stdout.strip()
                      or f"exit code {result.returncode}")
            raise AudioError(f"streamlink failed: {detail}")
        url = result.stdout.strip()
        logger.info("Resolved stream URL: %s…", url[:80])
        return url

    def _pcm_blocks(self, block_bytes: int) -> Iterator[bytes]:
        self._process = open_pcm_stream(self.resolve_url(), self.audio_cfg)
        stdout = self._process.stdout
        try:
            while True:
                data = stdout.read(block_bytes)
                if not data:
                    logger.info("Stream ended")
                    return
                yield data
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")

    def stop(self):
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class FileCapture(BaseCapture):
    """Captures audio from a local file — handy for testing without a stream."""

    def __init__(self, file_path: str | Path, config: Config | None = None):
        super().__init__(config)
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise AudioError(f"Audio file not found: {self.file_path}")

    def _pcm_blocks(self, block_bytes: int) -> Iterator[bytes]:
        pcm = decode_to_pcm(self.file_path, self.audio_cfg)
        for offset in range(0, len(pcm), block_bytes):
            yield pcm[offset:offset + block_bytes]


def open_capture(
    stream_url: str | None = None,
    file_path: str | Path | None = None,
    config: Config | None = None,
) -> BaseCapture:
    """Build the right capture for whichever source was given."""
    if bool(stream_url) == bool(file_path):
        raise ValueError("Provide exactly one of stream_url or file_path")
    if stream_url:
        return StreamCapture(stream_url, config=config)
    return FileCapture(file_path, config=config)
