"""
Live stream audio capture from Kick.com using streamlink + ffmpeg.

Produces a continuous stream of raw PCM audio chunks for downstream processing.
"""
import subprocess
import threading
import queue
import time
import struct
import logging
from pathlib import Path
from typing import Generator

from kickvoice.config import Config, AudioConfig

logger = logging.getLogger(__name__)


class StreamCapture:
    """Captures audio from a Kick.com stream in rolling chunks."""

    def __init__(self, stream_url: str, config: Config | None = None):
        self.stream_url = stream_url
        self.config = config or Config()
        self.audio_cfg = self.config.audio
        self._process: subprocess.Popen | None = None
        self._running = False

    def _build_streamlink_cmd(self) -> list[str]:
        """Build the streamlink command to extract the stream URL."""
        return [
            "streamlink",
            "--stream-url",
            self.stream_url,
            "best",
        ]

    def _build_ffmpeg_cmd(self, input_source: str = "pipe:0") -> list[str]:
        """Build ffmpeg command to convert stream to raw PCM audio."""
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", input_source,
            "-vn",                          # drop video
            "-acodec", "pcm_s16le",         # 16-bit signed little-endian
            "-ar", str(self.audio_cfg.sample_rate),
            "-ac", str(self.audio_cfg.channels),
            "-f", "s16le",                  # raw PCM output
            "pipe:1",                       # output to stdout
        ]

    def _get_stream_hls_url(self) -> str:
        """Use streamlink to resolve the HLS URL for the stream."""
        cmd = self._build_streamlink_cmd()
        logger.info(f"Resolving stream URL: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            # With --stream-url, streamlink reports errors (e.g. "No playable
            # streams found" for an offline channel) on stdout, not stderr.
            detail = (result.stderr.strip() or result.stdout.strip()
                      or f"exit code {result.returncode}")
            raise RuntimeError(f"streamlink failed: {detail}")
        url = result.stdout.strip()
        logger.info(f"Resolved HLS URL: {url[:80]}...")
        return url

    def capture_chunks(
        self,
        chunk_duration_s: float | None = None,
        chunk_stride_s: float | None = None,
    ) -> Generator[bytes, None, None]:
        """
        Yield raw PCM audio chunks of `chunk_duration_s` length.

        Each chunk is `sample_rate * channels * 2 * chunk_duration_s` bytes
        (16-bit = 2 bytes per sample). Duration/stride default to the audio
        config; pass both equal for non-overlapping blocks.
        """
        chunk_bytes, stride_bytes = _chunk_sizes(
            self.audio_cfg, chunk_duration_s, chunk_stride_s
        )

        # Resolve the actual stream URL via streamlink
        hls_url = self._get_stream_hls_url()

        # Launch ffmpeg reading from the HLS URL directly
        cmd = self._build_ffmpeg_cmd(input_source=hls_url)
        logger.info(f"Starting ffmpeg capture: {' '.join(cmd[:6])}...")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=chunk_bytes,
        )
        self._running = True

        buffer = b""
        try:
            while self._running:
                # Read stride_bytes at a time for overlapping windows
                data = self._process.stdout.read(stride_bytes)
                if not data:
                    logger.warning("Stream ended or ffmpeg exited")
                    break

                buffer += data

                # Yield a full chunk whenever we have enough data
                while len(buffer) >= chunk_bytes:
                    chunk = buffer[:chunk_bytes]
                    yield chunk
                    # Slide by stride (overlap = chunk - stride)
                    buffer = buffer[stride_bytes:]

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the capture process."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


class FileCapture:
    """
    Capture audio from a local file — useful for testing without a live stream.
    Simulates the same chunked output as StreamCapture.
    """

    def __init__(self, file_path: str | Path, config: Config | None = None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
        self.config = config or Config()
        self.audio_cfg = self.config.audio

    def _convert_to_pcm(self) -> bytes:
        """Convert any audio file to raw PCM via ffmpeg."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(self.file_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.audio_cfg.sample_rate),
            "-ac", str(self.audio_cfg.channels),
            "-f", "s16le",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        return result.stdout

    def capture_chunks(
        self,
        chunk_duration_s: float | None = None,
        chunk_stride_s: float | None = None,
    ) -> Generator[bytes, None, None]:
        """Yield PCM chunks from the file, simulating live capture."""
        pcm_data = self._convert_to_pcm()
        chunk_bytes, stride_bytes = _chunk_sizes(
            self.audio_cfg, chunk_duration_s, chunk_stride_s
        )

        offset = 0
        while offset + chunk_bytes <= len(pcm_data):
            yield pcm_data[offset : offset + chunk_bytes]
            offset += stride_bytes

        # Yield final partial chunk if there's remaining audio > min speech dur
        remaining = pcm_data[offset:]
        min_bytes = int(
            self.audio_cfg.sample_rate * 2 * self.audio_cfg.min_speech_duration_s
        )
        if len(remaining) >= min_bytes:
            # Zero-pad to full chunk size
            yield remaining + b"\x00" * (chunk_bytes - len(remaining))


def _chunk_sizes(
    audio_cfg: AudioConfig,
    chunk_duration_s: float | None,
    chunk_stride_s: float | None,
) -> tuple[int, int]:
    """(chunk_bytes, stride_bytes) for s16le audio, falling back to config."""
    bytes_per_s = audio_cfg.sample_rate * audio_cfg.channels * 2
    dur = chunk_duration_s if chunk_duration_s is not None else audio_cfg.chunk_duration_s
    stride = chunk_stride_s if chunk_stride_s is not None else audio_cfg.chunk_stride_s
    # Keep byte counts sample-aligned so PCM never gets split mid-sample.
    frame = 2 * audio_cfg.channels
    chunk_bytes = int(bytes_per_s * dur) // frame * frame
    stride_bytes = int(bytes_per_s * stride) // frame * frame
    return chunk_bytes, stride_bytes


def pcm_to_float(pcm_bytes: bytes) -> list[float]:
    """Convert raw PCM s16le bytes to float [-1.0, 1.0] samples."""
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    return [s / 32768.0 for s in samples]
