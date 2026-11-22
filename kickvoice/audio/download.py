"""Fetching audio from video sites (YouTube, Kick VODs, …) with yt-dlp."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from kickvoice.audio.ffmpeg import AudioError, require_tool

logger = logging.getLogger(__name__)


def download_audio(url: str, out_dir: str | Path, timeout: float = 1800) -> Path:
    """
    Download the audio track of `url` into `out_dir`.

    Returns the downloaded file. The caller decodes it like any other audio
    file, so there is no separate normalisation pass.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = out_dir / "audio.%(ext)s"
    cmd = [
        require_tool("yt-dlp", "pip install yt-dlp"),
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(template),
        "--no-playlist",
        url,
    ]
    logger.info("Downloading audio: %s", url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AudioError(f"yt-dlp timed out downloading {url}") from e
    if result.returncode != 0:
        raise AudioError(f"yt-dlp failed for {url}: {result.stderr.strip()}")

    files = sorted(p for p in out_dir.glob("audio.*") if p.is_file())
    if not files:
        raise AudioError(f"yt-dlp produced no audio file for {url}")
    return files[0]
