"""
Speaker enrollment: create voice profiles from audio files or YouTube videos.

Pipeline:
    Audio source → ffmpeg normalize → VAD → segment → embed → average → store
"""
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from kickvoice.config import Config
from kickvoice.live.detect import get_vad, SpeechSegment
from kickvoice.models.ecapa import SpeakerEncoder
from kickvoice.storage.profiles import ProfileStore

logger = logging.getLogger(__name__)


class Enroller:
    """Enroll new speakers from audio sources."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.encoder = SpeakerEncoder(config=self.config)
        self.vad = get_vad(use_silero=True, audio_cfg=self.config.audio)
        self.store = ProfileStore(config=self.config)

    def _download_youtube_audio(self, url: str, output_dir: Path) -> Path:
        """
        Download audio from a YouTube URL using yt-dlp.

        Returns path to the downloaded audio file.
        """
        output_path = output_dir / "yt_audio.%(ext)s"
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(output_path),
            "--no-playlist",
            url,
        ]
        logger.info(f"Downloading audio from: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        # Find the output file
        wav_files = list(output_dir.glob("yt_audio.*"))
        if not wav_files:
            raise RuntimeError("yt-dlp produced no output file")
        return wav_files[0]

    def _normalize_audio(self, input_path: Path, output_path: Path) -> Path:
        """Normalize audio to 16kHz mono WAV via ffmpeg."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.config.audio.sample_rate),
            "-ac", str(self.config.audio.channels),
            "-y",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg normalize failed: {result.stderr.decode()}")
        return output_path

    def _audio_to_pcm(self, audio_path: Path) -> bytes:
        """Read audio file as raw PCM bytes."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(self.config.audio.sample_rate),
            "-ac", str(self.config.audio.channels),
            "-f", "s16le",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        return result.stdout

    def enroll_from_file(
        self,
        name: str,
        audio_path: str | Path,
        notes: str = "",
    ) -> int:
        """
        Enroll a speaker from a local audio file.

        Args:
            name: Speaker name
            audio_path: Path to audio file (any format ffmpeg supports)
            notes: Optional notes

        Returns:
            Number of speech segments used for enrollment
        """
        audio_path = Path(audio_path)
        logger.info(f"Enrolling '{name}' from: {audio_path}")

        # Convert to PCM
        pcm_data = self._audio_to_pcm(audio_path)
        logger.info(
            f"Audio: {len(pcm_data) / (self.config.audio.sample_rate * 2):.1f}s "
            f"({len(pcm_data)} bytes)"
        )

        # Run VAD to get speech segments
        segments = self.vad.detect_speech(pcm_data)
        if not segments:
            raise ValueError(
                f"No speech detected in {audio_path}. "
                "Try adjusting VAD threshold or using a different file."
            )

        logger.info(f"Found {len(segments)} speech segments")

        # Extract embeddings from each segment
        embeddings = self.encoder.embed_segments(segments)
        logger.info(f"Extracted {len(embeddings)} embeddings")

        # Compute mean embedding (the speaker's "voiceprint")
        mean_embedding = self.encoder.compute_mean_embedding(embeddings)

        # Store profile
        self.store.add_profile(
            name=name,
            embedding=mean_embedding,
            source_urls=[str(audio_path)],
            n_segments=len(segments),
            notes=notes,
        )

        logger.info(
            f"Enrolled '{name}' with {len(segments)} segments "
            f"({sum(s.duration_s for s in segments):.1f}s total speech)"
        )
        return len(segments)

    def enroll_from_youtube(
        self,
        name: str,
        urls: list[str],
        notes: str = "",
    ) -> int:
        """
        Enroll a speaker from one or more YouTube video URLs.

        Downloads audio, runs VAD, extracts embeddings, and stores the profile.

        Args:
            name: Speaker name
            urls: List of YouTube URLs
            notes: Optional notes

        Returns:
            Total number of speech segments used
        """
        all_segments: list[SpeechSegment] = []

        with tempfile.TemporaryDirectory(prefix="kickvoice_") as tmpdir:
            tmpdir = Path(tmpdir)

            for i, url in enumerate(urls):
                logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

                # Download
                raw_audio = self._download_youtube_audio(url, tmpdir / f"dl_{i}")
                (tmpdir / f"dl_{i}").mkdir(exist_ok=True)
                raw_audio = self._download_youtube_audio(url, tmpdir / f"dl_{i}")

                # Normalize
                norm_audio = tmpdir / f"normalized_{i}.wav"
                self._normalize_audio(raw_audio, norm_audio)

                # Convert to PCM and run VAD
                pcm_data = self._audio_to_pcm(norm_audio)
                segments = self.vad.detect_speech(pcm_data)
                logger.info(f"  URL {i+1}: {len(segments)} speech segments")
                all_segments.extend(segments)

        if not all_segments:
            raise ValueError(
                f"No speech detected across {len(urls)} videos. "
                "Check the URLs or VAD settings."
            )

        # Extract embeddings and compute mean
        embeddings = self.encoder.embed_segments(all_segments)
        mean_embedding = self.encoder.compute_mean_embedding(embeddings)

        # Store
        self.store.add_profile(
            name=name,
            embedding=mean_embedding,
            source_urls=urls,
            n_segments=len(all_segments),
            notes=notes,
        )

        total_speech = sum(s.duration_s for s in all_segments)
        logger.info(
            f"Enrolled '{name}' from {len(urls)} videos: "
            f"{len(all_segments)} segments, {total_speech:.1f}s total speech"
        )
        return len(all_segments)
