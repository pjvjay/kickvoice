"""
Speaker embedding extraction using SpeechBrain's ECAPA-TDNN model.

Converts speech audio segments into fixed-length 192-dim vectors (embeddings)
that encode speaker identity. Two utterances from the same speaker will have
high cosine similarity, regardless of what words are spoken.
"""
import logging
import struct
from pathlib import Path

import numpy as np

from kickvoice.config import Config, EmbeddingConfig
from kickvoice.live.detect import SpeechSegment

logger = logging.getLogger(__name__)


class SpeakerEncoder:
    """
    Wraps SpeechBrain's pre-trained ECAPA-TDNN for speaker embedding extraction.
    Model is downloaded automatically on first use (~80MB).
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.emb_cfg = self.config.embedding
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        from speechbrain.inference.speaker import EncoderClassifier

        logger.info(
            f"Loading speaker encoder: {self.emb_cfg.model_source}"
        )
        savedir = Path(self.config.data_dir) / self.emb_cfg.model_savedir
        self._model = EncoderClassifier.from_hparams(
            source=self.emb_cfg.model_source,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        logger.info("Speaker encoder loaded")

    def embed_segment(self, segment: SpeechSegment) -> np.ndarray:
        """
        Extract a speaker embedding from a single speech segment.

        Args:
            segment: A SpeechSegment with PCM audio data

        Returns:
            numpy array of shape (192,) — the speaker embedding
        """
        import torch

        self._load_model()

        # Convert PCM bytes to float tensor
        audio_tensor = segment.to_float_tensor().unsqueeze(0)  # [1, n_samples]

        # Extract embedding
        with torch.no_grad():
            embedding = self._model.encode_batch(audio_tensor)

        # embedding shape: [1, 1, 192] → flatten to (192,)
        return embedding.squeeze().cpu().numpy()

    def embed_pcm(self, pcm_bytes: bytes) -> np.ndarray:
        """
        Extract embedding directly from raw PCM bytes.

        Args:
            pcm_bytes: Raw PCM s16le audio (16kHz mono)

        Returns:
            numpy array of shape (192,)
        """
        import torch

        self._load_model()

        n_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
        audio_tensor = torch.FloatTensor(samples).unsqueeze(0) / 32768.0

        with torch.no_grad():
            embedding = self._model.encode_batch(audio_tensor)

        return embedding.squeeze().cpu().numpy()

    def embed_file(self, audio_path: str | Path) -> np.ndarray:
        """
        Extract embedding from an audio file (any format ffmpeg supports).

        Converts to 16kHz mono PCM first, then extracts embedding.
        """
        import subprocess

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Convert to raw PCM
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(self.config.audio.sample_rate),
            "-ac", str(self.config.audio.channels),
            "-f", "s16le",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

        return self.embed_pcm(result.stdout)

    def embed_segments(
        self, segments: list[SpeechSegment]
    ) -> list[np.ndarray]:
        """Extract embeddings for multiple segments."""
        return [self.embed_segment(seg) for seg in segments]

    def compute_mean_embedding(
        self, embeddings: list[np.ndarray]
    ) -> np.ndarray:
        """
        Compute the centroid of multiple embeddings.
        Useful for creating a stable speaker profile from many utterances.
        """
        if not embeddings:
            raise ValueError("No embeddings to average")
        stacked = np.stack(embeddings)
        mean = stacked.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean
