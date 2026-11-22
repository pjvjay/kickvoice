"""
Speaker embeddings via SpeechBrain's ECAPA-TDNN (trained on VoxCeleb).

Turns speech audio into a fixed-length vector encoding *who* is speaking:
two utterances from the same person score high cosine similarity regardless
of the words spoken.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from kickvoice.audio.ffmpeg import decode_to_pcm, pcm_to_tensor
from kickvoice.audio.vad import SpeechSegment
from kickvoice.config import Config

logger = logging.getLogger(__name__)


class SpeakerEncoder:
    """
    Pre-trained ECAPA-TDNN wrapper. The model (~80MB) downloads on first use.

    All `embed_*` helpers funnel into `embed_pcm`, so there is exactly one
    code path that touches the model.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.emb_cfg = self.config.embedding
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = Path(self.config.data_dir) / self.emb_cfg.model_savedir
        logger.info("Loading speaker encoder: %s", self.emb_cfg.model_source)
        self._model = EncoderClassifier.from_hparams(
            source=self.emb_cfg.model_source,
            savedir=str(savedir),
            run_opts={"device": self.emb_cfg.device},
        )
        logger.info("Speaker encoder loaded")

    def embed_pcm(self, pcm_bytes: bytes) -> np.ndarray:
        """Embed raw PCM (16-bit mono at the configured sample rate)."""
        import torch

        self._load_model()
        audio = pcm_to_tensor(pcm_bytes).unsqueeze(0)  # [1, n_samples]
        with torch.no_grad():
            embedding = self._model.encode_batch(audio)
        return embedding.squeeze().cpu().numpy()  # [1, 1, D] → (D,)

    def embed_segment(self, segment: SpeechSegment) -> np.ndarray:
        return self.embed_pcm(segment.pcm_bytes)

    def embed_segments(self, segments: list[SpeechSegment]) -> list[np.ndarray]:
        return [self.embed_segment(s) for s in segments]

    def embed_file(self, audio_path: str | Path) -> np.ndarray:
        """Embed a whole audio file (any format ffmpeg can read)."""
        return self.embed_pcm(decode_to_pcm(audio_path, self.config.audio))

    @staticmethod
    def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Centroid of several embeddings, L2-normalised — a speaker's
        "voiceprint" built from many utterances.
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings to average")
        mean = np.stack(embeddings).mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    # Backwards-compatible alias for the pre-refactor name.
    compute_mean_embedding = mean_embedding
