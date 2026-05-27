"""
Voice Activity Detection (VAD) to isolate speech segments from audio chunks.

Uses Silero VAD (via torch.hub) for high-quality speech detection,
with an energy-based fallback for environments without torch.
"""
import logging
import struct
from dataclasses import dataclass

from kickvoice.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech region within an audio chunk."""
    start_sample: int
    end_sample: int
    pcm_bytes: bytes
    confidence: float = 0.0

    @property
    def duration_s(self) -> float:
        """Duration in seconds (assumes 16kHz mono 16-bit)."""
        return len(self.pcm_bytes) / (16000 * 2)

    def to_float_tensor(self):
        """Convert PCM bytes to a torch float tensor."""
        import torch
        n_samples = len(self.pcm_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", self.pcm_bytes)
        return torch.FloatTensor(samples) / 32768.0


class SileroVAD:
    """
    Silero VAD wrapper.
    Downloads the model on first use via torch.hub.
    """

    def __init__(self, audio_cfg: AudioConfig | None = None):
        self.audio_cfg = audio_cfg or AudioConfig()
        self._model = None
        self._utils = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model = model
        self._utils = utils
        logger.info("Silero VAD loaded")

    def detect_speech(
        self,
        pcm_bytes: bytes,
        threshold: float | None = None,
        min_duration_s: float | None = None,
    ) -> list[SpeechSegment]:
        """
        Detect speech segments in a raw PCM chunk.

        Args:
            pcm_bytes: Raw PCM s16le audio bytes (16kHz mono)
            threshold: VAD confidence threshold (default from config)
            min_duration_s: Minimum speech segment duration to keep

        Returns:
            List of SpeechSegment objects containing isolated speech audio
        """
        import torch

        self._load_model()

        threshold = threshold or self.audio_cfg.vad_threshold
        min_duration_s = min_duration_s or self.audio_cfg.min_speech_duration_s

        # Convert PCM bytes → float tensor
        n_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
        audio_tensor = torch.FloatTensor(samples) / 32768.0

        # Get speech timestamps from Silero
        get_speech_timestamps = self._utils[0]
        timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=self.audio_cfg.sample_rate,
            threshold=threshold,
            min_speech_duration_ms=int(min_duration_s * 1000),
            return_seconds=False,
        )

        segments = []
        for ts in timestamps:
            start = ts["start"]
            end = ts["end"]
            # Extract the PCM bytes for this segment
            # Each sample = 2 bytes (s16le)
            seg_bytes = pcm_bytes[start * 2 : end * 2]
            duration = (end - start) / self.audio_cfg.sample_rate

            if duration >= min_duration_s:
                segments.append(SpeechSegment(
                    start_sample=start,
                    end_sample=end,
                    pcm_bytes=seg_bytes,
                    confidence=threshold,
                ))

        logger.debug(
            f"VAD found {len(segments)} speech segments "
            f"(from {len(timestamps)} raw detections)"
        )
        return segments


class EnergyVAD:
    """
    Simple energy-based VAD fallback. No ML dependencies.
    Less accurate but useful for testing the pipeline.
    """

    def __init__(self, audio_cfg: AudioConfig | None = None):
        self.audio_cfg = audio_cfg or AudioConfig()

    def detect_speech(
        self,
        pcm_bytes: bytes,
        threshold: float | None = None,
        min_duration_s: float | None = None,
    ) -> list[SpeechSegment]:
        threshold = threshold or 0.02  # RMS energy threshold
        min_duration_s = min_duration_s or self.audio_cfg.min_speech_duration_s

        n_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
        float_samples = [s / 32768.0 for s in samples]

        # Compute energy in 30ms frames
        frame_size = int(self.audio_cfg.sample_rate * 0.03)
        min_frames = int(min_duration_s / 0.03)

        is_speech = []
        for i in range(0, len(float_samples) - frame_size, frame_size):
            frame = float_samples[i : i + frame_size]
            rms = (sum(x * x for x in frame) / len(frame)) ** 0.5
            is_speech.append(rms > threshold)

        # Group consecutive speech frames into segments
        segments = []
        in_seg = False
        seg_start = 0
        for i, speech in enumerate(is_speech):
            if speech and not in_seg:
                seg_start = i
                in_seg = True
            elif not speech and in_seg:
                if (i - seg_start) >= min_frames:
                    start_sample = seg_start * frame_size
                    end_sample = min(i * frame_size, n_samples)
                    seg_bytes = pcm_bytes[start_sample * 2 : end_sample * 2]
                    segments.append(SpeechSegment(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        pcm_bytes=seg_bytes,
                        confidence=0.5,
                    ))
                in_seg = False

        # Handle segment that runs to end of chunk
        if in_seg and (len(is_speech) - seg_start) >= min_frames:
            start_sample = seg_start * frame_size
            seg_bytes = pcm_bytes[start_sample * 2:]
            segments.append(SpeechSegment(
                start_sample=start_sample,
                end_sample=n_samples,
                pcm_bytes=seg_bytes,
                confidence=0.5,
            ))

        logger.debug(f"EnergyVAD found {len(segments)} speech segments")
        return segments


def get_vad(use_silero: bool = True, audio_cfg: AudioConfig | None = None):
    """Factory: return Silero VAD if available, else energy-based fallback."""
    if use_silero:
        try:
            import torch
            return SileroVAD(audio_cfg)
        except ImportError:
            logger.warning("torch not available, falling back to EnergyVAD")
    return EnergyVAD(audio_cfg)
