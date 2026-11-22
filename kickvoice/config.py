"""
kickvoice configuration
"""
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Rolling capture window
    chunk_duration_s: float = 10.0
    chunk_stride_s: float = 5.0
    # VAD settings
    vad_threshold: float = 0.5
    min_speech_duration_s: float = 1.5
    # ffmpeg format
    format: str = "wav"


@dataclass
class EmbeddingConfig:
    # ECAPA-TDNN via SpeechBrain
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_savedir: str = "pretrained_models/ecapa"
    embedding_dim: int = 192


@dataclass
class IdentifyConfig:
    # Cosine similarity threshold for a positive match
    match_threshold: float = 0.60
    # Above this = high confidence
    high_confidence_threshold: float = 0.75
    # Top-N candidates to consider
    top_n: int = 3


@dataclass
class ExtractConfig:
    """Settings for `extract-profiles` (discovering speakers in a stream)."""
    # How long to listen before clustering (0 = until Ctrl+C / stream end)
    capture_duration_s: float = 300.0
    # Capture block size. Blocks do not overlap, so no audio is embedded twice.
    chunk_duration_s: float = 30.0
    # Sliding embedding windows inside each VAD speech segment. Short windows
    # keep a single window from spanning two people taking quick turns
    # (in testing, 3s windows merged back-and-forth speakers that 2s split).
    window_s: float = 2.0
    window_hop_s: float = 1.0
    min_window_s: float = 1.5
    # Agglomerative (average-linkage) clustering cut-off, as cosine DISTANCE
    # (1 - similarity). Lower = more, tighter clusters (risk: one person split
    # in two). Higher = fewer clusters (risk: two people merged).
    # Model-specific: 0.70 targets ECAPA-TDNN (VoxCeleb) score ranges.
    cluster_distance_threshold: float = 0.70
    # Clusters with less speech than this are discarded (noise, music,
    # one-off TTS donations, crosstalk).
    min_speaker_speech_s: float = 20.0
    # Above this many windows, cluster a uniform subsample and assign the rest
    # to the nearest cluster (keeps memory/time bounded on long captures).
    max_cluster_windows: int = 4000
    # Length of the review clip written for each discovered speaker.
    sample_clip_s: float = 20.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    identify: IdentifyConfig = field(default_factory=IdentifyConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    # Paths
    data_dir: Path = Path.home() / ".kickvoice"
    profiles_dir: Path = Path.home() / ".kickvoice" / "profiles"

    @property
    def extractions_dir(self) -> Path:
        """Where extract-profiles sessions (candidates + review clips) live."""
        return Path(self.data_dir) / "extractions"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def save(self, path: Path | None = None):
        path = path or (self.data_dir / "config.json")
        data = {
            "audio": self.audio.__dict__,
            "embedding": self.embedding.__dict__,
            "identify": self.identify.__dict__,
            "extract": self.extract.__dict__,
            "data_dir": str(self.data_dir),
            "profiles_dir": str(self.profiles_dir),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or (Path.home() / ".kickvoice" / "config.json")
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        cfg = cls()
        for k, v in data.get("audio", {}).items():
            setattr(cfg.audio, k, v)
        for k, v in data.get("embedding", {}).items():
            setattr(cfg.embedding, k, v)
        for k, v in data.get("identify", {}).items():
            setattr(cfg.identify, k, v)
        for k, v in data.get("extract", {}).items():
            setattr(cfg.extract, k, v)
        if "data_dir" in data:
            cfg.data_dir = Path(data["data_dir"])
        if "profiles_dir" in data:
            cfg.profiles_dir = Path(data["profiles_dir"])
        return cfg
