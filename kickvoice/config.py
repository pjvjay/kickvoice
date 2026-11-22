"""
kickvoice configuration.

Defaults live here; `~/.kickvoice/config.json` overrides them, and CLI flags
override that. Creating a Config has no side effects — directories are made
when something is actually written.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".kickvoice"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Rolling capture window used by `monitor`
    chunk_duration_s: float = 10.0
    chunk_stride_s: float = 5.0
    # VAD settings
    vad_threshold: float = 0.5
    min_speech_duration_s: float = 1.5


@dataclass
class EmbeddingConfig:
    # ECAPA-TDNN via SpeechBrain
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_savedir: str = "pretrained_models/ecapa"
    embedding_dim: int = 192
    device: str = "cpu"          # "cuda" if you installed GPU PyTorch


@dataclass
class IdentifyConfig:
    # Cosine similarity needed to call it a match
    match_threshold: float = 0.60
    # Above this, the match is reported as high confidence
    high_confidence_threshold: float = 0.75
    # How many candidates to keep/show
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
class TranscribeConfig:
    """Settings for `transcribe` (speaker-attributed transcripts)."""
    # faster-whisper model: tiny / base / small / medium / large-v3, or a path.
    # Bigger = more accurate but slower; on CPU, "small" keeps up with live
    # speech on most machines. Watch the real-time factor printed while running.
    asr_model: str = "small"
    asr_device: str = "cpu"            # "cuda" with GPU PyTorch/CTranslate2
    asr_compute_type: str = "int8"     # int8 is fastest on CPU; "float16" on GPU
    language: str | None = None        # e.g. "en"; None = auto-detect
    beam_size: int = 5
    # Audio is processed in blocks; cuts are moved into pauses so words aren't
    # split across blocks.
    chunk_duration_s: float = 30.0
    # VAD minimum for *transcription* (shorter than for profiling so short
    # replies still appear in the transcript).
    min_speech_s: float = 0.5
    # Speaker-scoring windows (same idea as extract-profiles).
    window_s: float = 2.0
    window_hop_s: float = 1.0
    min_window_s: float = 1.0
    # A turn needs at least this much audio to be attributed "confident";
    # shorter turns top out at "probable".
    min_confident_turn_s: float = 2.0
    # Split a speaker's turn at pauses longer than this.
    max_pause_s: float = 1.0
    # Store text from unrecognised voices (useful context for analysis, e.g.
    # what the streamer is responding to). Set False to keep only enrolled
    # speakers' words.
    keep_unattributed: bool = True


@dataclass
class LLMConfig:
    """
    Language model access for research and live debate help.

    API keys are read from environment variables only (never stored in the
    config file): ANTHROPIC_API_KEY, or OPENAI_API_KEY for "openai".
    """
    # "anthropic", or "openai" for any OpenAI-compatible server
    # (OpenAI, Ollama at http://localhost:11434/v1, vLLM, LM Studio, …)
    provider: str = "anthropic"
    base_url: str | None = None          # None = the provider's default
    api_key_env: str | None = None       # None = ANTHROPIC_API_KEY / OPENAI_API_KEY
    # Thorough model for offline analysis; fast model for live suggestions.
    analysis_model: str = "claude-sonnet-4-5"
    live_model: str = "claude-haiku-4-5"
    timeout_s: float = 60.0
    # Optional embeddings (OpenAI-compatible /embeddings). Without them,
    # retrieval uses SQLite full-text search only.
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key_env: str | None = None


@dataclass
class ResearchConfig:
    """Settings for `research` (collecting a public figure's public record)."""
    # Identify yourself to the sites you fetch from (Wikimedia requires a UA
    # with contact info: https://meta.wikimedia.org/wiki/User-Agent_policy).
    user_agent: str = "kickvoice-research/0.1 (debate prep; set research.user_agent to add contact)"
    # Minimum seconds between requests to the same host.
    min_request_interval_s: float = 1.0
    cache_ttl_hours: float = 24.0
    # Per-run collection limits (keep first runs cheap; raise as needed).
    max_videos: int = 25
    max_posts: int = 1000
    # Service endpoints (overridable for proxies / testing).
    wikipedia_api: str = "https://en.wikipedia.org/w/api.php"
    wikidata_api: str = "https://www.wikidata.org/w/api.php"
    x_api: str = "https://api.twitter.com/2"
    reddit_api: str = "https://oauth.reddit.com"
    reddit_auth: str = "https://www.reddit.com/api/v1/access_token"
    brave_search_api: str = "https://api.search.brave.com/res/v1/web/search"
    # Credentials: environment variable *names* (values never stored).
    x_bearer_token_env: str = "X_BEARER_TOKEN"
    reddit_client_id_env: str = "REDDIT_CLIENT_ID"
    reddit_client_secret_env: str = "REDDIT_CLIENT_SECRET"
    brave_api_key_env: str = "BRAVE_API_KEY"
    # Claim extraction: characters of source text per LLM call.
    extraction_batch_chars: int = 12000


@dataclass
class DebateConfig:
    """Settings for `debate` (live counter-argument suggestions)."""
    # Shorter transcription blocks than `transcribe` → lower latency.
    chunk_duration_s: float = 8.0
    # Collect this many words of the opponent's speech before responding,
    # unless they stop talking first.
    min_point_words: int = 20
    # Respond after this long without new lines from the opponent.
    point_pause_s: float = 4.0
    # Evidence passed to the model per point.
    evidence_items: int = 12
    # Skip stale points if the model falls behind (keep only the newest N).
    max_queued_points: int = 2


SECTIONS = {
    "audio": AudioConfig,
    "embedding": EmbeddingConfig,
    "identify": IdentifyConfig,
    "extract": ExtractConfig,
    "transcribe": TranscribeConfig,
    "llm": LLMConfig,
    "research": ResearchConfig,
    "debate": DebateConfig,
}


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    identify: IdentifyConfig = field(default_factory=IdentifyConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data_dir: Path = DEFAULT_DATA_DIR
    # Defaults to <data_dir>/profiles when not set explicitly.
    profiles_dir: Path | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.profiles_dir = (Path(self.profiles_dir) if self.profiles_dir
                             else self.data_dir / "profiles")

    @property
    def extractions_dir(self) -> Path:
        """Where extract-profiles sessions (candidates + review clips) live."""
        return self.data_dir / "extractions"

    @property
    def transcripts_db(self) -> Path:
        """SQLite database of speaker-attributed transcripts."""
        return self.data_dir / "transcripts.db"

    @property
    def research_db(self) -> Path:
        """SQLite database of people, accounts, collected documents and claims."""
        return self.data_dir / "research.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"

    # ── persistence ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        data = {name: asdict(getattr(self, name)) for name in SECTIONS}
        data["data_dir"] = str(self.data_dir)
        data["profiles_dir"] = str(self.profiles_dir)
        return data

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """
        Read a config file. Missing file → defaults. Unknown keys are ignored
        so a newer config doesn't break an older install.
        """
        path = Path(path) if path else DEFAULT_DATA_DIR / "config.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid config file {path}: {e}") from e

        cfg = cls(
            data_dir=Path(data["data_dir"]) if data.get("data_dir") else DEFAULT_DATA_DIR,
            profiles_dir=Path(data["profiles_dir"]) if data.get("profiles_dir") else None,
        )
        for name, section_cls in SECTIONS.items():
            known = {f.name for f in fields(section_cls)}
            for key, value in (data.get(name) or {}).items():
                if key in known:
                    setattr(getattr(cfg, name), key, value)
        return cfg
