"""
Live sessions for agents: a stream transcribed in the background, exposed
as an event log the agent polls.

    thread: TranscriptionPipeline (short blocks)
        → every line        → "line" event
        → opponent's words  → PointAggregator → "point" event (with point_id)
        → every N seconds   → whole-session voice grouping → "voices" event
    agent:  get_live_updates(cursor, wait) long-polls the log

Everything the agent needs to act on is an event with a sequence number, so
a slow or restarted agent just continues from its last cursor.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from kickvoice.config import Config
from kickvoice.pipelines.debate import Point, PointAggregator, opponent_matcher
from kickvoice.pipelines.transcribe import speaker_label
from kickvoice.speakers.profiles import ProfileStore
from kickvoice.transcripts.refine import refine_session
from kickvoice.transcripts.store import TranscriptStore, Utterance

logger = logging.getLogger(__name__)

# (stream_url, file_path, config, on_utterance) → a TranscriptionPipeline-like object
PipelineFactory = Callable[[str | None, str | None, Config, Callable[[Utterance], None]], object]


def default_pipeline_factory(stream_url, file_path, config, on_utterance, audio_path=None):
    from kickvoice.pipelines.transcribe import TranscriptionPipeline

    return TranscriptionPipeline(
        stream_url=stream_url,
        file_path=file_path,
        config=config,
        on_utterance=on_utterance,
        out=lambda _m: None,
        audio_path=audio_path,
    )


@dataclass
class Event:
    seq: int
    type: str  # line | point | voices | status
    data: dict
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"seq": self.seq, "type": self.type, **self.data}


@dataclass
class Opponent:
    """Whose points to respond to."""

    research_name: str | None = None  # research subject (for evidence)
    voice: str | None = None  # their voice profile (to spot their lines)
    me: str | None = None  # your own voice (never the opponent)
    any_speaker: bool = False  # respond to every voice except `me`

    def describe(self) -> str:
        who = self.research_name or self.voice or ("anyone" if self.any_speaker else "nobody yet")
        via = f" (voice: {self.voice})" if self.voice and self.research_name else ""
        return who + via


class LiveSession:
    def __init__(
        self,
        source: str,
        is_file: bool,
        config: Config,
        opponent: Opponent,
        pipeline_factory: PipelineFactory = default_pipeline_factory,
        max_duration_s: float | None = None,
        keep_audio: bool = False,
    ):
        self.id = uuid.uuid4().hex[:8]
        # Recorded audio lets agents cut voice clips for the user to listen to.
        self.audio_path = (config.data_dir / "live_audio" / f"{self.id}.wav") if keep_audio else None
        self.source = source
        self.is_file = is_file
        self.config = config
        self.opponent = opponent
        self.max_duration_s = max_duration_s
        self.status = "starting"  # starting | running | stopped | finished | error
        self.error: str | None = None
        self.transcript_session: str | None = None
        self.started_at = time.time()

        self._factory = pipeline_factory
        self._pipeline = None
        self._stopping = False
        self._events: deque[Event] = deque(maxlen=config.agent.max_events)
        self._seq = 0
        self._cond = threading.Condition()
        self._refine_lock = threading.Lock()
        self._last_refine = time.monotonic()
        self.points: dict[str, Point] = {}
        self._aggregator = self._make_aggregator()
        self._thread = threading.Thread(target=self._run, name=f"live-{self.id}", daemon=True)

    # ── lifecycle ────────────────────────────────────────────────

    def start(self):
        self._thread.start()

    def stop(self, timeout: float = 30.0):
        self._stopping = True
        if self._pipeline is not None:
            self._pipeline.stop()
        self._thread.join(timeout=timeout)
        if self.status in ("starting", "running"):
            self._set_status("stopped")

    def _run(self):
        try:
            self._pipeline = self._factory(
                None if self.is_file else self.source,
                self.source if self.is_file else None,
                self.config,
                self._on_utterance,
                audio_path=self.audio_path,
            )
            self._set_status("running")
            session = self._pipeline.run(max_duration_s=self.max_duration_s)
            self.transcript_session = getattr(session, "id", self.transcript_session)
            with self._cond:
                point = self._aggregator.flush_if_idle(0.0, now=float("inf"))
                if point:
                    self._emit_point(point)
            self._emit("voices", {"voices": self.voice_summary(refresh=False)})
            self._set_status("stopped" if self._stopping else "finished")
        except Exception as e:  # surface, don't crash the server
            logger.exception("Live session %s failed", self.id)
            self.error = str(e)
            self._set_status("error")

    # ── called on the transcription thread ───────────────────────

    def _on_utterance(self, u: Utterance):
        if self.transcript_session is None:
            self.transcript_session = u.session_id
        with self._cond:
            self._emit(
                "line",
                {
                    "utterance_id": u.id,
                    "t": round(u.start_s, 1),
                    "speaker": speaker_label(u),
                    "attribution": u.attribution,
                    "text": u.text,
                    "is_opponent": self._aggregator.is_opponent(u),
                },
            )
            point = self._aggregator.add(u)
            if point:
                self._emit_point(point)
        if time.monotonic() - self._last_refine >= self.config.agent.refine_interval_s:
            self._emit("voices", {"voices": self.voice_summary(refresh=True)})

    # ── polling ──────────────────────────────────────────────────

    def updates(self, cursor: int = 0, wait_s: float = 0.0, max_events: int = 100) -> dict:
        """Events after `cursor`, waiting up to `wait_s` for at least one."""
        wait_s = max(0.0, min(wait_s, self.config.agent.max_wait_s))
        deadline = time.monotonic() + wait_s
        with self._cond:
            while True:
                point = self._aggregator.flush_if_idle(self.config.debate.point_pause_s)
                if point:
                    self._emit_point(point)
                fresh = [e for e in self._events if e.seq > cursor]
                remaining = deadline - time.monotonic()
                if fresh or remaining <= 0 or self.status not in ("starting", "running"):
                    break
                self._cond.wait(timeout=min(remaining, 1.0))  # wake up to check idle points
            fresh = fresh[:max_events]
            oldest = self._events[0].seq if self._events else self._seq + 1
        return {
            "session_id": self.id,
            "status": self.status,
            "error": self.error,
            "events": [e.to_dict() for e in fresh],
            "next_cursor": fresh[-1].seq if fresh else max(cursor, self._seq),
            "missed_events": max(0, oldest - cursor - 1) if cursor and oldest > cursor + 1 else 0,
            "opponent": self.opponent.describe(),
        }

    # ── voices ───────────────────────────────────────────────────

    def refine(self) -> None:
        """Group the transcript by voice using the whole session (+ current profiles)."""
        if not self.transcript_session:
            return
        with self._refine_lock:
            names, matrix = ProfileStore(config=self.config).embedding_matrix()
            with TranscriptStore(self.config.transcripts_db) as store:
                refine_session(store, self.transcript_session, names, matrix, self.config)
            self._last_refine = time.monotonic()

    def voice_summary(self, refresh: bool = True) -> list[dict]:
        """Distinct voices so far: who they are, how much they spoke, sample lines."""
        if refresh:
            self.refine()
        if not self.transcript_session:
            return []
        with TranscriptStore(self.config.transcripts_db) as store:
            utterances = store.utterances(self.transcript_session)
        groups: dict[str, list[Utterance]] = {}
        for u in utterances:
            key = f"group:{u.voice_group}" if u.voice_group else f"speaker:{u.speaker or 'unknown'}"
            groups.setdefault(key, []).append(u)
        voices = []
        for lines in groups.values():
            speakers = {u.speaker for u in lines if u.speaker}
            samples = sorted(lines, key=lambda u: -len(u.text))[:3]
            voices.append(
                {
                    "voice_group": lines[0].voice_group,
                    "speaker": next(iter(speakers)) if len(speakers) == 1 else None,
                    "label": speaker_label(max(lines, key=lambda u: u.duration)),
                    "speech_seconds": round(sum(u.duration for u in lines), 1),
                    "lines": len(lines),
                    "is_opponent": any(self._aggregator.is_opponent(u) for u in lines),
                    "first_heard": round(min(u.start_s for u in lines), 1),
                    "sample_lines": [
                        {"utterance_id": u.id, "t": round(u.start_s, 1), "text": u.text} for u in samples
                    ],
                }
            )
        voices.sort(key=lambda v: -v["speech_seconds"])
        return voices

    def reload_profiles(self):
        """Recognise newly enrolled voices from the next block on."""
        if self._pipeline is not None and hasattr(self._pipeline, "request_profile_reload"):
            self._pipeline.request_profile_reload()

    # ── opponent ─────────────────────────────────────────────────

    def set_opponent(self, opponent: Opponent):
        with self._cond:
            self.opponent = opponent
            self._aggregator = self._make_aggregator()
            self._emit("status", {"status": self.status, "opponent": opponent.describe()})

    def _make_aggregator(self) -> PointAggregator:
        o = self.opponent
        return PointAggregator(
            opponent_matcher(o.voice, o.me, o.any_speaker),
            speaker_label,
            min_words=self.config.debate.min_point_words,
        )

    # ── internals (call with self._cond held, or from one thread) ─

    def _emit_point(self, point: Point):
        point_id = f"p{len(self.points) + 1}"
        self.points[point_id] = point
        self._emit(
            "point",
            {
                "point_id": point_id,
                "t": round(point.start_s, 1),
                "text": point.text,
                "opponent": self.opponent.describe(),
            },
        )

    def _emit(self, type_: str, data: dict):
        with self._cond:
            self._seq += 1
            self._events.append(Event(self._seq, type_, data))
            self._cond.notify_all()

    def _set_status(self, status: str):
        self.status = status
        self._emit("status", {"status": status, "error": self.error})

    def info(self) -> dict:
        return {
            "session_id": self.id,
            "source": self.source,
            "status": self.status,
            "error": self.error,
            "opponent": self.opponent.describe(),
            "research_subject": self.opponent.research_name,
            "opponent_voice": self.opponent.voice,
            "transcript_session": self.transcript_session,
            "running_for_s": round(time.time() - self.started_at),
            "points": len(self.points),
            "last_event": self._seq,
            "keeps_audio": self.audio_path is not None,
        }


class LiveSessions:
    """Registry of live sessions in this server process."""

    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def add(self, session: LiveSession) -> LiveSession:
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> LiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"No live session '{session_id}'. Start one with start_live_session.")
        return session

    def all(self) -> list[LiveSession]:
        with self._lock:
            return list(self._sessions.values())

    def stop_all(self):
        for session in self.all():
            session.stop(timeout=10)
