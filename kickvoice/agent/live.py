"""
A live stream session the agent can poll.

The transcription pipeline runs in a background thread; every line is kept
(with who said it), and the tracked speakers' lines are grouped into
"points" — complete arguments worth responding to. The agent pulls points
one at a time with `next_point()`, which blocks until one is ready.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from kickvoice.config import Config
from kickvoice.pipelines.debate import PointAggregator
from kickvoice.transcripts.attribution import UNKNOWN
from kickvoice.transcripts.store import Utterance

logger = logging.getLogger(__name__)

PipelineFactory = Callable[..., object]


@dataclass
class LiveLine:
    index: int
    at: str                      # wall clock when transcribed
    offset_s: float              # seconds into the stream session
    speaker: str | None          # voice profile (None = unrecognised)
    label: str                   # "Jane", "Jane?" (probable), "(unknown)"
    attribution: str
    text: str


@dataclass
class LivePoint:
    id: int
    at: str
    offset_s: float
    end_offset_s: float
    speaker: str | None
    label: str
    attribution: str
    text: str
    context: list[str] = field(default_factory=list)
    delivered: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("delivered")
        return d


def _stream_start(pipeline) -> datetime | None:
    session = getattr(pipeline, "session", None)
    started = getattr(session, "started_at", None)
    return datetime.fromisoformat(started) if started else None


def _default_pipeline(source: str, config: Config, on_utterance):
    from kickvoice.pipelines.transcribe import TranscriptionPipeline

    is_url = "://" in source
    return TranscriptionPipeline(stream_url=source if is_url else None,
                                 file_path=None if is_url else source,
                                 config=config, on_utterance=on_utterance,
                                 out=lambda msg: logger.info(msg.strip()))


class LiveSession:
    def __init__(
        self,
        config: Config,
        source: str,
        speakers: list[str] | None = None,
        duration_s: float | None = None,
        pipeline_factory: PipelineFactory = _default_pipeline,
    ):
        from kickvoice.pipelines.transcribe import speaker_label

        self.config = config
        self.source = source
        self.speakers = [s for s in (speakers or []) if s]
        self.duration_s = duration_s
        self._factory = pipeline_factory
        self._label = speaker_label
        tracked = {s.casefold() for s in self.speakers}

        def is_tracked(u: Utterance) -> bool:
            if not tracked:
                return True                     # every voice can make a point
            return (u.speaker or "").casefold() in tracked and u.attribution != UNKNOWN

        self.aggregator = PointAggregator(is_tracked, speaker_label,
                                          min_words=config.debate.min_point_words)
        self.lines: list[LiveLine] = []
        self.points: list[LivePoint] = []
        self.started_at = datetime.now()
        self.finished = False
        self.error: str | None = None
        self.pipeline = None
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="kickvoice-live", daemon=True)

    # ── lifecycle ────────────────────────────────────────────────

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            # Built inside the thread: its database connections belong here.
            self.pipeline = self._factory(self.source, self.config, self._on_line)
            self.pipeline.run(max_duration_s=self.duration_s)
        except Exception as e:                  # surfaced via status(), not lost
            logger.exception("Live session failed")
            self.error = str(e) or type(e).__name__
        finally:
            with self._cond:
                point = self.aggregator.flush_if_idle(0.0, now=float("inf"))
                if point:
                    self._add_point(point)
                self.finished = True
                self._cond.notify_all()

    def stop(self, timeout: float = 60.0):
        if self.pipeline is not None:
            self.pipeline.request_stop()
        self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def transcript_session_id(self) -> str | None:
        session = getattr(self.pipeline, "session", None)
        return getattr(session, "id", None)

    def reload_profiles(self):
        if self.pipeline is not None:
            self.pipeline.reload_profiles()

    # ── called from the transcription thread ─────────────────────

    def _on_line(self, u: Utterance):
        with self._cond:
            self.lines.append(LiveLine(
                index=len(self.lines), at=datetime.now().isoformat(timespec="seconds"),
                offset_s=round(u.start_s, 1), speaker=u.speaker, label=self._label(u),
                attribution=u.attribution, text=u.text))
            point = self.aggregator.add(u)
            if point:
                self._add_point(point)
            self._cond.notify_all()

    def _add_point(self, p):
        self.points.append(LivePoint(
            id=len(self.points) + 1, at=datetime.now().isoformat(timespec="seconds"),
            offset_s=round(p.start_s, 1), end_offset_s=round(p.end_s, 1),
            speaker=p.speaker, label=p.label,
            attribution=p.attribution, text=p.text, context=p.context[-8:]))

    # ── called by agent tools ────────────────────────────────────

    def next_point(self, timeout_s: float = 30.0, speaker: str | None = None) -> LivePoint | None:
        """
        The oldest point not yet handed out (optionally from one speaker),
        waiting up to `timeout_s` for one. A speaker who stops talking has
        their point closed after `debate.point_pause_s`.
        """
        wanted = speaker.casefold() if speaker else None
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while True:
                for point in self.points:
                    if not point.delivered and (
                            wanted is None or (point.speaker or "").casefold() == wanted):
                        point.delivered = True
                        return point
                idle = self.aggregator.flush_if_idle(self.config.debate.point_pause_s)
                if idle:
                    self._add_point(idle)
                    continue
                remaining = deadline - time.monotonic()
                if self.finished or remaining <= 0:
                    return None
                self._cond.wait(timeout=min(remaining, 0.5))

    def behind_live_s(self, point: LivePoint) -> float | None:
        """How long ago (wall clock) the point finished being said."""
        start = _stream_start(self.pipeline)
        if start is None:
            return None
        said = start.timestamp() + point.end_offset_s
        return round(max(0.0, datetime.now().timestamp() - said), 1)

    def lines_since(self, cursor: int = 0, limit: int = 50) -> tuple[list[LiveLine], int]:
        with self._cond:
            chunk = self.lines[cursor:cursor + limit]
        return chunk, cursor + len(chunk)

    def status(self) -> dict:
        with self._cond:
            heard: dict[str, int] = {}
            for line in self.lines:
                heard[line.label] = heard.get(line.label, 0) + 1
            return {
                "source": self.source,
                "state": "running" if self.running else ("failed" if self.error else "finished"),
                "error": self.error,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "elapsed_s": round((datetime.now() - self.started_at).total_seconds()),
                "transcript_session": self.transcript_session_id,
                "lines": len(self.lines),
                "points_waiting": sum(not p.delivered for p in self.points),
                "points_total": len(self.points),
                "tracking": self.speakers or "everyone",
                "lines_by_voice": heard,
                # > 1.0 means transcription can't keep up with the stream
                "realtime_factor": round(getattr(getattr(self.pipeline, "stats", None),
                                                 "realtime_factor", 0.0), 2),
            }
