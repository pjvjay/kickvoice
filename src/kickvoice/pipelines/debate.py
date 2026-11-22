"""
Live debate assistant: transcribe the stream, detect the opponent's points,
and show grounded counter-arguments while the debate is happening.

    TranscriptionPipeline (main thread, short blocks for low latency)
        → each line → PointAggregator: collects the opponent's words into a "point"
        → queue (newest points win if the model falls behind)
    worker thread
        → CounterEngine.suggest (retrieval + fast model + validation)
        → printed suggestion + JSONL log for review after the stream
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

from kickvoice import console
from kickvoice.config import Config
from kickvoice.console import fmt_clock
from kickvoice.pipelines.session import source_slug
from kickvoice.research.counter import CounterEngine, Suggestion
from kickvoice.research.store import AUTHORSHIP_LABELS
from kickvoice.transcripts.attribution import UNKNOWN
from kickvoice.transcripts.store import Utterance

logger = logging.getLogger(__name__)


@dataclass
class Point:
    text: str
    start_s: float
    context: list[str]


class PointAggregator:
    """
    Turn a stream of transcript lines into the opponent's "points".

    A point is ready when the opponent has said at least `min_words` and then
    someone else speaks or they pause; very long monologues are split at
    `max_words` so suggestions keep coming.
    """

    def __init__(
        self,
        is_opponent: Callable[[Utterance], bool],
        label: Callable[[Utterance], str],
        min_words: int = 20,
        max_words: int = 120,
        context_lines: int = 8,
    ):
        self.is_opponent = is_opponent
        self.label = label
        self.min_words = min_words
        self.max_words = max_words
        self.context: deque[str] = deque(maxlen=context_lines)
        self._buffer: list[Utterance] = []
        self._last_line_at = 0.0

    @property
    def buffered_words(self) -> int:
        return sum(len(u.text.split()) for u in self._buffer)

    def add(self, u: Utterance, now: float | None = None) -> Point | None:
        point = None
        if self.is_opponent(u):
            self._buffer.append(u)
            self._last_line_at = now if now is not None else time.monotonic()
            if self.buffered_words >= self.max_words:
                point = self._flush()
        elif self._buffer and self.buffered_words >= max(5, self.min_words // 2):
            point = self._flush()  # they stopped; someone else is talking
        elif self._buffer:
            self._buffer.clear()  # a short aside — not worth a response
        self.context.append(f"{self.label(u)}: {u.text}")
        return point

    def flush_if_idle(self, pause_s: float, now: float | None = None) -> Point | None:
        now = now if now is not None else time.monotonic()
        if self._buffer and self.buffered_words >= self.min_words and now - self._last_line_at >= pause_s:
            return self._flush()
        return None

    def _flush(self) -> Point:
        text = " ".join(u.text for u in self._buffer)
        point = Point(text, self._buffer[0].start_s, list(self.context))
        self._buffer.clear()
        return point


class DebateAssistant:
    """Glue between live transcript lines and the counter engine (thread-safe)."""

    def __init__(
        self,
        engine: CounterEngine,
        aggregator: PointAggregator,
        config: Config,
        log_path: Path | None = None,
        render: Callable[[Point, Suggestion], None] | None = None,
    ):
        self.engine = engine
        self.aggregator = aggregator
        self.dcfg = config.debate
        self.log_path = log_path
        self.render = render or render_suggestion
        self._queue: deque[Point] = deque()
        self._cond = threading.Condition()
        self._stop = False
        self.skipped = 0
        self.suggestions: list[Suggestion] = []
        self._worker = threading.Thread(target=self._loop, name="debate-assistant", daemon=True)

    # called from the transcription thread
    def on_utterance(self, u: Utterance):
        with _print_lock:
            _print_line(u, self.aggregator)
        with self._cond:
            point = self.aggregator.add(u)
            if point:
                self._enqueue(point)

    def start(self):
        self._worker.start()

    def stop(self, drain_timeout: float = 30.0):
        """Finish the opponent's last point (if any), then stop the worker."""
        with self._cond:
            point = self.aggregator.flush_if_idle(0.0, now=float("inf"))
            if point:
                self._enqueue(point)
            self._stop = True
            self._cond.notify_all()
        self._worker.join(timeout=drain_timeout)

    def _enqueue(self, point: Point):
        self._queue.append(point)
        while len(self._queue) > self.dcfg.max_queued_points:
            self._queue.popleft()  # stale: the debate has moved on
            self.skipped += 1
        self._cond.notify_all()

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._stop, timeout=0.5)
                if not self._queue:
                    idle_point = self.aggregator.flush_if_idle(self.dcfg.point_pause_s)
                    if idle_point:
                        self._queue.append(idle_point)
                if not self._queue:
                    if self._stop:
                        return
                    continue
                point = self._queue.popleft()
            suggestion = self.engine.suggest(point.text, point.context)
            self.suggestions.append(suggestion)
            with _print_lock:
                self.render(point, suggestion)
            self._log(point, suggestion)

    def _log(self, point: Point, s: Suggestion):
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "stream_offset_s": round(point.start_s, 1),
            "point": point.text,
            "summary": s.summary,
            "is_argument": s.is_argument,
            "counters": [
                {
                    "text": c.text,
                    "evidence": [
                        {
                            "ref": e.ref,
                            "quote": e.quote,
                            "url": e.url,
                            "date": e.published_at,
                            "authorship": e.authorship,
                        }
                        for e in c.evidence
                    ],
                    "notes": [n.text for n in c.notes],
                }
                for c in s.counters
            ],
            "contradictions": [
                {
                    "explanation": c.explanation,
                    "quote": c.evidence.quote,
                    "url": c.evidence.url,
                    "date": c.evidence.published_at,
                }
                for c in s.contradictions
            ],
            "latency_s": round(s.latency_s, 2),
            "model": s.model,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── terminal output ──────────────────────────────────────────────

_print_lock = threading.Lock()


def _print_line(u: Utterance, aggregator: PointAggregator):
    who = aggregator.label(u)
    colour = "cyan" if aggregator.is_opponent(u) else "bright_black"
    console.echo(click.style(f"[{fmt_clock(u.start_s)}] {who}: {u.text}", fg=colour))


def render_suggestion(point: Point, s: Suggestion):
    if not s.is_argument and not s.counters and not s.contradictions:
        if s.summary.startswith("(no suggestion"):
            console.hint(f"   {s.summary}")
        return
    bar = click.style("│", fg="magenta")
    console.echo(click.style(f"\n┌─ 💡 {s.summary or point.text[:90]}", fg="magenta", bold=True))
    for c in s.contradictions:
        e = c.evidence
        console.echo(f"{bar} {click.style('⚠ CONTRADICTION', fg='red', bold=True)}: {c.explanation}")
        console.echo(
            f'{bar}    "{e.quote}" — {(e.published_at or "undated")[:10]}, '
            f"{AUTHORSHIP_LABELS.get(e.authorship, e.authorship)}"
        )
        if e.url:
            console.echo(f"{bar}    {click.style(e.url, fg='blue')}")
    for i, c in enumerate(s.counters, 1):
        tag = "" if c.sourced else click.style(" (reasoning, no source)", fg="bright_black")
        console.echo(f"{bar} {click.style(f'↩ {i}.', fg='green', bold=True)} {c.text}{tag}")
        for e in c.evidence[:2]:
            console.echo(
                f'{bar}      └ "{e.quote[:140]}" ({(e.published_at or "undated")[:10]}'
                f"{', ' + e.url if e.url else ''})"
            )
        for n in c.notes[:1]:
            console.echo(f"{bar}      └ your notes: {n.text[:140]}")
    console.echo(click.style(f"└─ {s.latency_s:.1f}s · {s.model}", fg="magenta"))


def debate_log_path(config: Config, source: str) -> Path:
    return Path(config.data_dir) / "debates" / f"{datetime.now():%Y%m%d-%H%M%S}_{source_slug(source)}.jsonl"


def opponent_matcher(
    voice_profile: str | None, me: str | None = None, any_speaker: bool = False
) -> Callable[[Utterance], bool]:
    """Which live lines count as the opponent's."""
    vp = (voice_profile or "").casefold()
    mine = (me or "").casefold()

    def is_opponent(u: Utterance) -> bool:
        speaker = (u.speaker or "").casefold()
        if any_speaker:
            return not (mine and speaker == mine)
        return bool(vp) and speaker == vp and u.attribution != UNKNOWN

    return is_opponent
