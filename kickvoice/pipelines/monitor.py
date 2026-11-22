"""
Live monitoring: who is speaking right now?

    capture → VAD → embed → compare with profiles → callback per detection
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import click

from kickvoice import console
from kickvoice.audio.capture import open_capture
from kickvoice.audio.vad import get_vad
from kickvoice.config import Config
from kickvoice.speakers.encoder import SpeakerEncoder
from kickvoice.speakers.identify import IdentifyResult, SpeakerIdentifier

logger = logging.getLogger(__name__)


@dataclass
class MonitorEvent:
    """One identification from the monitor."""
    result: IdentifyResult
    timestamp: datetime

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        if self.result.matched:
            return (f"[{ts}] {self.result.speaker_name} detected "
                    f"(confidence: {self.result.similarity:.2f})")
        return f"[{ts}] Unknown speaker (best: {self.result.similarity:.2f})"


EventCallback = Callable[[MonitorEvent], None]


def default_callback(event: MonitorEvent):
    """Print each detection, colour-coded by confidence."""
    if event.result.matched:
        colour = "green" if event.result.confidence == "high" else "yellow"
        console.echo(click.style(str(event), fg=colour))
    else:
        console.hint(str(event))


@dataclass
class MonitorStats:
    chunks_processed: int = 0
    segments_detected: int = 0
    identifications: int = 0
    matches: int = 0
    by_speaker: dict[str, int] = field(default_factory=dict)


class Monitor:
    """
    Real-time speaker identification.

        Monitor(stream_url="https://kick.com/somechannel").run()
    """

    def __init__(
        self,
        stream_url: str | None = None,
        file_path: str | None = None,
        config: Config | None = None,
        callback: EventCallback | None = None,
        *,
        capture=None,
        vad=None,
        encoder=None,
        store=None,
    ):
        self.config = config or Config()
        self.callback = callback or default_callback
        self.capture = capture if capture is not None else open_capture(
            stream_url, file_path, config=self.config)
        self.vad = vad if vad is not None else get_vad(
            use_silero=True, audio_cfg=self.config.audio)
        self.encoder = encoder if encoder is not None else SpeakerEncoder(config=self.config)
        self.identifier = SpeakerIdentifier(config=self.config, store=store)
        self.stats = MonitorStats()
        self._running = False

    def run(self):
        """Process audio until the stream ends or the user stops it."""
        profiles = self.identifier.store.list_profiles()
        if not profiles:
            console.warn("No speaker profiles enrolled.")
            console.hint("   Enroll one:        ./run.sh enroll NAME --file audio.wav")
            console.hint("   Or discover them:  ./run.sh extract-profiles --stream URL")
            return self.stats

        console.heading(f"\n🎙  Monitoring with {len(profiles)} enrolled speaker(s)")
        console.info(f"   Profiles: {', '.join(p.name for p in profiles)}")
        console.info(f"   Chunk: {self.config.audio.chunk_duration_s}s, "
                     f"Stride: {self.config.audio.chunk_stride_s}s")
        console.info(f"   Match threshold: {self.config.identify.match_threshold}")
        console.hint("   Press Ctrl+C to stop\n")

        self._running = True
        started = datetime.now()
        chunks = self.capture.capture_chunks()
        try:
            for chunk in chunks:
                if not self._running:
                    break
                self.stats.chunks_processed += 1
                self._process_chunk(chunk)
        except KeyboardInterrupt:
            console.echo("\n\n🛑 Monitoring stopped by user")
        finally:
            self._running = False
            chunks.close()
            self._print_stats(datetime.now() - started)
        return self.stats

    def _process_chunk(self, chunk: bytes):
        for segment in self.vad.detect_speech(chunk):
            self.stats.segments_detected += 1
            embedding = self.encoder.embed_segment(segment)
            result = self.identifier.identify(embedding, segment_duration_s=segment.duration_s)
            self.stats.identifications += 1
            if result.matched:
                self.stats.matches += 1
                self.stats.by_speaker[result.speaker_name] = (
                    self.stats.by_speaker.get(result.speaker_name, 0) + 1)
            self.callback(MonitorEvent(result=result, timestamp=datetime.now()))

    def stop(self):
        """Ask the monitor to stop (from a signal handler or another thread)."""
        self._running = False
        self.capture.stop()

    def _print_stats(self, elapsed: timedelta):
        s = self.stats
        console.heading(f"\n📊 Session Summary ({elapsed})")
        console.info(f"   Chunks processed:  {s.chunks_processed}")
        console.info(f"   Speech segments:   {s.segments_detected}")
        console.info(f"   Identifications:   {s.identifications}")
        console.info(f"   Matched speakers:  {s.matches}")
        if s.identifications:
            console.info(f"   Match rate:        {s.matches / s.identifications * 100:.1f}%")
        for name, count in sorted(s.by_speaker.items(), key=lambda kv: -kv[1]):
            console.info(f"     · {name}: {count}")
