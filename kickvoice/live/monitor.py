"""
Live monitoring pipeline.

Ties together stream capture → VAD → embedding → identification
into a single real-time processing loop.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Callable

from kickvoice.config import Config
from kickvoice.live.capture import StreamCapture, FileCapture
from kickvoice.live.detect import get_vad, SpeechSegment
from kickvoice.models.ecapa import SpeakerEncoder
from kickvoice.live.identify import SpeakerIdentifier, IdentifyResult

logger = logging.getLogger(__name__)


class MonitorEvent:
    """An identification event from the monitor."""

    def __init__(self, result: IdentifyResult, timestamp: datetime):
        self.result = result
        self.timestamp = timestamp

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        if self.result.matched:
            return (
                f"[{ts}] {self.result.speaker_name} detected "
                f"(confidence: {self.result.similarity:.2f})"
            )
        return f"[{ts}] Unknown speaker (best: {self.result.similarity:.2f})"


# Type for callback functions
EventCallback = Callable[[MonitorEvent], None]


def default_callback(event: MonitorEvent):
    """Default callback: print to stdout."""
    if event.result.matched:
        color = "\033[92m" if event.result.confidence == "high" else "\033[93m"
        reset = "\033[0m"
        print(f"{color}{event}{reset}")
    else:
        print(f"\033[90m{event}\033[0m")


class Monitor:
    """
    Main monitoring pipeline.

    Usage:
        monitor = Monitor(stream_url="https://kick.com/somechannel")
        monitor.run()  # blocks, prints identifications
    """

    def __init__(
        self,
        stream_url: str | None = None,
        file_path: str | None = None,
        config: Config | None = None,
        callback: EventCallback | None = None,
    ):
        if not stream_url and not file_path:
            raise ValueError("Provide either stream_url or file_path")

        self.config = config or Config()
        self.callback = callback or default_callback

        # Initialize pipeline components
        if stream_url:
            self.capture = StreamCapture(stream_url, config=self.config)
        else:
            self.capture = FileCapture(file_path, config=self.config)

        self.vad = get_vad(use_silero=True, audio_cfg=self.config.audio)
        self.encoder = SpeakerEncoder(config=self.config)
        self.identifier = SpeakerIdentifier(config=self.config)

        # State
        self._running = False
        self._stats = {
            "chunks_processed": 0,
            "segments_detected": 0,
            "identifications": 0,
            "matches": 0,
        }

    def run(self):
        """
        Start the monitoring loop. Blocks until stopped or stream ends.
        """
        n_profiles = self.identifier.store.count
        if n_profiles == 0:
            logger.warning(
                "No speaker profiles enrolled! "
                "Run `kickvoice enroll` first."
            )
            print(
                "\n⚠  No speaker profiles enrolled.\n"
                "   Run: kickvoice enroll --name 'Name' --file audio.wav\n"
            )
            return

        print(f"\n🎙  Monitoring stream with {n_profiles} enrolled speaker(s)")
        print(f"   Profiles: {', '.join(p.name for p in self.identifier.store.list_profiles())}")
        print(f"   Chunk: {self.config.audio.chunk_duration_s}s, "
              f"Stride: {self.config.audio.chunk_stride_s}s")
        print(f"   Match threshold: {self.config.identify.match_threshold}")
        print("   Press Ctrl+C to stop\n")

        self._running = True
        start_time = datetime.now()

        try:
            for chunk_bytes in self.capture.capture_chunks():
                if not self._running:
                    break

                self._stats["chunks_processed"] += 1

                # Step 1: VAD — find speech in this chunk
                segments = self.vad.detect_speech(chunk_bytes)
                if not segments:
                    continue

                self._stats["segments_detected"] += len(segments)

                # Step 2: Extract embeddings for each speech segment
                for segment in segments:
                    embedding = self.encoder.embed_segment(segment)

                    # Step 3: Identify the speaker
                    result = self.identifier.identify(
                        embedding,
                        segment_duration_s=segment.duration_s,
                    )

                    self._stats["identifications"] += 1
                    if result.matched:
                        self._stats["matches"] += 1

                    # Step 4: Fire the callback
                    event = MonitorEvent(
                        result=result,
                        timestamp=datetime.now(),
                    )
                    self.callback(event)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
        finally:
            self._running = False
            elapsed = datetime.now() - start_time
            self._print_stats(elapsed)

    def stop(self):
        """Signal the monitor to stop."""
        self._running = False
        if hasattr(self.capture, 'stop'):
            self.capture.stop()

    def _print_stats(self, elapsed: timedelta):
        """Print summary statistics."""
        s = self._stats
        print(f"\n📊 Session Summary ({elapsed})")
        print(f"   Chunks processed:  {s['chunks_processed']}")
        print(f"   Speech segments:   {s['segments_detected']}")
        print(f"   Identifications:   {s['identifications']}")
        print(f"   Matched speakers:  {s['matches']}")
        if s["identifications"] > 0:
            rate = s["matches"] / s["identifications"] * 100
            print(f"   Match rate:        {rate:.1f}%")
