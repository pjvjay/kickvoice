"""
Profile extraction: discover the distinct voices on a stream.

Pipeline:
    Stream/file → non-overlapping 30s blocks → VAD → short sliding windows (2s)
      → ECAPA embeddings → agglomerative clustering → one candidate per voice
      → match each candidate against known profiles → save session for review
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from kickvoice import console
from kickvoice.audio.capture import open_capture
from kickvoice.audio.vad import get_vad
from kickvoice.config import Config
from kickvoice.console import fmt_clock, fmt_duration
from kickvoice.pipelines.session import ExtractionSession
from kickvoice.speakers.cluster import (
    SpeakerCluster,
    cluster_speakers,
    make_windows,
    representative_windows,
)
from kickvoice.speakers.encoder import SpeakerEncoder
from kickvoice.speakers.identify import SpeakerIdentifier
from kickvoice.speakers.profiles import ProfileStore

logger = logging.getLogger(__name__)


@dataclass
class _Segment:
    start: int          # absolute sample offset in the capture
    pcm: bytes


class ProfileExtractor:
    """
    Listen to a stream (or file), find each distinct speaker, and save them as
    candidates for the user to approve.

    Usage:
        ex = ProfileExtractor(stream_url="https://kick.com/somechannel")
        session = ex.run(max_duration_s=300)   # None if nobody was found
    """

    def __init__(
        self,
        stream_url: str | None = None,
        file_path: str | None = None,
        config: Config | None = None,
        *,
        capture=None,
        vad=None,
        encoder=None,
        store=None,
        out: Callable[[str], None] = console.echo,
    ):
        if not (stream_url or file_path or capture):
            raise ValueError("Provide stream_url, file_path, or capture")
        self.config = config or Config()
        self.source = stream_url or str(file_path or "capture")
        self.out = out

        # Any component can be replaced (tests inject fakes; no model needed).
        if capture is None:
            capture = open_capture(stream_url, file_path, config=self.config)
        if vad is None:
            vad = get_vad(use_silero=True, audio_cfg=self.config.audio)
        if encoder is None:
            encoder = SpeakerEncoder(config=self.config)
        if store is None:
            store = ProfileStore(config=self.config)
        self.capture, self.vad, self.encoder, self.store = capture, vad, encoder, store

        self.sr = self.config.audio.sample_rate
        self._segments: list[_Segment] = []
        self._embeddings: list[np.ndarray] = []
        self._spans: list[tuple[int, int]] = []     # absolute window spans
        self._window_seg: list[int] = []            # window → segment index
        self._offset = 0                            # samples captured so far
        self._speech_samples = 0

    # ── capture + embed ──────────────────────────────────────────

    @property
    def captured_seconds(self) -> float:
        return self._offset / self.sr

    def _process_chunk(self, chunk: bytes):
        ex = self.config.extract
        win = int(ex.window_s * self.sr)
        hop = int(ex.window_hop_s * self.sr)
        min_len = int(ex.min_window_s * self.sr)

        for seg in self.vad.detect_speech(chunk):
            seg_idx = len(self._segments)
            self._segments.append(_Segment(self._offset + seg.start_sample, seg.pcm_bytes))
            self._speech_samples += seg.end_sample - seg.start_sample
            for s, e in make_windows(seg.start_sample, seg.end_sample, win, hop, min_len):
                emb = self.encoder.embed_pcm(chunk[s * 2:e * 2])
                self._embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))
                self._spans.append((self._offset + s, self._offset + e))
                self._window_seg.append(seg_idx)

        self._offset += len(chunk) // 2

    def _capture(self, max_duration_s: float | None):
        block = self.config.extract.chunk_duration_s
        chunks = self.capture.capture_chunks(chunk_duration_s=block, chunk_stride_s=block)
        try:
            for chunk in chunks:
                self._process_chunk(chunk)
                self.out(
                    f"   [{fmt_clock(self.captured_seconds)}] "
                    f"speech so far: {fmt_duration(self._speech_samples / self.sr)}, "
                    f"{len(self._embeddings)} voice windows"
                )
                if max_duration_s and self.captured_seconds >= max_duration_s:
                    break
        except KeyboardInterrupt:
            pass   # Ctrl+C = stop listening; analyse what we have
        finally:
            chunks.close()   # stops streamlink/ffmpeg
        self.out(f"\n⏹  Stopped listening after {fmt_duration(self.captured_seconds)}.")

    # ── cluster + match + save ───────────────────────────────────

    def _clip_pcm(self, window_ids: list[int]) -> bytes:
        gap = b"\x00\x00" * int(0.3 * self.sr)   # short silence between pieces
        parts = []
        for i in window_ids:
            seg = self._segments[self._window_seg[i]]
            s, e = self._spans[i]
            parts.append(seg.pcm[(s - seg.start) * 2:(e - seg.start) * 2])
        return gap.join(parts)

    def _best_match(self, centroid: np.ndarray):
        if self.store.count == 0:
            return None
        return SpeakerIdentifier(config=self.config, store=self.store).identify(centroid)

    def run(self, max_duration_s: float | None = None) -> ExtractionSession | None:
        ex = self.config.extract
        if max_duration_s is None:
            max_duration_s = ex.capture_duration_s

        if max_duration_s:
            limit = f"up to {fmt_duration(max_duration_s)}"
        else:
            limit = "until the stream ends" if "://" in self.source else "whole file"
        self.out(f"\n🎧 Listening to {self.source} ({limit}; Ctrl+C to stop early)")
        self.out(f"   Known profiles: {self.store.count}\n")

        self._capture(max_duration_s)

        if not self._embeddings:
            self.out("\n⚠  No usable speech was captured — nothing to extract.")
            return None

        self.out(f"🔬 Grouping {len(self._embeddings)} voice windows by speaker…")
        emb = np.stack(self._embeddings)
        clusters = cluster_speakers(
            emb, self._spans, self.sr,
            distance_threshold=ex.cluster_distance_threshold,
            min_speech_s=ex.min_speaker_speech_s,
            max_windows=ex.max_cluster_windows,
        )
        if not clusters:
            self.out(
                f"\n⚠  No single voice had at least {fmt_duration(ex.min_speaker_speech_s)} "
                f"of speech (captured {fmt_duration(self._speech_samples / self.sr)} in total).\n"
                "   Listen for longer (--duration) or lower --min-speech."
            )
            return None

        session = ExtractionSession.create(
            self.config.extractions_dir, self.source, self.config.embedding.model_source
        )
        session.captured_seconds = self.captured_seconds
        session.speech_seconds = self._speech_samples / self.sr
        session.settings = {
            "cluster_distance_threshold": ex.cluster_distance_threshold,
            "min_speaker_speech_s": ex.min_speaker_speech_s,
            "window_s": ex.window_s,
            "window_hop_s": ex.window_hop_s,
            "match_threshold": self.config.identify.match_threshold,
        }
        for cluster in clusters:
            self._add_candidate(session, cluster, emb)
        session.save()

        self._print_summary(session)
        return session

    def _add_candidate(self, session: ExtractionSession, cluster: SpeakerCluster, emb):
        match = self._best_match(cluster.centroid)
        clip_ids = representative_windows(
            cluster, emb, self._spans, self.sr, self.config.extract.sample_clip_s
        )
        session.add_candidate(
            embedding=cluster.centroid,
            sample_pcm=self._clip_pcm(clip_ids),
            sample_rate=self.sr,
            speech_seconds=round(cluster.speech_seconds, 2),
            n_windows=cluster.n_windows,
            first_heard_s=round(cluster.first_sample / self.sr, 2),
            cohesion=round(cluster.cohesion, 4),
            best_match=match.candidates[0][0] if match else None,
            best_similarity=round(match.similarity, 4) if match else 0.0,
            matched=bool(match and match.matched),
        )

    def _print_summary(self, session: ExtractionSession):
        self.out(
            f"\n🗣  Found {len(session.candidates)} speaker(s) in "
            f"{fmt_duration(session.captured_seconds)} of audio "
            f"({fmt_duration(session.speech_seconds)} of speech)\n"
        )
        self.out(f"   {'#':<3}{'Speech':<10}{'First heard':<13}Best match")
        for i, c in enumerate(session.candidates, 1):
            if c.matched:
                match = f"✅ {c.best_match} ({c.best_similarity:.2f})"
            elif c.best_match:
                match = f"🆕 new voice (closest: {c.best_match} {c.best_similarity:.2f})"
            else:
                match = "🆕 new voice"
            self.out(
                f"   {i:<3}{fmt_duration(c.speech_seconds):<10}"
                f"{fmt_clock(c.first_heard_s):<13}{match}"
            )
        self.out(f"\n   Session saved: {session.path}")
