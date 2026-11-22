"""
Speaker-attributed transcription: who said what, and when.

    capture (30s blocks, cut at pauses)
      → VAD (speech regions)
      → ASR (timestamped words; words outside speech dropped)
      → speaker windows scored against enrolled profiles
      → each word labelled → words grouped into turns
      → each turn re-scored from its own audio → provisional label (printed live)
      → TranscriptStore (SQLite)
    at the end of the session:
      → lines grouped by voice; each profile claims at most one voice group
        → final confident / probable / unknown labels (see transcripts/refine.py)

Turn embeddings are stored too, so transcripts can be re-attributed later
(after enrolling someone new) without keeping any audio.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import click
import numpy as np

from kickvoice import console
from kickvoice.audio.capture import open_capture
from kickvoice.audio.ffmpeg import pcm_to_numpy
from kickvoice.audio.vad import get_vad
from kickvoice.config import Config
from kickvoice.console import fmt_clock, fmt_duration
from kickvoice.pipelines.session import source_slug
from kickvoice.speakers.cluster import make_windows
from kickvoice.speakers.encoder import SpeakerEncoder
from kickvoice.speakers.profiles import ProfileStore, normalise
from kickvoice.transcripts.asr import FasterWhisperRecognizer, SpeechRecognizer
from kickvoice.transcripts.attribution import (
    CONFIDENT, PROBABLE, UNKNOWN, ScoredWindow, Turn, finalise_turn,
    group_turns, label_words, speech_cut_point, words_in_speech,
)
from kickvoice.transcripts.refine import RefineResult, refine_session
from kickvoice.transcripts.store import TranscriptSession, TranscriptStore, Utterance

logger = logging.getLogger(__name__)

UtteranceCallback = Callable[[Utterance], None]


@dataclass
class TranscriptionStats:
    audio_seconds: float = 0.0
    processing_seconds: float = 0.0
    utterances: int = 0
    dropped_unattributed: int = 0
    refine: RefineResult = field(default_factory=RefineResult)

    @property
    def realtime_factor(self) -> float:
        """Processing time ÷ audio time (< 1 keeps up with a live stream)."""
        return self.processing_seconds / self.audio_seconds if self.audio_seconds else 0.0


def speaker_label(u: Utterance) -> str:
    """'xQc', 'xQc?' (probable), '(voice 3)' or '(unknown)'."""
    if u.attribution == CONFIDENT:
        return u.speaker
    if u.attribution == PROBABLE:
        return f"{u.speaker}?"
    return f"(voice {u.voice_group})" if u.voice_group else "(unknown)"


def print_utterance(u: Utterance):
    """One coloured line per turn."""
    colour = {CONFIDENT: "green", PROBABLE: "yellow"}.get(u.attribution, "bright_black")
    who = click.style(speaker_label(u), fg=colour, bold=u.attribution == CONFIDENT)
    console.echo(f"[{fmt_clock(u.start_s)}] {who}: {u.text}")


class TranscriptionPipeline:
    """
        TranscriptionPipeline(stream_url="https://kick.com/somechannel").run(600)

    Every component is injectable (tests use fakes; no models needed).
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
        recognizer: SpeechRecognizer | None = None,
        profiles: ProfileStore | None = None,
        transcripts: TranscriptStore | None = None,
        on_utterance: UtteranceCallback | None = print_utterance,
        out: Callable[[str], None] = console.echo,
    ):
        self.config = config or Config()
        self.cfg = self.config.transcribe
        self.sr = self.config.audio.sample_rate
        self.source = stream_url or str(file_path or "capture")
        self.capture = capture if capture is not None else open_capture(
            stream_url, file_path, config=self.config)
        self.vad = vad if vad is not None else get_vad(use_silero=True, audio_cfg=self.config.audio)
        self.encoder = encoder if encoder is not None else SpeakerEncoder(config=self.config)
        self.recognizer = recognizer if recognizer is not None else FasterWhisperRecognizer(self.config)
        self.profiles = profiles if profiles is not None else ProfileStore(config=self.config)
        self.transcripts = (transcripts if transcripts is not None
                            else TranscriptStore(self.config.transcripts_db))
        self.on_utterance = on_utterance
        self.out = out

        # Voiceprints are fixed for the run; new enrolments apply via `reattribute`.
        self.names, self.matrix = self.profiles.embedding_matrix()
        self.stats = TranscriptionStats()
        self.session: TranscriptSession | None = None
        self._warned_slow = False

    # ── main loop ────────────────────────────────────────────────

    def run(self, max_duration_s: float | None = None) -> TranscriptSession:
        """Transcribe until the source ends, the duration is reached, or Ctrl+C."""
        load = getattr(self.recognizer, "load", None)
        if load is not None:
            self.out(f"Loading speech recognition model ({self.recognizer.model_name})…")
            load()          # fail fast, before a session exists or capture starts
        self.session = self.transcripts.create_session(
            session_id=f"{time.strftime('%Y%m%d-%H%M%S')}_{source_slug(self.source)}",
            source=self.source,
            asr_model=self.recognizer.model_name,
            embedding_model=self.config.embedding.model_source,
            settings={
                "match_threshold": self.config.identify.match_threshold,
                "high_confidence_threshold": self.config.identify.high_confidence_threshold,
                "min_confident_turn_s": self.cfg.min_confident_turn_s,
                "profiles": self.names,
            },
        )
        limit = (f"up to {fmt_duration(max_duration_s)}" if max_duration_s
                 else ("until the stream ends" if "://" in self.source else "whole file"))
        self.out(f"\n📝 Transcribing {self.source} ({limit}; Ctrl+C to stop)")
        if self.names:
            self.out(f"   Recognising: {', '.join(self.names)}")
        else:
            self.out("   No profiles enrolled — every speaker will be 'unknown' "
                     "(run `transcripts reattribute` after enrolling).")
        self.out(f"   Session: {self.session.id}\n")

        block_s = self.cfg.chunk_duration_s
        chunks = self.capture.capture_chunks(chunk_duration_s=block_s, chunk_stride_s=block_s)
        carry, carry_offset, consumed = b"", 0, 0
        try:
            for block in chunks:
                piece, piece_offset = carry + block, carry_offset
                consumed += len(block) // 2
                cut = self._process(piece, piece_offset, final=False)
                carry = piece[cut * 2:]
                carry_offset = piece_offset + cut
                if max_duration_s and consumed / self.sr >= max_duration_s:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            chunks.close()
        try:
            if carry:
                self._process(carry, carry_offset, final=True)
        except KeyboardInterrupt:
            pass

        self.transcripts.finish_session(self.session.id, consumed / self.sr)
        self.session.captured_seconds = consumed / self.sr
        # Final say: attribute using the whole session (see transcripts/refine.py).
        self.stats.refine = refine_session(
            self.transcripts, self.session.id, self.names, self.matrix, self.config)
        if not self.cfg.keep_unattributed:
            self._drop_unknown_after_refine()
        self._print_summary()
        return self.session

    def _drop_unknown_after_refine(self):
        """Refinement can demote lines to unknown; honour --drop-unknown for those too."""
        unknown = [u.id for u in self.transcripts.utterances(self.session.id)
                   if u.attribution == UNKNOWN]
        self.transcripts.delete_utterances(unknown)
        self.stats.dropped_unattributed += len(unknown)
        self.stats.utterances -= len(unknown)

    # ── one block ────────────────────────────────────────────────

    def _process(self, pcm: bytes, offset: int, final: bool) -> int:
        """
        Transcribe `pcm` (which starts `offset` samples into the capture).
        Returns how many samples were consumed; the rest is carried over so
        an utterance running past the block edge is transcribed whole.
        """
        started = time.monotonic()
        n = len(pcm) // 2
        cut = n
        try:
            segments = self.vad.detect_speech(pcm, min_duration_s=self.cfg.min_speech_s)
            speech = [(s.start_sample, s.end_sample) for s in segments]
            if not final:
                point = speech_cut_point(speech, n, edge_tolerance=int(0.3 * self.sr))
                # Never carry more than one block, or a monologue could grow forever.
                if point is not None and n - point <= int(self.cfg.chunk_duration_s * self.sr):
                    cut = point
                    speech = [(s, e) for s, e in speech if e <= cut]
            if speech:
                for utterance in self._transcribe(pcm[:cut * 2], offset, speech):
                    self._emit(utterance)
        finally:
            # Counted even if Ctrl+C lands mid-block, so the summary stays true.
            self.stats.audio_seconds += cut / self.sr
            self.stats.processing_seconds += time.monotonic() - started
        self._warn_if_slow()
        return cut

    def _transcribe(self, pcm: bytes, offset: int, speech: list[tuple[int, int]]) -> list[Utterance]:
        sr = self.sr
        speech_s = [(s / sr, e / sr) for s, e in speech]
        words = self.recognizer.transcribe(pcm_to_numpy(pcm), sr)
        words = words_in_speech(words, speech_s)
        if not words:
            return []

        windows = self._score_windows(pcm, speech)
        labelled = label_words(words, windows, self.names, self.config.identify.match_threshold)
        # Same "different voice" cut-off extract-profiles clusters with.
        turns = group_turns(labelled, self.cfg.max_pause_s,
                            change_distance=self.config.extract.cluster_distance_threshold)

        utterances = []
        for turn in turns:
            self._finalise(turn, pcm)
            if not turn.text:
                continue
            utterances.append(Utterance(
                session_id=self.session.id,
                start_s=offset / sr + turn.start,
                end_s=offset / sr + turn.end,
                text=turn.text,
                speaker=turn.final_speaker,
                similarity=turn.similarity,
                attribution=turn.attribution,
                asr_confidence=turn.asr_confidence,
                contested=turn.contested,
                embedding=turn.embedding,
            ))
        return utterances

    def _score_windows(self, pcm: bytes, speech: list[tuple[int, int]]) -> list[ScoredWindow]:
        """
        Embed short windows across the speech. Scored against profiles when
        there are any; the embeddings alone still detect voice changes.
        """
        sr = self.sr
        windows = []
        for s, e in speech:
            for ws, we in make_windows(s, e, int(self.cfg.window_s * sr),
                                       int(self.cfg.window_hop_s * sr),
                                       int(self.cfg.min_window_s * sr)):
                emb = normalise(self.encoder.embed_pcm(pcm[ws * 2:we * 2]))
                scores = self.matrix @ emb if self.names else np.zeros(0)
                windows.append(ScoredWindow(ws / sr, we / sr, scores, emb))
        return windows

    def _finalise(self, turn: Turn, pcm: bytes):
        sr = self.sr
        ident = self.config.identify
        scores = embedding = None
        if turn.duration >= self.cfg.min_window_s:
            s, e = int(turn.start * sr), int(turn.end * sr)
            embedding = normalise(self.encoder.embed_pcm(pcm[s * 2:e * 2]))
            if self.names:
                scores = self.matrix @ embedding
        finalise_turn(turn, scores, self.names, ident.match_threshold,
                      ident.high_confidence_threshold, self.cfg.min_confident_turn_s,
                      turn_embedding=embedding)

    # ── output ───────────────────────────────────────────────────

    def _emit(self, u: Utterance):
        # Unknown lines are stored even with --drop-unknown: the whole-session
        # check may still recognise them. Leftover unknowns are dropped after.
        self.transcripts.add_utterances([u])
        self.stats.utterances += 1
        if self.on_utterance:
            self.on_utterance(u)

    def _warn_if_slow(self):
        if (self._warned_slow or "://" not in self.source
                or self.stats.audio_seconds < 60 or self.stats.realtime_factor <= 1.0):
            return
        self._warned_slow = True
        console.warn(
            f"Transcription is running slower than the stream "
            f"({self.stats.realtime_factor:.1f}× real time) and will fall behind. "
            "Try a smaller --model, or a GPU (transcribe.asr_device = \"cuda\").")

    def _print_summary(self):
        s = self.stats
        final = self.transcripts.utterances(self.session.id)
        levels = {lvl: sum(u.attribution == lvl for u in final)
                  for lvl in (CONFIDENT, PROBABLE, UNKNOWN)}
        speech: dict[str, float] = {}
        for u in final:
            if u.speaker:
                speech[u.speaker] = speech.get(u.speaker, 0.0) + u.duration

        self.out(f"\n📊 Transcribed {fmt_duration(self.session.captured_seconds)} of audio "
                 f"→ {len(final)} line(s) [{s.realtime_factor:.2f}× real time]")
        self.out(f"   {s.refine.voice_groups} distinct voice(s), "
                 f"{s.refine.matched_groups} matched to a profile"
                 + (f"; whole-session check revised {s.refine.changed} live label(s)"
                    if s.refine.changed else ""))
        self.out(f"   confident: {levels[CONFIDENT]} · probable: {levels[PROBABLE]}"
                 f" · unknown: {levels[UNKNOWN]}"
                 + (f" (+{s.dropped_unattributed} unknown not stored)"
                    if s.dropped_unattributed else ""))
        for name, secs in sorted(speech.items(), key=lambda kv: -kv[1]):
            self.out(f"     · {name}: {fmt_duration(secs)}")
        self.out(f"\n   Session: {self.session.id}")
        self.out(f"   Read it:  ./run.sh transcripts show {self.session.id}")
