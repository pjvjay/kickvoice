"""
Interactive approval of speakers discovered by `extract-profiles`.

For each pending candidate the user hears a clip (optional) and chooses:
  a  accept the suggested match → add this audio to that known profile
  n  create a new profile (asks for a name)
  m  merge into some other existing profile
  s  skip (discard this voice)
  p  play the review clip
  q  quit — remaining candidates stay pending for `kickvoice review`

Nothing is written to the profile store without an explicit choice; the
default (Enter) is always "skip".
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from kickvoice import console
from kickvoice.config import Config
from kickvoice.console import fmt_clock, fmt_duration
from kickvoice.pipelines.session import (
    CREATED, MERGED, SKIPPED, Candidate, ExtractionSession, channel_name,
)
from kickvoice.speakers.identify import SpeakerIdentifier
from kickvoice.speakers.profiles import ProfileStore

InputFn = Callable[[str], str]
OutFn = Callable[[str], None]
PlayerFn = Callable[[Path], bool]


class _Quit(Exception):
    pass


@dataclass
class ReviewSummary:
    created: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    skipped: int = 0
    remaining: int = 0


def ffplay_player(path: Path) -> bool:
    """Play a clip with ffplay (ships with ffmpeg). False if unavailable."""
    if not shutil.which("ffplay"):
        return False
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
            check=False,
        )
    except KeyboardInterrupt:
        pass   # Ctrl+C stops playback, not the review
    return True


class Reviewer:
    _summary: ReviewSummary

    def __init__(
        self,
        session: ExtractionSession,
        config: Config | None = None,
        *,
        store=None,
        input_fn: InputFn = input,
        out: OutFn = console.echo,
        player: PlayerFn | None = ffplay_player,
    ):
        self.session = session
        self.config = config or Config()
        self.store = store if store is not None else ProfileStore(config=self.config)
        # Shares the store, so a profile approved for speaker 1 is matched
        # against speaker 2 (catches one voice split into two clusters).
        self.identifier = SpeakerIdentifier(config=self.config, store=self.store)
        self.input_fn = input_fn
        self.out = out
        self.player = player

    # ── entry point ──────────────────────────────────────────────

    def run(self) -> ReviewSummary:
        summary = self._summary = ReviewSummary()
        model = self.config.embedding.model_source
        if self.session.embedding_model != model:
            self.out(
                f"❌ Session was extracted with '{self.session.embedding_model}' but the "
                f"configured model is '{model}'. Voiceprints from different models "
                "can't be compared — re-run extract-profiles."
            )
            summary.remaining = len(self.session.pending)
            return summary

        pending = self.session.pending
        total = len(self.session.candidates)
        self.out(f"\n📝 Review: {len(pending)} speaker(s) awaiting approval "
                 f"from {self.session.source}")
        try:
            for cand in pending:
                self._review_one(cand, total)
        except _Quit:
            pass

        summary.remaining = len(self.session.pending)
        self._print_summary(summary)
        return summary

    # ── per-candidate flow ───────────────────────────────────────

    def _review_one(self, cand: Candidate, total: int):
        emb = self.session.embedding(cand)
        match = self.identifier.identify(emb) if self.store.count else None
        suggested = match.speaker_name if match and match.matched else None

        idx = int(cand.id.rsplit("_", 1)[-1])
        self.out("\n" + "─" * 60)
        self.out(
            f"Speaker {idx} of {total} · {fmt_duration(cand.speech_seconds)} of speech · "
            f"first heard at {fmt_clock(cand.first_heard_s)}"
        )
        self.out(f"  Review clip: {self.session.sample_path(cand)}")
        if suggested:
            self.out(f"  ✅ Matches known profile: {suggested} "
                     f"(similarity {match.similarity:.2f}, {match.confidence})")
            others = [f"{n} {s:.2f}" for n, s in match.candidates[1:]]
            if others:
                self.out(f"     Next closest: {', '.join(others)}")
        elif match:
            self.out(f"  🆕 No match among {self.store.count} known profile(s) "
                     f"(closest: {match.candidates[0][0]} {match.similarity:.2f})")
        else:
            self.out("  🆕 New voice (no profiles enrolled yet)")

        while True:
            self.out("")
            if suggested:
                self.out(f"  [a] Yes, this is {suggested} — add this audio to their profile")
                self.out("  [n] No, it's someone else — create a new profile")
            else:
                self.out("  [n] Create a new profile for this voice")
            self.out("  [m] Merge into an existing profile")
            self.out("  [p] Play clip   [s] Skip   [q] Quit (finish later)")
            choice = self._ask("  Choice [s]: ").lower() or "s"

            if choice == "p":
                self._play(cand)
            elif choice == "s":
                return self._decide(cand, SKIPPED, None)
            elif choice == "q":
                raise _Quit
            elif choice == "a" and suggested:
                return self._merge(cand, suggested)
            elif choice == "n":
                if self._create(cand):
                    return
            elif choice == "m":
                target = self._pick_existing(match)
                if target:
                    return self._merge(cand, target)
            else:
                self.out("  ? Unrecognised choice")

    # ── actions ──────────────────────────────────────────────────

    def _create(self, cand: Candidate) -> bool:
        default = self._default_name(cand)
        hint = f" [{default}]" if default else ""
        while True:
            name = self._ask(f"  Name for this speaker{hint} (blank = back): ").strip()
            if not name:
                if not default:
                    return False
                name = default
                default = None   # a second blank goes back instead of reusing it
                hint = ""
            existing = self.store.find_profile(name)
            if existing:
                yn = self._ask(
                    f"  A profile named '{existing.name}' already exists. "
                    "Add this audio to it? [y/N]: "
                ).strip().lower()
                if yn == "y":
                    self._merge(cand, existing.name)
                    return True
                continue
            self.store.add_profile(
                name=name,
                embedding=self.session.embedding(cand),
                source_urls=[self.session.source],
                n_segments=cand.n_windows,
                notes=self._note(cand),
                speech_seconds=cand.speech_seconds,
                embedding_model=self.session.embedding_model,
            )
            self.out(f"  ✅ Created profile '{name}'")
            self._decide(cand, CREATED, name)
            return True

    def _merge(self, cand: Candidate, name: str):
        self.store.merge_into_profile(
            name=name,
            embedding=self.session.embedding(cand),
            speech_seconds=cand.speech_seconds,
            n_segments=cand.n_windows,
            source_urls=[self.session.source],
            note=self._note(cand),
        )
        self.out(f"  ✅ Added {fmt_duration(cand.speech_seconds)} of speech to '{name}'")
        self._decide(cand, MERGED, name)

    def _pick_existing(self, match) -> str | None:
        if not self.store.count:
            self.out("  No existing profiles to merge into.")
            return None
        if match:
            self.out("  Closest profiles: " +
                     ", ".join(f"{n} ({s:.2f})" for n, s in match.candidates))
        name = self._ask("  Existing profile name (blank = back): ").strip()
        if not name:
            return None
        profile = self.store.find_profile(name)
        if not profile:
            self.out(f"  ? No profile named '{name}'")
            return None
        return profile.name

    def _play(self, cand: Candidate):
        path = self.session.sample_path(cand)
        if not (self.player and self.player(path)):
            self.out(f"  (No audio player found — open this file to listen: {path})")

    def _decide(self, cand: Candidate, status: str, profile_name: str | None):
        cand.status = status
        cand.profile_name = profile_name
        cand.reviewed_at = datetime.now().isoformat()
        self.session.save()   # persist every decision; safe to quit anytime
        if status == CREATED:
            self._summary.created.append(profile_name)
        elif status == MERGED:
            self._summary.merged.append(profile_name)
        elif status == SKIPPED:
            self._summary.skipped += 1

    # ── helpers ──────────────────────────────────────────────────

    def _ask(self, prompt: str) -> str:
        try:
            return self.input_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            self.out("")
            raise _Quit

    def _default_name(self, cand: Candidate) -> str | None:
        """
        Suggest the channel name for the most talkative voice on a Kick
        channel — usually the streamer. Only a suggestion: the user still
        has to confirm it.
        """
        if cand.id != "speaker_1":
            return None
        name = channel_name(self.session.source)
        if name and not self.store.find_profile(name):
            return name
        return None

    def _note(self, cand: Candidate) -> str:
        return (f"Extracted from {self.session.source} on {self.session.created_at[:10]} "
                f"({fmt_duration(cand.speech_seconds)} speech, session {self.session.id})")

    def _print_summary(self, s: ReviewSummary):
        self.out("\n" + "─" * 60)
        parts = []
        if s.created:
            parts.append(f"created {', '.join(dict.fromkeys(s.created))}")
        if s.merged:
            parts.append(f"updated {', '.join(dict.fromkeys(s.merged))}")
        if s.skipped:
            parts.append(f"skipped {s.skipped}")
        self.out("📋 " + ("; ".join(parts) if parts else "No changes made") + ".")
        if s.remaining:
            self.out(f"   {s.remaining} speaker(s) still pending — resume with:\n"
                     f"   ./run.sh review --session {self.session.id}")
