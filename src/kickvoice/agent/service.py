"""
Everything an agent can do with kickvoice, as plain Python.

The MCP server (server.py) is a thin layer over this class: each public
method is one tool and returns JSON-serialisable data. Keeping the logic
here makes it testable without MCP and usable from other agent frameworks.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kickvoice.agent import live
from kickvoice.agent.chat import copy_to_clipboard, fit_to_chat, quoted_spans, verify_message
from kickvoice.agent.jobs import JobRunner
from kickvoice.agent.live import LiveSession, LiveSessions, Opponent
from kickvoice.config import Config
from kickvoice.research.store import (
    AUTHORSHIP_LABELS,
    CONFIRMED,
    FIRST_HAND,
    Evidence,
    ResearchStore,
)
from kickvoice.speakers.profiles import ProfileStore
from kickvoice.transcripts.store import TranscriptStore

logger = logging.getLogger(__name__)

PLATFORMS = ("x", "youtube", "reddit", "twitch", "wikipedia", "website")


class ToolError(ValueError):
    """A problem the agent should read and act on (bad id, missing setup…)."""


class AgentService:
    def __init__(
        self,
        config: Config,
        *,
        pipeline_factory=None,
        llm_factory: Callable[[Config], object] | None = None,
        fetcher_factory: Callable[[Config], object] | None = None,
        clipboard: Callable[[str], tuple[bool, str]] = copy_to_clipboard,
    ):
        self.config = config
        self.sessions = LiveSessions()
        self.jobs = JobRunner()
        self._pipeline_factory = pipeline_factory  # None = live.default_pipeline_factory
        self._llm_factory = llm_factory or _default_llm
        self._fetcher_factory = fetcher_factory or _default_fetcher
        self._clipboard = clipboard

    def close(self):
        self.sessions.stop_all()
        self.jobs.shutdown()

    def _research(self) -> ResearchStore:
        return ResearchStore(self.config.research_db)

    # ── live sessions ────────────────────────────────────────────

    def start_live_session(
        self,
        source: str,
        opponent: str | None = None,
        me: str | None = None,
        any_speaker: bool = False,
        duration_s: float | None = None,
        keep_audio: bool = False,
    ) -> dict:
        is_file = "://" not in source
        if is_file and not Path(source).expanduser().exists():
            raise ToolError(f"'{source}' is neither a URL nor an existing file")
        opp, warnings = self._resolve_opponent(opponent, me, any_speaker)
        config = copy.deepcopy(self.config)
        config.transcribe.chunk_duration_s = config.debate.chunk_duration_s  # low latency
        factory = self._pipeline_factory or live.default_pipeline_factory  # resolved per call
        session = LiveSession(
            str(Path(source).expanduser()) if is_file else source,
            is_file,
            config,
            opp,
            factory,
            max_duration_s=duration_s,
            keep_audio=keep_audio,
        )
        self.sessions.add(session)
        session.start()
        return {
            **session.info(),
            "warnings": warnings,
            "next": "Poll get_live_updates(session_id, cursor=0, wait_seconds=20). "
            "'point' events are the opponent's arguments.",
        }

    def get_live_updates(
        self, session_id: str, cursor: int = 0, wait_seconds: float = 15, max_events: int = 100
    ) -> dict:
        return self._session(session_id).updates(cursor, wait_seconds, max_events)

    def list_voices(self, session_id: str) -> dict:
        session = self._session(session_id)
        voices = session.voice_summary(refresh=True)
        return {
            "session_id": session.id,
            "opponent": session.opponent.describe(),
            "voices": voices,
            "hint": "Unknown voices can be enrolled with enroll_voice(utterance_id=<any of "
            "their sample lines>, name=...). Confirm who they are first.",
        }

    def enroll_voice(
        self,
        session_id: str,
        utterance_id: int,
        name: str,
        research_subject: str | None = None,
        merge_if_exists: bool = False,
    ) -> dict:
        """Create (or extend) a voice profile from one voice in the live session."""
        session = self._session(session_id)
        if not session.transcript_session:
            raise ToolError("Nothing transcribed yet")
        session.refine()
        with TranscriptStore(self.config.transcripts_db) as transcripts:
            anchor = transcripts.get_utterance(int(utterance_id))
            if anchor is None or anchor.session_id != session.transcript_session:
                raise ToolError(f"Utterance {utterance_id} isn't part of this session")
            if anchor.voice_group is None:
                raise ToolError(
                    "That line is too short to identify a voice; pick a longer one from list_voices"
                )
            same_voice = [
                u
                for u in transcripts.utterances(session.transcript_session)
                if u.voice_group == anchor.voice_group and u.embedding is not None
            ]
        speech = sum(u.duration for u in same_voice)
        if speech < self.config.agent.min_enroll_speech_s:
            raise ToolError(
                f"Only {speech:.0f}s of speech from that voice so far; need "
                f"{self.config.agent.min_enroll_speech_s:.0f}s. Try again later."
            )
        embedding = np.mean([u.embedding / np.linalg.norm(u.embedding) for u in same_voice], axis=0)

        profiles = ProfileStore(config=self.config)
        existing = profiles.find_profile(name)
        note = f"Enrolled from live session {session.source} ({speech:.0f}s speech)"
        if existing and not merge_if_exists:
            raise ToolError(
                f"A voice profile '{existing.name}' already exists. Pass "
                "merge_if_exists=true to add this audio to it."
            )
        if existing:
            profiles.merge_into_profile(
                existing.name,
                embedding,
                speech_seconds=speech,
                n_segments=len(same_voice),
                source_urls=[session.source],
                note=note,
            )
            profile_name = existing.name
        else:
            profiles.add_profile(
                name,
                embedding,
                source_urls=[session.source],
                n_segments=len(same_voice),
                notes=note,
                speech_seconds=speech,
            )
            profile_name = name

        linked = None
        if research_subject:
            with self._research() as store:
                person = store.get_person(research_subject) or store.add_person(research_subject)
                person.voice_profile = profile_name
                store.update_person(person)
                linked = person.name
        session.reload_profiles()
        session.refine()
        return {
            "voice_profile": profile_name,
            "speech_seconds": round(speech, 1),
            "lines_used": len(same_voice),
            "merged": bool(existing),
            "linked_research_subject": linked,
            "hint": "Call set_opponent to respond to this voice's points.",
        }

    def export_voice_clip(
        self, session_id: str, utterance_id: int, max_seconds: float = 20.0, out_dir: str | None = None
    ) -> dict:
        """Write a WAV of one voice's clearest lines so a person can listen before approving."""
        import wave

        session = self._session(session_id)
        if session.audio_path is None or not session.audio_path.exists():
            raise ToolError("This session isn't recording audio; start it with keep_audio=true")
        if not session.transcript_session:
            raise ToolError("Nothing transcribed yet")
        session.refine()
        with TranscriptStore(self.config.transcripts_db) as transcripts:
            anchor = transcripts.get_utterance(int(utterance_id))
            if anchor is None or anchor.session_id != session.transcript_session:
                raise ToolError(f"Utterance {utterance_id} isn't part of this session")
            lines = (
                [
                    u
                    for u in transcripts.utterances(session.transcript_session)
                    if u.voice_group == anchor.voice_group
                ]
                if anchor.voice_group
                else [anchor]
            )
        chosen, total = [], 0.0
        for u in sorted(lines, key=lambda u: -u.duration):  # clearest = longest lines
            if total >= max_seconds:
                break
            chosen.append(u)
            total += u.duration
        chosen.sort(key=lambda u: u.start_s)

        out = Path(out_dir).expanduser() if out_dir else self.config.data_dir / "clips"
        out.mkdir(parents=True, exist_ok=True)
        clip = out / f"{session.id}_voice{anchor.voice_group or 'x'}_{anchor.id}.wav"
        with wave.open(str(session.audio_path), "rb") as src, wave.open(str(clip), "wb") as dst:
            rate = src.getframerate()
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(rate)
            gap = b"\x00" * src.getsampwidth() * int(0.3 * rate)
            for i, u in enumerate(chosen):
                src.setpos(min(int(u.start_s * rate), src.getnframes()))
                dst.writeframes((gap if i else b"") + src.readframes(int(u.duration * rate)))
        return {
            "path": str(clip),
            "seconds": round(total, 1),
            "lines": len(chosen),
            "voice_group": anchor.voice_group,
            "transcript": [u.text for u in chosen],
        }

    def set_opponent(
        self, session_id: str, opponent: str | None = None, me: str | None = None, any_speaker: bool = False
    ) -> dict:
        session = self._session(session_id)
        opp, warnings = self._resolve_opponent(opponent, me, any_speaker)
        session.set_opponent(opp)
        return {
            "session_id": session.id,
            "opponent": opp.describe(),
            "research_subject": opp.research_name,
            "opponent_voice": opp.voice,
            "warnings": warnings,
        }

    def stop_live_session(self, session_id: str) -> dict:
        session = self._session(session_id)
        session.stop()
        return session.info()

    def list_live_sessions(self) -> dict:
        return {"sessions": [s.info() for s in self.sessions.all()]}

    def _session(self, session_id: str) -> LiveSession:
        try:
            return self.sessions.get(session_id)
        except KeyError as e:
            raise ToolError(str(e).strip("'\"")) from e

    def _resolve_opponent(self, name, me, any_speaker) -> tuple[Opponent, list[str]]:
        warnings = []
        if not name:
            if not any_speaker:
                warnings.append(
                    "No opponent set: lines are transcribed but no points are "
                    "detected. Use set_opponent once you know who to respond to."
                )
            return Opponent(me=me, any_speaker=any_speaker), warnings
        with self._research() as store:
            person = store.get_person(name)
        if person:
            if not person.voice_profile and not any_speaker:
                warnings.append(
                    f"{person.name} has no voice profile, so their lines can't be "
                    "recognised yet. Use list_voices + enroll_voice once they speak."
                )
            return Opponent(person.name, person.voice_profile, me, any_speaker), warnings
        profile = ProfileStore(config=self.config).find_profile(name)
        if profile:
            warnings.append(
                f"'{profile.name}' is a voice profile without research; counters "
                "will have no record to draw on (research_add_subject)."
            )
            return Opponent(None, profile.name, me, any_speaker), warnings
        raise ToolError(f"'{name}' is neither a research subject nor a voice profile")

    # ── research ─────────────────────────────────────────────────

    def research_subjects(self) -> dict:
        with self._research() as store:
            rows = []
            for p in store.list_persons():
                s = store.stats(p.id)
                rows.append(
                    {
                        "name": p.name,
                        "description": p.description,
                        "voice_profile": p.voice_profile,
                        "documents": s["documents"],
                        "claims": s["claims"],
                        "unanalysed_excerpts": s["chunks"] - s["analyzed"],
                    }
                )
        return {"subjects": rows}

    def research_profile(self, name: str) -> dict:
        from kickvoice.pipelines.research import rejected_document_count

        with self._research() as store:
            person = self._person(store, name)
            stats = store.stats(person.id)
            return {
                "name": person.name,
                "description": person.description,
                "voice_profile": person.voice_profile,
                "wikidata": person.wikidata_id,
                "stats": stats,
                "top_topics": [{"topic": t, "claims": n} for t, n in store.topics(person.id)[:20]],
                "accounts": [
                    {
                        "id": a.id,
                        "platform": a.platform,
                        "handle": a.handle,
                        "status": a.status,
                        "found_via": a.found_via,
                    }
                    for a in store.accounts(person.id)
                ],
                "documents_from_rejected_accounts": rejected_document_count(store, person),
            }

    def research_find_identity(
        self, name: str, handles: dict[str, str] | None = None, wikidata: str | None = None
    ) -> dict:
        """Look up candidate Wikidata person records. Doesn't link anything."""
        from kickvoice.pipelines.research import discover

        handles = {k: v for k, v in (handles or {}).items() if v}
        unknown = set(handles) - set(PLATFORMS)
        if unknown:
            raise ToolError(f"Unknown platform(s) {sorted(unknown)}; use {PLATFORMS}")
        result = discover(self._fetcher_factory(self.config), self.config, name, handles, qid=wikidata)
        return {
            "candidates": [
                {
                    "wikidata": c.record.qid,
                    "label": c.record.label,
                    "description": c.record.description,
                    "matched_by": c.matched_by,
                    "strong_match": c.strong,
                    "wikipedia": c.record.wikipedia_title,
                    "accounts": [{"platform": p, "handle": h} for p, h, _ in c.record.accounts],
                }
                for c in result.candidates
            ],
            "skipped_not_people": [
                {"wikidata": r.qid, "label": r.label, "description": r.description} for r in result.skipped
            ],
            "unmatched_handles": result.unmatched_handles,
            "note": (
                "A name-only match is NOT proof of identity — confirm with the user before "
                "research_link_wikidata."
                if result.searched_by_name
                else ""
            ),
        }

    def research_link_wikidata(self, name: str, wikidata: str, confirm_accounts: bool = False) -> dict:
        from kickvoice.pipelines.research import link_record
        from kickvoice.research.sources import wiki

        fetcher = self._fetcher_factory(self.config)
        try:
            record = wiki.get_person(fetcher, self.config.research.wikidata_api, wikidata.strip().upper())
        except ValueError as e:
            raise ToolError(str(e)) from e
        with self._research() as store:
            try:
                outcome = link_record(
                    store, name, record, lambda p, h: CONFIRMED if confirm_accounts else None
                )
            except ValueError as e:
                raise ToolError(str(e)) from e
            accounts = store.accounts(outcome.person.id)
        return {
            "name": outcome.person.name,
            "wikidata": record.qid,
            "description": record.description,
            "relinked_from": outcome.relinked_from,
            "rejected_from_old_record": [f"{a.platform}:{a.handle}" for a in outcome.dropped],
            "accounts": [
                {"id": a.id, "platform": a.platform, "handle": a.handle, "status": a.status} for a in accounts
            ],
        }

    def research_add_subject(
        self,
        name: str,
        description: str = "",
        voice_profile: str | None = None,
        accounts: dict[str, str] | None = None,
    ) -> dict:
        """Add/update a subject with accounts the user vouches for (marked confirmed)."""
        from kickvoice.pipelines.research import account_url

        accounts = {k: v for k, v in (accounts or {}).items() if v}
        unknown = set(accounts) - set(PLATFORMS)
        if unknown:
            raise ToolError(f"Unknown platform(s) {sorted(unknown)}; use {PLATFORMS}")
        with self._research() as store:
            person = store.get_person(name) or store.add_person(name)
            if description:
                person.description = description
            if voice_profile:
                person.voice_profile = voice_profile
            store.update_person(person)
            for platform, handle in accounts.items():
                store.add_account(
                    person.id,
                    platform,
                    handle,
                    account_url(platform, handle),
                    status=CONFIRMED,
                    found_via="added by agent (user-provided)",
                )
        return self.research_profile(name)

    def research_collect(
        self,
        name: str,
        sources: list[str] | None = None,
        voice: bool = False,
        appearances: int = 0,
        web_search: int = 0,
        max_videos: int | None = None,
        max_posts: int | None = None,
    ) -> dict:
        from kickvoice.pipelines.research import CollectOptions, collect

        with self._research() as store:
            person = self._person(store, name)
        options = CollectOptions(
            only=tuple(sources or ()),
            voice=voice,
            appearances=appearances,
            web_search=web_search,
            max_videos=max_videos,
            max_posts=max_posts,
        )

        def run(progress):
            with self._research() as store:
                p = store.get_person(person.name)
                results = collect(
                    store,
                    p,
                    self.config,
                    options,
                    fetcher=self._fetcher_factory(self.config),
                    progress=progress,
                )
                stats = store.stats(p.id)
            return {
                "by_source": {
                    k: {"new": v.new, "updated": v.updated, "unchanged": v.unchanged, "errors": v.errors[:5]}
                    for k, v in results.items()
                },
                "documents": stats["documents"],
                "unanalysed_excerpts": stats["chunks"] - stats["analyzed"],
            }

        job = self.jobs.submit("collect", f"collect {person.name}", run)
        return {
            **job.to_dict(),
            "hint": "Poll job_status(job_id). Raw excerpts are searchable "
            "as soon as they're collected; research_analyze adds "
            "verified claims.",
        }

    def research_analyze(self, name: str, limit: int | None = None) -> dict:
        from kickvoice.research.extract import estimate_tokens, extract_claims

        with self._research() as store:
            person = self._person(store, name)
            pending = store.chunks_to_analyze(person.id, limit=limit)
        if not pending:
            return {"status": "nothing to analyse"}
        tokens_in, tokens_out = estimate_tokens(pending, self.config)
        llm = self._llm_factory(self.config)

        def run(progress):
            with self._research() as store:
                p = store.get_person(person.name)
                stats = extract_claims(store, p, llm, self.config, limit_chunks=limit, progress=progress)
            return {
                "claims": stats.claims,
                "excerpts": stats.chunks,
                "rejected_unverifiable_quotes": stats.rejected_quotes,
                "failed_batches": stats.failed_batches,
                "errors": stats.errors[:3],
                "tokens": {"input": stats.input_tokens, "output": stats.output_tokens},
            }

        job = self.jobs.submit("analyze", f"analyze {person.name}", run)
        return {
            **job.to_dict(),
            "excerpts": len(pending),
            "estimated_tokens": {"input": tokens_in, "output": tokens_out},
            "model": self.config.llm.analysis_model,
        }

    def job_status(self, job_id: str, wait_seconds: float = 0) -> dict:
        try:
            job = self.jobs.wait(job_id, min(wait_seconds, self.config.agent.max_wait_s))
        except KeyError as e:
            raise ToolError(str(e).strip("'\"")) from e
        return job.to_dict()

    def research_search(self, name: str, query: str, limit: int = 10) -> dict:
        with self._research() as store:
            person = self._person(store, name)
            evidence = self._retriever(store, person).search(query, k=limit)
            insincere = _insincere(store, evidence)
        return {"subject": person.name, "results": [_evidence_dict(e, insincere) for e in evidence]}

    # ── counter-arguments ────────────────────────────────────────

    def get_point_evidence(
        self,
        session_id: str | None = None,
        point_id: str | None = None,
        text: str | None = None,
        opponent: str | None = None,
        limit: int = 10,
    ) -> dict:
        """The opponent's point plus the relevant parts of their record — for the agent to write a reply."""
        point_text, context, subject = self._point(session_id, point_id, text, opponent)
        if subject is None:
            return {
                "point": point_text,
                "context": context,
                "evidence": [],
                "note": "No research subject for this opponent — replies can only use reasoning. "
                "Use research_add_subject / research_collect.",
            }
        with self._research() as store:
            person = self._person(store, subject)
            evidence = self._retriever(store, person).search(point_text, k=limit)
            insincere = _insincere(store, evidence)
        return {
            "point": point_text,
            "context": context,
            "opponent": person.name,
            "evidence": [_evidence_dict(e, insincere) for e in evidence],
            "rules": (
                "Quote only first_hand, sincere evidence, verbatim. Cite refs. Mention a "
                "year only if the evidence is from it. Run verify_message before offering "
                f"a reply. Chat replies: max {self.config.agent.chat_max_chars} characters."
            ),
        }

    def suggest_counters(
        self,
        session_id: str | None = None,
        point_id: str | None = None,
        text: str | None = None,
        opponent: str | None = None,
        style: str = "chat",
        max_chars: int | None = None,
    ) -> dict:
        """Counter-arguments written by kickvoice's configured model, already fact-checked."""
        from kickvoice.research.counter import CounterEngine

        max_chars = max_chars or self.config.agent.chat_max_chars
        point_text, context, subject = self._point(session_id, point_id, text, opponent)
        if subject is None:
            raise ToolError(
                "No research subject for this opponent; set one with set_opponent or pass opponent="
            )
        llm = self._llm_factory(self.config)
        with self._research() as store:
            person = self._person(store, subject)
            engine = CounterEngine(store, person, llm, self.config, retriever=self._retriever(store, person))
            try:
                s = engine.suggest(point_text, context, style=style, max_chars=max_chars)
            except ValueError as e:
                raise ToolError(str(e)) from e
            insincere = _insincere(store, [e for c in s.counters for e in c.evidence])
        replies = []
        for c in s.counters:
            check = verify_message(c.text, c.evidence, max_chars, opponent=person.name)
            replies.append(
                {
                    "text": c.text,
                    "verification": check.to_dict(),
                    "sourced": c.sourced,
                    "evidence": [_evidence_dict(e, insincere) for e in c.evidence],
                    "notes": [n.text for n in c.notes],
                }
            )
        return {
            "point": point_text,
            "their_point": s.summary,
            "is_argument": s.is_argument,
            "replies": replies,
            "contradictions": [
                {
                    "explanation": c.explanation,
                    "quote": c.evidence.quote,
                    "date": c.evidence.published_at,
                    "url": c.evidence.url,
                    "ref": c.evidence.ref,
                }
                for c in s.contradictions
            ],
            "dropped_by_validation": s.dropped,
            "latency_s": round(s.latency_s, 2),
            "model": s.model,
        }

    def verify_message(
        self, opponent: str, message: str, cited_refs: list[str] | None = None, max_chars: int | None = None
    ) -> dict:
        """Fact-check a reply (quotes, years, attributions, length) against the opponent's record."""
        max_chars = max_chars or self.config.agent.chat_max_chars
        with self._research() as store:
            person = self._person(store, opponent)
            evidence = store.evidence_by_refs(person.id, cited_refs or [])
            for span in quoted_spans(message):  # also look for uncited quotes
                evidence += store.search_claims(person.id, span.text, 5)
                evidence += store.search_chunks(person.id, span.text, 5)
        return verify_message(message, evidence, max_chars, opponent=person.name).to_dict()

    def format_for_chat(self, text: str, max_chars: int | None = None) -> dict:
        formatted = fit_to_chat(text, max_chars or self.config.agent.chat_max_chars)
        return {"text": formatted, "length": len(formatted)}

    def copy_to_clipboard(self, text: str) -> dict:
        ok, detail = self._clipboard(text)
        return {"copied": ok, "detail": detail, "length": len(text)}

    # ── helpers ──────────────────────────────────────────────────

    def _point(self, session_id, point_id, text, opponent) -> tuple[str, list[str], str | None]:
        """(point text, context lines, research subject) from a live point or given text."""
        context: list[str] = []
        subject = opponent
        if point_id:
            if not session_id:
                raise ToolError("point_id needs session_id")
            session = self._session(session_id)
            point = session.points.get(point_id)
            if point is None:
                raise ToolError(f"No point '{point_id}' in session {session_id}")
            text, context = point.text, point.context
            subject = subject or session.opponent.research_name
        elif session_id and not subject:
            subject = self._session(session_id).opponent.research_name
        if not text:
            raise ToolError("Give point_id (from a 'point' event) or text")
        return text, context, subject

    def _person(self, store: ResearchStore, name: str):
        person = store.get_person(name)
        if person is None:
            raise ToolError(f"No research subject '{name}'. See research_subjects / research_add_subject.")
        return person

    def _retriever(self, store, person):
        from kickvoice.research.llm import make_embedder
        from kickvoice.research.retrieve import Retriever

        return Retriever(store, person, make_embedder(self.config))


def _insincere(store: ResearchStore, evidence: list[Evidence]) -> set[str]:
    return store.insincere_refs(evidence)


def _evidence_dict(e: Evidence, insincere: set[str]) -> dict:
    return {
        "ref": e.ref,
        "type": "claim" if e.kind == "claim" else "excerpt",
        "first_hand": e.authorship in FIRST_HAND,
        "authorship": AUTHORSHIP_LABELS.get(e.authorship, e.authorship),
        "sincere": e.ref not in insincere,
        "date": (e.published_at or "")[:10] or None,
        "source": e.source,
        "url": e.url or None,
        **({"claim": e.text, "topic": e.topic} if e.kind == "claim" else {}),
        "quote": e.quote,
    }


def _default_llm(config: Config):
    from kickvoice.research.llm import LLMError, _api_key, make_llm

    try:
        if config.llm.provider == "anthropic":
            _api_key(config.llm.api_key_env or "ANTHROPIC_API_KEY")
        return make_llm(config)
    except LLMError as e:
        raise ToolError(
            f"{e} (needed for server-side suggestions/analysis; the agent can "
            "still use get_point_evidence and write replies itself)"
        ) from e


def _default_fetcher(config: Config):
    from kickvoice.research.net import Fetcher

    return Fetcher(config)
