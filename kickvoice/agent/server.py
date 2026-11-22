"""
kickvoice as an MCP server: listen to a live stream, know who's speaking,
research them, and help a viewer reply in chat.

The agent (Claude Desktop, Claude Code, or any MCP host) does the thinking
and writing; these tools give it senses and memory:

    hear     live_start → live_next_point (blocks until someone makes a point)
    who      live_voices → save_voice (after you confirm a name)
    know     research_* (their public record, with verified quotes)
    reply    check_chat_message → copy_to_clipboard (you paste it yourself)

Nothing here posts to chat: replies are drafts for the viewer to review,
edit and send from their own account.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from kickvoice.agent.chat import check_message, copy_to_clipboard
from kickvoice.agent.jobs import JobRunner
from kickvoice.agent.live import LiveSession, _default_pipeline
from kickvoice.config import Config
from kickvoice.research.store import (
    AUTHORSHIP_LABELS, CONFIRMED, FIRST_HAND, Evidence, ResearchStore,
)

logger = logging.getLogger(__name__)

INSTRUCTIONS = """kickvoice listens to a live stream (e.g. a Kick debate), identifies speakers by voice,
and keeps a cited record of public figures' past statements, so you can help the user — a
viewer — reply in chat with sharp, well-sourced counter-arguments.

Typical loop (see the `chat_copilot` prompt):
1. live_start(source) once. 2. live_next_point() repeatedly — it waits for someone to finish
a point and returns it with evidence from the speaker's record when they're researched.
3. Draft 1-3 short replies in the user's chat style, grounded in that evidence.
4. check_chat_message() on each draft; fix any issues. 5. Show drafts; copy_to_clipboard()
the one the user picks. The user posts it themselves — never claim to have posted anything.

Unknown voice? live_voices() lists everyone heard; ask the user who they are before
save_voice(). Not researched yet? research_add / research_discover (ask the user to confirm
accounts), then research_collect and research_analyze (background jobs; check job_status)."""


class AgentState:
    """Everything the tools share: config, the live session, background jobs."""

    def __init__(self, config: Config, pipeline_factory=_default_pipeline):
        self.config = config
        self.pipeline_factory = pipeline_factory
        self.live: LiveSession | None = None
        self.jobs = JobRunner()
        self.lock = threading.Lock()

    def store(self) -> ResearchStore:
        # One connection per call: tools run on different worker threads.
        return ResearchStore(self.config.research_db)


class ToolFailure(Exception):
    """An anticipated failure whose message should reach the agent."""


def evidence_item(ev: Evidence) -> dict:
    item = {"ref": ev.ref, "kind": ev.kind, "quote": ev.quote,
            "said_by": AUTHORSHIP_LABELS.get(ev.authorship, ev.authorship),
            "first_hand": ev.authorship in FIRST_HAND,
            "date": (ev.published_at or "")[:10] or None, "source": ev.source, "url": ev.url or None}
    if ev.kind == "claim":
        item["claim"] = ev.text
        item["topic"] = ev.topic
    return item


# ── tool implementations (plain functions: easy to test) ─────────

def live_start(state: AgentState, source: str, speakers: list[str] | None = None,
               duration_s: float | None = None) -> dict:
    with state.lock:
        if state.live is not None and state.live.running:
            raise ToolFailure(f"Already listening to {state.live.source}. Call live_stop first.")
        if "://" not in source and not os.path.exists(source):
            raise ToolFailure(f"Not a URL or an existing file: {source}")
        config = state.config
        config.transcribe.chunk_duration_s = config.debate.chunk_duration_s   # low latency
        state.live = LiveSession(config, source, speakers, duration_s, state.pipeline_factory)
        state.live.start()
    from kickvoice.speakers.profiles import ProfileStore

    voices = [p.name for p in ProfileStore(config=state.config).list_profiles()]
    with state.store() as store:
        researched = {v: (p.name if (p := store.person_by_voice(v)) else None) for v in voices}
    return {"listening_to": source, "tracking": speakers or "everyone",
            "recognisable_voices": [{"voice": v, "research_subject": researched[v]} for v in voices],
            "next": "Call live_next_point to wait for the first point."}


def _live(state: AgentState) -> LiveSession:
    if state.live is None:
        raise ToolFailure("Not listening to anything. Call live_start first.")
    return state.live


def live_next_point(state: AgentState, timeout_s: float = 30.0, speaker: str | None = None) -> dict:
    live = _live(state)
    point = live.next_point(timeout_s=min(max(timeout_s, 0.0), 120.0), speaker=speaker)
    if point is None:
        status = live.status()
        reason = ("stream ended" if status["state"] != "running" else
                  f"no new point within {timeout_s:.0f}s")
        return {"point": None, "reason": reason, "error": status["error"]}

    result: dict[str, Any] = {"point": point.to_dict(), "behind_live_s": live.behind_live_s(point),
                              "subject": None, "evidence": []}
    if point.speaker is None:
        result["note"] = ("Unrecognised voice. live_voices() shows who's speaking; if the user "
                          "knows who this is, save_voice() then research them.")
        return result
    with state.store() as store:
        person = store.person_by_voice(point.speaker)
        if person is None:
            result["note"] = (f"No research record is linked to voice '{point.speaker}'. "
                              "research_add(name, voice_profile=...) links one.")
            return result
        from kickvoice.research.retrieve import Retriever

        hits = Retriever(store, person).search(point.text, k=state.config.debate.evidence_items)
        result["subject"] = {"name": person.name, "description": person.description,
                             "claims_on_record": store.stats(person.id)["claims"]}
        result["evidence"] = [evidence_item(e) for e in hits]
    if point.attribution != "confident":
        result["note"] = ("Speaker attribution for this point is only "
                          f"'{point.attribution}' — don't claim they said it if unsure.")
    return result


def live_status(state: AgentState) -> dict:
    return _live(state).status()


def live_transcript(state: AgentState, since: int = 0, limit: int = 40) -> dict:
    lines, cursor = _live(state).lines_since(max(0, since), min(max(1, limit), 200))
    return {"lines": [vars(line) for line in lines], "next_cursor": cursor}


def live_voices(state: AgentState) -> dict:
    """Group everything heard so far by voice (runs the whole-session check)."""
    from kickvoice.speakers.profiles import ProfileStore
    from kickvoice.transcripts.refine import refine_session
    from kickvoice.transcripts.store import TranscriptStore

    live = _live(state)
    session_id = live.transcript_session_id
    if session_id is None:
        return {"voices": [], "note": "Nothing transcribed yet."}
    names, matrix = ProfileStore(config=state.config).embedding_matrix()
    with TranscriptStore(state.config.transcripts_db) as transcripts:
        refine_session(transcripts, session_id, names, matrix, state.config)
        utterances = transcripts.utterances(session_id)
    groups: dict[int, list] = {}
    for u in utterances:
        if u.voice_group is not None:
            groups.setdefault(u.voice_group, []).append(u)
    with state.store() as store:
        voices = []
        for g, us in sorted(groups.items()):
            speaker = next((u.speaker for u in us if u.speaker), None)
            person = store.person_by_voice(speaker) if speaker else None
            voices.append({
                "voice_group": g, "voice": speaker,
                "research_subject": person.name if person else None,
                "speech_s": round(sum(u.duration for u in us), 1), "lines": len(us),
                "first_heard_s": round(min(u.start_s for u in us), 1),
                "sample": max(us, key=lambda u: len(u.text)).text[:200],
            })
    return {"voices": voices, "transcript_session": session_id,
            "note": "Unnamed groups are unrecognised voices. Ask the user who they are before "
                    "calling save_voice."}


def save_voice(state: AgentState, voice_group: int, name: str, merge: bool = False) -> dict:
    """Turn a voice group from the live session into a voice profile."""
    from kickvoice.speakers.profiles import ProfileStore, normalise
    from kickvoice.transcripts.store import TranscriptStore

    live = _live(state)
    session_id = live.transcript_session_id
    if session_id is None:
        raise ToolFailure("Nothing transcribed yet.")
    name = name.strip()
    if not name:
        raise ToolFailure("A name is required.")
    with TranscriptStore(state.config.transcripts_db) as transcripts:
        group = [u for u in transcripts.utterances(session_id)
                 if u.voice_group == voice_group and u.embedding is not None]
    if not group:
        raise ToolFailure(f"No voice group {voice_group} — call live_voices first.")
    known = {u.speaker for u in group if u.speaker}
    if known:
        raise ToolFailure(f"Voice group {voice_group} is already recognised as {', '.join(known)}.")
    speech = sum(u.duration for u in group)
    embedding = normalise(np.mean([normalise(u.embedding) for u in group], axis=0))
    profiles = ProfileStore(config=state.config)
    existing = profiles.find_profile(name)
    note = f"Saved from live stream {live.source} (voice group {voice_group}, session {session_id})"
    if existing and not merge:
        raise ToolFailure(f"A voice profile named '{existing.name}' exists. Pass merge=true to "
                          "add this audio to it (only if it's the same person).")
    if existing:
        profiles.merge_into_profile(existing.name, embedding, speech_seconds=speech,
                                    n_segments=len(group), source_urls=[live.source], note=note)
        name = existing.name
    else:
        profiles.add_profile(name, embedding, source_urls=[live.source], n_segments=len(group),
                             notes=note, speech_seconds=speech)
    live.reload_profiles()
    warning = None
    if speech < state.config.extract.min_speaker_speech_s:
        warning = (f"Only {speech:.0f}s of speech — the profile may be unreliable; it will "
                   "improve if you merge more audio later.")
    return {"saved": name, "merged": bool(existing), "speech_s": round(speech, 1),
            "lines": len(group), "warning": warning,
            "next": f"New lines from this voice will be labelled '{name}'. To research them: "
                    f"research_add('{name}', voice_profile='{name}', ...)."}


def live_stop(state: AgentState) -> dict:
    live = _live(state)
    live.stop()
    status = live.status()
    status["note"] = "Stopped. The transcript stays searchable (kickvoice transcripts show)."
    return status


# research

def _person(store: ResearchStore, name: str):
    person = store.get_person(name)
    if person is None:
        known = ", ".join(p.name for p in store.list_persons()) or "none yet"
        raise ToolFailure(f"No research subject '{name}' (known: {known}). Use research_add.")
    return person


def research_subjects(state: AgentState) -> dict:
    with state.store() as store:
        return {"subjects": [
            {"name": p.name, "voice": p.voice_profile, "description": p.description,
             **{k: v for k, v in store.stats(p.id).items() if k in ("documents", "claims")}}
            for p in store.list_persons()]}


def research_profile(state: AgentState, name: str) -> dict:
    from kickvoice.pipelines.research import purge_rejected

    with state.store() as store:
        person = _person(store, name)
        stats = store.stats(person.id)
        return {
            "name": person.name, "description": person.description, "voice": person.voice_profile,
            "wikidata": person.wikidata_id, "stats": stats,
            "top_topics": [{"topic": t, "claims": n} for t, n in store.topics(person.id)[:15]],
            "accounts": [{"id": a.id, "platform": a.platform, "handle": a.handle,
                          "status": a.status, "found_via": a.found_via}
                         for a in store.accounts(person.id)],
            "documents_from_rejected_accounts": purge_rejected(store, person, dry_run=True),
        }


def research_search(state: AgentState, name: str, query: str, limit: int = 8) -> dict:
    from kickvoice.research.llm import make_embedder
    from kickvoice.research.retrieve import Retriever

    with state.store() as store:
        person = _person(store, name)
        hits = Retriever(store, person, make_embedder(state.config)).search(
            query, k=min(max(1, limit), 25))
        return {"subject": person.name, "results": [evidence_item(e) for e in hits]}


def research_claims(state: AgentState, name: str, topic: str) -> dict:
    with state.store() as store:
        person = _person(store, name)
        claims = store.claims(person.id, topic=topic)
        return {"subject": person.name, "topic": topic, "claims": [
            {"ref": f"C{c.id}", "claim": c.claim, "quote": c.quote, "sincere": c.sincere,
             "said_by": AUTHORSHIP_LABELS.get(c.authorship, c.authorship),
             "date": (c.published_at or "")[:10] or None, "url": c.url or None}
            for c in claims[:60]]}


def research_add(state: AgentState, name: str, voice_profile: str | None = None,
                 description: str | None = None, accounts: dict[str, str] | None = None) -> dict:
    from kickvoice.cli_research import PLATFORM_OPTIONS, _account_url

    accounts = {k.lower(): v for k, v in (accounts or {}).items() if v}
    unknown = set(accounts) - set(PLATFORM_OPTIONS)
    if unknown:
        raise ToolFailure(f"Unknown platform(s): {', '.join(sorted(unknown))}. "
                          f"Use: {', '.join(PLATFORM_OPTIONS)}")
    with state.store() as store:
        person = store.get_person(name) or store.add_person(name)
        if voice_profile:
            person.voice_profile = voice_profile
        if description:
            person.description = description
        store.update_person(person)
        for platform, handle in accounts.items():
            store.add_account(person.id, platform, handle, _account_url(platform, handle),
                              status=CONFIRMED, found_via="added by the user (via agent)")
        return research_profile(state, person.name)


def research_discover(state: AgentState, name: str, accounts: dict[str, str] | None = None,
                      wikidata_id: str | None = None) -> dict:
    """Find candidate Wikidata person records. Links nothing."""
    from kickvoice.pipelines.research import discover
    from kickvoice.research.net import Fetcher

    result = discover(Fetcher(state.config), state.config, name,
                      {k.lower(): v for k, v in (accounts or {}).items() if v}, qid=wikidata_id)
    return {
        "candidates": [{"wikidata_id": c.record.qid, "label": c.record.label,
                        "description": c.record.description,
                        "matched_by": c.matched_by, "strong_match": c.strong,
                        "accounts": [{"platform": p, "handle": h} for p, h, _ in c.record.accounts],
                        "wikipedia": c.record.wikipedia_title} for c in result.candidates],
        "skipped_not_people": [{"wikidata_id": r.qid, "label": r.label,
                                "description": r.description} for r in result.skipped],
        "handles_not_on_wikidata": result.unmatched_handles,
        "note": "Nothing is linked yet. Show the user the candidates; only after they confirm "
                "who it is, call research_link_record.",
    }


def research_link_record(state: AgentState, name: str, wikidata_id: str,
                         confirm_accounts: bool = False) -> dict:
    from kickvoice.pipelines.research import link_record
    from kickvoice.research.net import Fetcher
    from kickvoice.research.sources import wiki

    record = wiki.get_person(Fetcher(state.config), state.config.research.wikidata_api,
                             wikidata_id.strip().upper())
    if not record.is_human:
        raise ToolFailure(f"{record.qid} is {record.description or record.label!r}, not a person.")
    with state.store() as store:
        result = link_record(store, name, record,
                             lambda platform, handle: CONFIRMED if confirm_accounts else None)
    profile = research_profile(state, name)
    profile["relinked_from"] = result.relinked_from
    profile["rejected_old_accounts"] = [f"{a.platform}:{a.handle}" for a in result.rejected]
    return profile


def research_collect(state: AgentState, name: str, sources: list[str] | None = None,
                     voice: bool = False, appearances: int = 0,
                     max_videos: int | None = None, max_posts: int | None = None) -> dict:
    from kickvoice.pipelines.research import CollectOptions, collect

    with state.store() as store:
        person = _person(store, name)
        if not store.accounts(person.id, status=CONFIRMED) and not person.voice_profile:
            raise ToolFailure(f"{person.name} has no confirmed accounts to collect from.")
    options = CollectOptions(only=tuple(sources or ()), voice=voice, appearances=appearances,
                             max_videos=max_videos, max_posts=max_posts)

    def run(progress):
        with state.store() as store:
            person = store.get_person(name)
            results = collect(store, person, state.config, options, progress=progress)
            return {"by_source": {k: {"new": v.new, "updated": v.updated, "unchanged": v.unchanged,
                                      "errors": v.errors[:5]} for k, v in results.items()},
                    "on_record": store.stats(person.id)}

    job = state.jobs.submit("collect", f"collect {name}", run)
    return {"job": job.id, "note": "Runs in the background; poll job_status. Then research_analyze."}


def research_analyze(state: AgentState, name: str, limit: int | None = None) -> dict:
    from kickvoice.research.extract import estimate_tokens, extract_claims
    from kickvoice.research.llm import LLMError, make_llm

    cfg = state.config.llm
    if cfg.provider == "anthropic":
        key_env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        if not os.environ.get(key_env):
            raise ToolFailure(f"Claim extraction uses kickvoice's own model access: set {key_env} "
                              "where the MCP server runs (or configure llm.provider).")
    with state.store() as store:
        person = _person(store, name)
        pending = store.chunks_to_analyze(person.id, limit=limit)
        if not pending:
            return {"job": None, "note": "Nothing new to analyse."}
        tokens_in, tokens_out = estimate_tokens(pending, state.config)

    def run(progress):
        try:
            llm = make_llm(state.config)
        except LLMError as e:
            raise RuntimeError(str(e)) from e
        with state.store() as store:
            person = store.get_person(name)
            stats = extract_claims(store, person, llm, state.config, limit_chunks=limit,
                                   progress=progress)
            return {"claims_added": stats.claims, "excerpts": stats.chunks,
                    "rejected_quotes": stats.rejected_quotes, "failed_batches": stats.failed_batches,
                    "tokens": {"in": stats.input_tokens, "out": stats.output_tokens}}

    job = state.jobs.submit("analyze", f"analyze {name}", run)
    return {"job": job.id, "excerpts": len(pending),
            "estimated_tokens": {"in": tokens_in, "out": tokens_out},
            "note": "Runs in the background; poll job_status."}


def job_status(state: AgentState, job_id: str | None = None) -> dict:
    if job_id:
        job = state.jobs.get(job_id)
        if job is None:
            raise ToolFailure(f"No job {job_id}.")
        return job.to_dict()
    return {"jobs": [j.to_dict(progress_tail=1) for j in state.jobs.all()]}


# chat

def check_chat_message(state: AgentState, text: str, subject: str | None = None,
                       evidence_refs: list[str] | None = None) -> dict:
    evidence_texts: list[str] = []
    problems: list[str] = []
    if evidence_refs:
        if not subject:
            raise ToolFailure("evidence_refs need the subject they belong to.")
        with state.store() as store:
            person = _person(store, subject)
            for ref in evidence_refs:
                ref = str(ref).strip("[] ").upper()
                kind, num = ref[:1], ref[1:]
                if not num.isdigit():
                    problems.append(f"Bad evidence ref {ref!r}")
                    continue
                if kind == "C":
                    claim = store.get_claim(int(num))
                    if claim is None or claim.person_id != person.id:
                        problems.append(f"{ref} isn't in {person.name}'s record")
                        continue
                    evidence_texts += [claim.quote, store.chunk_text(claim.chunk_id) or ""]
                elif kind == "K":
                    text_k = store.chunk_text(int(num))
                    if text_k is None:
                        problems.append(f"{ref} not found")
                        continue
                    evidence_texts.append(text_k)
                else:
                    problems.append(f"Bad evidence ref {ref!r}")
    chat = state.config.chat
    check = check_message(text, evidence_texts, chat.max_chars, chat.allow_links)
    check.issues = problems + check.issues
    check.ok = not check.issues
    return vars(check)


def copy_text(state: AgentState, text: str) -> dict:
    ok, how = copy_to_clipboard(text)
    return {"copied": ok, "via" if ok else "reason": how,
            "note": "Paste it into chat yourself." if ok else "Show the text so the user can copy it."}


COPILOT_PROMPT = """You are the user's chat copilot for a live stream they're WATCHING. The user is a
viewer; they want sharp replies they can paste into chat under their own name.

Loop until the user says stop:
1. If not listening yet: live_start("{source}"{speakers_arg}).
2. live_next_point(timeout_s=45). If null, call it again (say briefly if the stream ended).
3. Decide if the point deserves a reply (skip banter, jokes, small talk, and points where
   behind_live_s is over ~60 — chat has moved on). If it does, write
   2-3 alternative chat replies:
   - in the user's voice: {style}
   - each at most {max_chars} characters, one line, no markdown, no hashtags
   - argue with the point, not the person: no insults, slurs, or anything about their private
     life — that gets messages deleted and users banned, and it convinces nobody
   - ground them in the `evidence` when it's relevant. Only put words in quotation marks if
     they appear verbatim in an evidence item, and pass its ref to check_chat_message.
     "their own words" evidence can be quoted as theirs; "reported about them" cannot.
   - if the point's attribution isn't "confident", don't claim that person said it
   - no invented facts or statistics; if unsure, ask a pointed question instead
4. check_chat_message(draft, subject, evidence_refs) for each; fix every issue.
5. Show the drafts numbered, each with its source link if any (links usually can't go in chat).
   When the user picks one, copy_to_clipboard(it). They post it themselves — never say you did.
6. One reply per point at most: chats have slow mode and spam filters.

If a speaker is unrecognised and the user tells you who they are, use live_voices then
save_voice, then research_add / research_discover → research_collect → research_analyze
(background; check job_status) so their record is available for later points."""


def copilot_prompt(config: Config, source: str = "", speakers: str = "", style: str = "") -> str:
    speakers_list = [s.strip() for s in speakers.split(",") if s.strip()]
    return COPILOT_PROMPT.format(
        source=source or "<stream URL the user gives you>",
        speakers_arg=f", speakers={speakers_list!r}" if speakers_list else "",
        style=style or config.chat.style or "casual, concise, confident; match how the user writes",
        max_chars=config.chat.max_chars)


# ── MCP wiring ───────────────────────────────────────────────────

def build_server(config: Config, state: AgentState | None = None):
    """Create the MCP server with all tools registered."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    state = state or AgentState(config)
    server = MCPServer("kickvoice", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    local_write = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
    web = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

    def tool(fn, annotations, description, name=None):
        def call(**kwargs):
            try:
                return fn(state, **kwargs)
            except ToolFailure as e:
                raise ToolError(str(e)) from e
        call.__name__ = name or fn.__name__
        call.__signature__ = _without_state(fn)
        call.__annotations__ = {k: v for k, v in fn.__annotations__.items() if k != "state"}
        server.tool(name=name or fn.__name__, description=description,
                    annotations=annotations)(call)

    tool(live_start, web, "Start listening to a stream URL (or local file): transcribes it, "
         "recognises enrolled voices, and groups speakers' words into points. `speakers` limits "
         "points to those voice profiles (default: everyone).")
    tool(live_next_point, read_only, "Wait (up to timeout_s) for the next completed point "
         "someone makes, with evidence from that speaker's research record if they have one.")
    tool(live_status, read_only, "Is the stream still being transcribed, any error, how many "
         "lines/points so far, and realtime_factor (> 1.0 = falling behind the stream).")
    tool(live_transcript, read_only, "Raw transcript lines since a cursor (for context).")
    tool(live_voices, read_only, "Everyone heard so far, grouped by voice: recognised profile "
         "or an unnamed voice group, with speech time and a sample line.")
    tool(save_voice, local_write, "Create a voice profile from a voice group (see live_voices). "
         "Only call after the user confirms who it is.")
    tool(live_stop, local_write, "Stop listening.")
    tool(research_subjects, read_only, "Researched people and how much is on record.")
    tool(research_profile, read_only, "One subject: accounts, stats, top topics.")
    tool(research_search, read_only, "Search a subject's record (verified claims and excerpts) "
         "for anything relevant to a query.")
    tool(research_claims, read_only, "A subject's claims on a topic, with quotes and dates.")
    tool(research_add, local_write, "Add/update a research subject with accounts the USER gave "
         "you (platform → handle: x, youtube, reddit, twitch, wikipedia, website) and their "
         "voice profile. Never guess handles.")
    tool(research_discover, web, "Find candidate Wikidata records (people only) for a name or "
         "known handles. Links nothing — show the user, then research_link_record.")
    tool(research_link_record, web, "Link the Wikidata record the user confirmed. "
         "confirm_accounts=true only if the user approved its accounts.")
    tool(research_collect, web, "Background job: collect the subject's public record from "
         "confirmed accounts (voice=true: transcribe videos, keep only their voice).")
    tool(research_analyze, web, "Background job: extract claims with verified quotes (uses "
         "kickvoice's own model access and costs tokens).")
    tool(job_status, read_only, "Status/result of a background job (or all jobs).")
    tool(check_chat_message, read_only, "Check a chat draft before showing it: cleans it to one "
         "line, enforces the chat length limit, and verifies anything in quotation marks against "
         "the cited evidence refs. Fix every issue it reports.")
    tool(copy_text, local_write, "Copy the reply the user picked to their clipboard.",
         name="copy_to_clipboard")

    @server.prompt(name="chat_copilot", description="Live chat copilot: counter-arguments for "
                   "the stream you're watching, drafted in your voice for you to paste.")
    def chat_copilot(source: str = "", speakers: str = "", style: str = "") -> str:
        return copilot_prompt(config, source, speakers, style)

    return server


def _without_state(fn):
    import inspect

    sig = inspect.signature(fn)
    return sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "state"])


def serve_stdio(config: Config):
    """Run over stdio (how MCP hosts launch local servers)."""
    build_server(config).run("stdio")
