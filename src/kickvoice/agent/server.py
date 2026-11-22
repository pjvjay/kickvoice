"""
kickvoice as an MCP server — run with `kickvoice mcp` (stdio).

An MCP-capable agent (Claude Desktop, Claude Code, …) gets tools to:
  - follow a live stream: who's speaking, what the opponent is arguing
  - profile voices and enroll new ones
  - research the opponent's public record (background jobs)
  - draft chat replies grounded in that record, fact-check them, and copy
    the one you choose to your clipboard

The agent drafts; you decide what to paste. Nothing is ever posted to chat.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from pydantic import Field

from kickvoice.agent.service import AgentService, ToolError
from kickvoice.config import Config

logger = logging.getLogger(__name__)

INSTRUCTIONS = """kickvoice follows a live stream debate and helps the user reply in the stream chat.

Typical loop:
1. start_live_session(source, opponent) — opponent is a research subject (see research_subjects).
2. get_live_updates(session_id, cursor, wait_seconds=20) repeatedly, passing back next_cursor.
   "line" events are the transcript; "point" events are the opponent's arguments.
3. For a point: get_point_evidence(session_id, point_id) and write a short chat reply yourself,
   or call suggest_counters(session_id, point_id) for ready-made, pre-checked replies.
4. verify_message before presenting any reply you wrote. Show the user 1-3 options;
   copy_to_clipboard only the one they pick. Never claim anything was posted to chat.

Voices: list_voices shows who is talking. If the opponent isn't recognised yet, identify the
voice from what's said (ask the user to confirm), then enroll_voice + set_opponent.

Research (if the record is thin): research_find_identity → confirm with the user →
research_link_wikidata / research_add_subject → research_collect → research_analyze.
These run as background jobs; check job_status. Excerpts are searchable as soon as collected.

Accuracy rules (quoting someone wrong in public backfires):
- Put quote marks (of any kind) only around words that appear verbatim, as one unbroken
  passage, in first_hand, sincere evidence. Never skip words with "...".
- "reported about them" evidence is what others say — never present it as their words.
- Mention a date/year only if the cited evidence is from it. Don't invent statistics.
- Attack the argument, not the person."""


def _run(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Turn anticipated failures into messages the agent can read and act on."""
    try:
        return fn(*args, **kwargs)
    except KeyError as e:  # str(KeyError) adds repr quotes
        raise MCPToolError(str(e.args[0]) if e.args else "not found") from e
    except (ToolError, ValueError, RuntimeError) as e:
        raise MCPToolError(str(e)) from e


def build_server(service: AgentService) -> MCPServer:
    mcp = MCPServer("kickvoice", instructions=INSTRUCTIONS)
    s = service

    # ── live ─────────────────────────────────────────────────────

    @mcp.tool()
    def start_live_session(
        source: Annotated[
            str, Field(description="Stream URL (e.g. https://kick.com/channel) or a local recording path")
        ],
        opponent: Annotated[
            str | None, Field(description="Research subject (or voice profile) whose points to counter")
        ] = None,
        me: Annotated[
            str | None, Field(description="The user's own voice profile, never treated as the opponent")
        ] = None,
        any_speaker: Annotated[
            bool, Field(description="Treat every voice except `me` as the opponent")
        ] = False,
        duration_s: Annotated[float | None, Field(description="Stop after this many seconds")] = None,
        keep_audio: Annotated[
            bool, Field(description="Record the audio so voice clips can be exported for review")
        ] = False,
    ) -> dict:
        """Start transcribing a live stream in the background and detecting the opponent's points."""
        return _run(s.start_live_session, source, opponent, me, any_speaker, duration_s, keep_audio)

    @mcp.tool()
    def get_live_updates(
        session_id: str,
        cursor: Annotated[int, Field(description="next_cursor from the previous call (0 at first)")] = 0,
        wait_seconds: Annotated[
            float, Field(description="Wait up to this long for new events (max 30)")
        ] = 15,
        max_events: int = 100,
    ) -> dict:
        """New events since `cursor`: transcript lines, the opponent's points, voice changes, status."""
        return _run(s.get_live_updates, session_id, cursor, wait_seconds, max_events)

    @mcp.tool()
    def list_voices(session_id: str) -> dict:
        """Distinct voices heard so far: who they are (if enrolled), speech time, sample lines."""
        return _run(s.list_voices, session_id)

    @mcp.tool()
    def enroll_voice(
        session_id: str,
        utterance_id: Annotated[
            int, Field(description="Any line spoken by that voice (from list_voices samples)")
        ],
        name: Annotated[str, Field(description="Who it is — confirm with the user first")],
        research_subject: Annotated[
            str | None, Field(description="Also link this voice to a research subject")
        ] = None,
        merge_if_exists: bool = False,
    ) -> dict:
        """Create a voice profile from one voice in the live session so it's recognised from now on."""
        return _run(s.enroll_voice, session_id, utterance_id, name, research_subject, merge_if_exists)

    @mcp.tool()
    def export_voice_clip(
        session_id: str,
        utterance_id: Annotated[int, Field(description="Any line spoken by that voice")],
        max_seconds: float = 20.0,
        out_dir: str | None = None,
    ) -> dict:
        """Save a short WAV of one voice (needs keep_audio) so the user can listen before approving."""
        return _run(s.export_voice_clip, session_id, utterance_id, max_seconds, out_dir)

    @mcp.tool()
    def set_opponent(
        session_id: str,
        opponent: Annotated[str | None, Field(description="Research subject or voice profile")] = None,
        me: str | None = None,
        any_speaker: bool = False,
    ) -> dict:
        """Change whose arguments produce 'point' events."""
        return _run(s.set_opponent, session_id, opponent, me, any_speaker)

    @mcp.tool()
    def stop_live_session(session_id: str) -> dict:
        """Stop a live session (its transcript is kept)."""
        return _run(s.stop_live_session, session_id)

    @mcp.tool()
    def list_live_sessions() -> dict:
        """Live sessions in this server."""
        return _run(s.list_live_sessions)

    # ── research ─────────────────────────────────────────────────

    @mcp.tool()
    def research_subjects() -> dict:
        """People with a research record (documents, verified claims, voice profile)."""
        return _run(s.research_subjects)

    @mcp.tool()
    def research_profile(name: str) -> dict:
        """A subject's record: accounts, document counts, main topics."""
        return _run(s.research_profile, name)

    @mcp.tool()
    def research_search(
        name: str,
        query: Annotated[str, Field(description="What to look for in their record")],
        limit: int = 10,
    ) -> dict:
        """Search a subject's record. Results carry refs, verbatim quotes, dates, links and authorship."""
        return _run(s.research_search, name, query, limit)

    @mcp.tool()
    def research_find_identity(
        name: str,
        handles: Annotated[
            dict[str, str] | None,
            Field(description='Known accounts, e.g. {"x": "handle", "youtube": "@channel"}'),
        ] = None,
        wikidata: Annotated[str | None, Field(description="A Wikidata id to check, e.g. Q12345")] = None,
    ) -> dict:
        """Find candidate Wikidata person records and their official accounts. Links nothing."""
        return _run(s.research_find_identity, name, handles, wikidata)

    @mcp.tool()
    def research_link_wikidata(
        name: str,
        wikidata: str,
        confirm_accounts: Annotated[
            bool, Field(description="Mark the record's accounts confirmed (only after the user agrees)")
        ] = False,
    ) -> dict:
        """Link a research subject to a Wikidata person record (people only)."""
        return _run(s.research_link_wikidata, name, wikidata, confirm_accounts)

    @mcp.tool()
    def research_add_subject(
        name: str,
        description: str = "",
        voice_profile: str | None = None,
        accounts: Annotated[
            dict[str, str] | None, Field(description='Accounts the user vouches for, e.g. {"x": "handle"}')
        ] = None,
    ) -> dict:
        """Add or update a research subject with accounts the user has confirmed are theirs."""
        return _run(s.research_add_subject, name, description, voice_profile, accounts)

    @mcp.tool()
    def research_collect(
        name: str,
        sources: Annotated[
            list[str] | None,
            Field(description="Limit to: wikipedia, youtube, x, reddit, website, twitch, transcripts, web"),
        ] = None,
        voice: Annotated[
            bool, Field(description="Voice-verified video transcription (slow, precise)")
        ] = False,
        appearances: Annotated[
            int, Field(description="Also search N videos on other channels (needs voice)")
        ] = 0,
        web_search: int = 0,
        max_videos: int | None = None,
        max_posts: int | None = None,
    ) -> dict:
        """Start collecting a subject's public record from their confirmed accounts (background job)."""
        return _run(s.research_collect, name, sources, voice, appearances, web_search, max_videos, max_posts)

    @mcp.tool()
    def research_analyze(name: str, limit: int | None = None) -> dict:
        """Extract verified claims from collected content with the configured model (background job; costs tokens)."""
        return _run(s.research_analyze, name, limit)

    @mcp.tool()
    def job_status(job_id: str, wait_seconds: float = 0) -> dict:
        """Progress/result of a background job (optionally wait up to 30s for it to finish)."""
        return _run(s.job_status, job_id, wait_seconds)

    # ── replies ──────────────────────────────────────────────────

    @mcp.tool()
    def get_point_evidence(
        session_id: str | None = None,
        point_id: Annotated[str | None, Field(description="From a 'point' event")] = None,
        text: Annotated[str | None, Field(description="Or any statement to counter")] = None,
        opponent: str | None = None,
        limit: int = 10,
    ) -> dict:
        """The opponent's point plus the most relevant parts of their record, for writing a reply."""
        return _run(s.get_point_evidence, session_id, point_id, text, opponent, limit)

    @mcp.tool()
    def suggest_counters(
        session_id: str | None = None,
        point_id: str | None = None,
        text: str | None = None,
        opponent: str | None = None,
        style: Annotated[str, Field(description='"chat" (typed chat message) or "spoken"')] = "chat",
        max_chars: int | None = None,
    ) -> dict:
        """Reply options written by kickvoice's configured model, each already fact-checked."""
        return _run(s.suggest_counters, session_id, point_id, text, opponent, style, max_chars)

    @mcp.tool()
    def verify_message(
        opponent: str,
        message: Annotated[str, Field(description="The reply you intend to offer")],
        cited_refs: Annotated[
            list[str] | None, Field(description='Evidence refs it relies on, e.g. ["C12", "K7"]')
        ] = None,
        max_chars: int | None = None,
    ) -> dict:
        """Fact-check a reply: quotes must be verbatim first-hand, years backed by evidence, fits chat."""
        return _run(s.verify_message, opponent, message, cited_refs, max_chars)

    @mcp.tool()
    def format_for_chat(text: str, max_chars: int | None = None) -> dict:
        """Clean text into a single plain chat line within the length limit."""
        return _run(s.format_for_chat, text, max_chars)

    @mcp.tool()
    def copy_to_clipboard(text: Annotated[str, Field(description="The reply the user chose")]) -> dict:
        """Copy the chosen reply to the user's clipboard so they can paste it into chat."""
        return _run(s.copy_to_clipboard, text)

    # ── prompt: the live loop ────────────────────────────────────

    @mcp.prompt()
    def live_chat_copilot(stream_url: str, opponent: str = "") -> str:
        """Follow a debate stream and draft chat replies to the opponent's arguments."""
        who = f"the opponent is {opponent}" if opponent else "ask me who the opponent is"
        return (
            f"Follow the debate at {stream_url}; {who}.\n"
            "1. Check research_subjects/research_profile. If the opponent has little on record, "
            "offer to research them (research_find_identity, then collect/analyze in the background).\n"
            "2. start_live_session, then keep calling get_live_updates with the returned cursor.\n"
            "3. If the opponent's voice isn't recognised, use list_voices, tell me which voice you "
            "think is them and why, and enroll_voice only after I confirm.\n"
            "4. For each 'point' event: get_point_evidence (or suggest_counters), then give me 1-3 "
            "chat replies of at most "
            f"{s.config.agent.chat_max_chars} characters, each passed through verify_message. Show "
            "the source (date + link) under each reply so I can judge it, but keep the reply "
            "itself plain chat text.\n"
            "5. When I pick one, copy_to_clipboard it. I paste it myself.\n"
            "Skip banter and small talk. Keep going until I say stop, then stop_live_session."
        )

    return mcp


def run_stdio(config: Config):
    """Serve over stdio (what Claude Desktop / Claude Code launch)."""
    service = AgentService(config)
    server = build_server(service)
    try:
        server.run("stdio")
    finally:
        service.close()
