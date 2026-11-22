"""
`extract-profiles` agent: find everyone who speaks on a stream and build
voice profiles for them — with the user's approval for each one.

    setup     agent starts a live session (recording audio for clips)
    listen    orchestrator follows the stream for the requested time
    identify  agent goes through each distinct voice: works out who it is
              (what's said, the channel, Wikidata for public figures), exports
              a clip, asks the user, enrolls approved voices, records a
              decision per voice
    report    report.md / report.json + the full trace
"""

from __future__ import annotations

import click

from kickvoice.agent.goals.base import GoalAgent
from kickvoice.agent.toolbox import LocalTool, LocalToolError, schema
from kickvoice.console import fmt_duration
from kickvoice.pipelines.session import channel_name

DECISIONS = ("created", "updated", "recognized", "skipped", "pending")

SYSTEM = """You are kickvoice's profile-extraction agent. You work through kickvoice's tools
(an MCP server) to build voice profiles for the people who speak on a live stream.

Rules you must follow:
- NEVER create or change a voice profile without the user's explicit approval via ask_user.
  If ask_user reports no user is available, record the voice as "pending" and move on.
- Identify people only from evidence: what is said on stream (introductions, names people
  use for each other), the channel name (the channel owner usually speaks the most), and
  Wikidata for public figures (research_find_identity). Say how confident you are and why.
- Only look up public figures. For anyone else, just ask the user what to call them.
- A name match on Wikidata is not proof — confirm with the user before linking research.
- Every voice with enough speech gets exactly one record_decision call.
- Be concise. Finish with `finish` and a short summary."""


class ExtractProfilesAgent(GoalAgent):
    command = "extract-profiles"
    system_prompt = SYSTEM

    def __init__(self, ctx, source: str, duration_s: float | None, min_speech_s: float):
        super().__init__(ctx)
        self.source = source
        self.duration_s = duration_s
        self.min_speech_s = min_speech_s
        self.decisions: list[dict] = []
        self.voices: list[dict] = []
        self.lines_heard = 0
        self.toolbox.add_local(self._record_decision_tool())

    # ── run ──────────────────────────────────────────────────────

    async def run(self) -> dict:
        self.echo(
            click.style(f"\n🤖 Profile-extraction agent ({self.ctx.policy}) — {self.source}", bold=True)
        )
        start_args = {
            "source": self.source,
            "keep_audio": True,
            **({"duration_s": self.duration_s} if self.duration_s else {}),
        }
        if self.ctx.policy == "llm":
            await self.episode(
                "setup",
                self._setup_brief(start_args),
                ["research_subjects", "list_live_sessions", "start_live_session", "note"],
            )
        await self.ensure_session(start_args)
        self.tracer.say(
            "🎧 Listening"
            + (f" for {fmt_duration(self.duration_s)}" if self.duration_s else "")
            + " (Ctrl+C to stop early)…"
        )

        interrupted = await self.until_interrupted(lambda: self.follow(self._on_event, self.duration_s))
        await self.stop_session()
        self.tracer.say(f"\n🔬 Heard {self.lines_heard} line(s). Identifying voices…")
        if not interrupted or self.lines_heard:
            if self.ctx.policy == "llm":
                await self.episode(
                    "identify",
                    self._identify_brief(),
                    [
                        "list_voices",
                        "export_voice_clip",
                        "research_subjects",
                        "research_profile",
                        "research_find_identity",
                        "research_add_subject",
                        "research_link_wikidata",
                        "enroll_voice",
                        "ask_user",
                        "record_decision",
                        "note",
                    ],
                )
            else:
                await self._identify_by_rules()
        voices = await self.call("list_voices", {"session_id": self.session_id})
        self.voices = voices.data.get("voices", []) if voices.ok else []
        return self.report()

    def _on_event(self, event: dict):
        if event.get("type") == "line":
            self.lines_heard += 1
            self.print_line(event)

    # ── briefs ───────────────────────────────────────────────────

    def _setup_brief(self, start_args: dict) -> str:
        return (
            f"Stream/source: {self.source}\n"
            f"Goal: build voice profiles for everyone who speaks.\n"
            f"Now: glance at existing research subjects (optional), then call "
            f"start_live_session with arguments {start_args} — keep_audio must be true so you "
            f"can export clips later. Then finish."
        )

    def _identify_brief(self) -> str:
        channel = channel_name(self.source)
        return (
            f"Stream: {self.source}" + (f" (channel owner: {channel})" if channel else "") + "\n"
            f"Live session id: {self.session_id} (stopped; audio was recorded)\n"
            f"Minimum speech to consider a voice: {self.min_speech_s:.0f}s\n\n"
            "Do this:\n"
            "1. list_voices.\n"
            "2. For each voice with at least the minimum speech:\n"
            "   - already recognised (has a speaker name): record_decision(decision='recognized').\n"
            "   - otherwise: decide who it likely is and why; export_voice_clip; then ask_user to "
            "confirm a name, putting your reasoning, 2-3 sample lines and the clip path in "
            "`details`.\n"
            "   - if approved: enroll_voice(session_id, utterance_id, name). If they're a public "
            "figure and the user wants research, also research_add_subject (or "
            "research_link_wikidata after confirming the record) with voice_profile=name.\n"
            "   - record_decision for the voice (created / updated / skipped / pending).\n"
            "3. finish with a short summary."
        )

    # ── rules policy (no model) ──────────────────────────────────

    async def _identify_by_rules(self):
        listed = await self.call("list_voices", {"session_id": self.session_id}, actor="rules")
        voices = listed.data.get("voices", []) if listed.ok else []
        for number, voice in enumerate(voices, 1):
            if voice["speech_seconds"] < self.min_speech_s or not voice["sample_lines"]:
                continue
            label = voice["speaker"] or f"voice {voice.get('voice_group') or number}"
            if voice["speaker"]:
                await self._decide(
                    label, voice, "recognized", voice["speaker"], "matched an existing voice profile"
                )
                continue
            anchor = voice["sample_lines"][0]["utterance_id"]
            clip = await self.call(
                "export_voice_clip",
                {
                    "session_id": self.session_id,
                    "utterance_id": anchor,
                    "out_dir": str(self.ctx.run_dir.clips),
                },
                actor="rules",
            )
            samples = "\n".join(f"   “{s['text']}”" for s in voice["sample_lines"][:3])
            details = (
                f"   {fmt_duration(voice['speech_seconds'])} of speech, {voice['lines']} line(s):\n"
                f"{samples}" + (f"\n   Clip: {clip.data['path']}" if clip.ok else "")
            )
            answer = await self.call(
                "ask_user",
                {
                    "question": f"Who is {label}? Type a name to create a voice profile, or press "
                    "Enter to skip.",
                    "details": details,
                },
                actor="rules",
            )
            name = answer.data.get("answer") if answer.ok else None
            clip_path = clip.data["path"] if clip.ok else None
            if name is None:
                await self._decide(label, voice, "pending", None, "no user to approve", clip_path)
            elif not name:
                await self._decide(label, voice, "skipped", None, "user skipped", clip_path)
            else:
                enrolled = await self.call(
                    "enroll_voice",
                    {"session_id": self.session_id, "utterance_id": anchor, "name": name},
                    actor="rules",
                )
                if enrolled.ok:
                    decision = "updated" if enrolled.data.get("merged") else "created"
                    await self._decide(label, voice, decision, name, "named by the user", clip_path)
                else:
                    await self._decide(
                        label, voice, "pending", name, f"enroll failed: {enrolled.data}", clip_path
                    )

    async def _decide(self, label, voice, decision, name, reasoning, clip=None):
        await self.call(
            "record_decision",
            {
                "voice": label,
                "voice_group": voice.get("voice_group"),
                "decision": decision,
                "name": name,
                "reasoning": reasoning,
                "clip": clip,
            },
            actor="rules",
        )

    # ── local tool ───────────────────────────────────────────────

    def _record_decision_tool(self) -> LocalTool:
        async def record_decision(args):
            decision = args.get("decision")
            if decision not in DECISIONS:
                raise LocalToolError(f"decision must be one of {DECISIONS}")
            entry = {
                "voice": str(args.get("voice") or ""),
                "voice_group": args.get("voice_group"),
                "decision": decision,
                "name": args.get("name"),
                "confidence": args.get("confidence"),
                "reasoning": args.get("reasoning", ""),
                "clip": args.get("clip"),
            }
            self.decisions.append(entry)
            colour = {"created": "green", "updated": "green", "recognized": "cyan"}.get(decision, "yellow")
            self.echo(
                click.style(
                    f"   • {entry['voice']}: {decision}" + (f" → {entry['name']}" if entry["name"] else ""),
                    fg=colour,
                )
            )
            self.tracer.event("decision", **entry)
            return {"recorded": True}

        return LocalTool(
            "record_decision",
            "Record the outcome for one voice (exactly once per voice).",
            schema(
                {
                    "voice": {"type": "string", "description": "Label shown by list_voices"},
                    "voice_group": {"type": ["integer", "null"]},
                    "decision": {"type": "string", "enum": list(DECISIONS)},
                    "name": {"type": ["string", "null"]},
                    "confidence": {"type": ["string", "null"], "description": "high / medium / low"},
                    "reasoning": {"type": "string"},
                    "clip": {"type": ["string", "null"]},
                },
                ["voice", "decision", "reasoning"],
            ),
            record_decision,
        )

    # ── report ───────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "command": self.command,
            "source": self.source,
            "policy": self.ctx.policy,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "lines_heard": self.lines_heard,
            "min_speech_s": self.min_speech_s,
            "voices": [
                {k: v for k, v in voice.items() if k != "sample_lines"}
                | {"samples": [s["text"] for s in voice.get("sample_lines", [])]}
                for voice in self.voices
            ],
            "decisions": self.decisions,
            "profiles_created": [d["name"] for d in self.decisions if d["decision"] == "created"],
            "profiles_updated": [d["name"] for d in self.decisions if d["decision"] == "updated"],
            "pending": [d["voice"] for d in self.decisions if d["decision"] == "pending"],
            "stats": self.stats(),
        }


def render_markdown(report: dict) -> str:
    out = [
        f"# Profile extraction — {report['source']}",
        "",
        f"- Policy: `{report['policy']}` · lines heard: {report['lines_heard']} · "
        f"session `{report['session_id']}` ({report['session_status']})",
        f"- Profiles created: {', '.join(report['profiles_created']) or '—'}",
        f"- Profiles updated: {', '.join(report['profiles_updated']) or '—'}",
        f"- Pending approval: {', '.join(report['pending']) or '—'}",
        "",
    ]
    out += ["## Voices", "", "| Voice | Speech | Lines | Recognised as |", "|---|---|---|---|"]
    for v in report["voices"]:
        out.append(
            f"| {v.get('label')} | {fmt_duration(v.get('speech_seconds', 0))} | "
            f"{v.get('lines')} | {v.get('speaker') or '—'} |"
        )
    out += ["", "## Decisions", ""]
    if not report["decisions"]:
        out.append("_No decisions recorded._")
    for d in report["decisions"]:
        out.append(
            f"- **{d['voice']}** → {d['decision']}"
            + (f" as **{d['name']}**" if d.get("name") else "")
            + (f" ({d['confidence']} confidence)" if d.get("confidence") else "")
        )
        if d.get("reasoning"):
            out.append(f"  - Why: {d['reasoning']}")
        if d.get("clip"):
            out.append(f"  - Clip: `{d['clip']}`")
    s = report["stats"]
    out += [
        "",
        "## Agent activity",
        "",
        f"- {s['llm_calls']} model call(s), {s['tool_calls']} tool call(s) "
        f"({s['tool_errors']} failed), {s['input_tokens'] + s['output_tokens']:,} tokens, "
        f"{fmt_duration(s['elapsed_s'])}",
        "- Episodes: "
        + (
            ", ".join(f"{e['goal']} ({e['outcome']}, {e['steps']} steps)" for e in s["episodes"])
            or "— (rules policy)"
        ),
        "- Full step-by-step log: `trace.jsonl`",
    ]
    return "\n".join(out) + "\n"
