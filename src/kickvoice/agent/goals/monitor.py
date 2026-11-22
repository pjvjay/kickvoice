"""
`monitor` agent: follow a debate stream, know who's speaking, and answer the
opponent's arguments with chat-ready replies grounded in their record.

    setup     agent checks what's known about the opponent (may start research),
              then starts the live session
    live      orchestrator follows the stream; for each of the opponent's points
              the agent drafts replies (reply episodes); if the opponent isn't
              recognised, the agent works out who's who (identify episode)
    replies   every proposed reply is fact-checked by the orchestrator itself
              (verify_message over MCP) before it's shown; the user types its
              number to copy it. Nothing is ever posted to chat.
"""

from __future__ import annotations

from collections import deque

import anyio
import click

from kickvoice.agent.chat import has_quote_marks
from kickvoice.agent.episode import EpisodeLimits
from kickvoice.agent.goals.base import GoalAgent
from kickvoice.agent.toolbox import LocalTool, LocalToolError, schema
from kickvoice.console import fmt_duration

SYSTEM = """You are kickvoice's debate copilot. You work through kickvoice's tools (an MCP
server) while a live debate stream is running, helping a viewer reply in the stream chat.

Rules you must follow:
- Replies are short chat messages (plain text, one line, within the character limit), written
  as the viewer, addressed to the opponent. No markdown, hashtags, emojis or links.
- Put quote marks (of any kind) ONLY around words copied exactly from the opponent's
  first-hand, sincere evidence, as one unbroken passage: never skip words with "...".
  "Reported about them" evidence is never their words. Mention a year only if the evidence is
  from it. Never invent facts, statistics or quotes.
- Attack the argument, not the person.
- Offer replies only through propose_replies; it fact-checks them. If some are rejected, fix
  them (or drop them) and propose again.
- Never create profiles, link identities or spend money (research_analyze) without the user's
  approval via ask_user. If no user is available, don't.
- Be fast and concise; skip banter. End with `finish`."""


class MonitorAgent(GoalAgent):
    command = "monitor"
    system_prompt = SYSTEM

    def __init__(self, ctx, source: str, opponent: str | None, me: str | None, duration_s: float | None):
        super().__init__(ctx)
        self.source = source
        self.opponent_hint = opponent
        self.me = me
        self.duration_s = duration_s
        self.research_subject: str | None = None  # set from what the session resolves
        self.points: dict[str, dict] = {}
        self.replies: list[dict] = []  # everything offered to the user (numbered)
        self.rejected: list[dict] = []
        self.copied: list[dict] = []
        self.lines_heard = 0
        self.opponent_heard = False
        self._queue: deque[dict] = deque()
        self._skipped_points = 0
        self._wake = anyio.Event()
        self._identify_requested = False
        self._identify_attempted: set = set()
        self._commands_handled = 0
        self._commands_baseline = ctx.terminal.commands_delivered  # lines from before this run
        self._working = False
        self._voice_speech: dict = {}
        self.toolbox.add_local(self._propose_replies_tool())

    # ── run ──────────────────────────────────────────────────────

    async def run(self) -> dict:
        self.echo(click.style(f"\n🤖 Debate copilot ({self.ctx.policy}) — {self.source}", bold=True))
        start_args = {
            "source": self.source,
            **({"opponent": self.opponent_hint} if self.opponent_hint else {}),
            **({"me": self.me} if self.me else {}),
        }
        if self.ctx.policy == "llm":
            await self.episode(
                "setup",
                self._setup_brief(start_args),
                [
                    "research_subjects",
                    "research_profile",
                    "research_find_identity",
                    "research_collect",
                    "job_status",
                    "start_live_session",
                    "ask_user",
                    "note",
                ],
            )
        await self.ensure_session(start_args)
        if self.terminal.interactive:
            self.echo(
                click.style(
                    "   Type a reply's number to copy it · r = list replies · q = stop", fg="bright_black"
                )
            )

        async def live():
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._worker)
                tg.start_soon(self._commands, tg.cancel_scope)
                tg.start_soon(self.terminal.read_stdin)
                await self.follow(self._on_event, self.duration_s)
                await self._drain()
                tg.cancel_scope.cancel()

        await self.until_interrupted(live)
        await self.stop_session()
        return self.report()

    # ── events → work ────────────────────────────────────────────

    def _on_event(self, event: dict):
        kind = event.get("type")
        if kind == "line":
            self.lines_heard += 1
            self.opponent_heard |= bool(event.get("is_opponent"))
            self.print_line(event, highlight=bool(event.get("is_opponent")))
        elif kind == "point":
            self.points[event["point_id"]] = event
            self._queue.append({"kind": "reply", "point": event})
            while sum(1 for w in self._queue if w["kind"] == "reply") > self.config.debate.max_queued_points:
                stale = next(w for w in self._queue if w["kind"] == "reply")
                self._queue.remove(stale)
                self._skipped_points += 1
                self.tracer.event(
                    "point_skipped", point_id=stale["point"]["point_id"], reason="newer points waiting"
                )
            self._wake.set()
        elif kind == "voices":
            self._maybe_identify(event.get("voices", []))

    def _maybe_identify(self, voices: list[dict]):
        """Ask the agent who's who once an unrecognised voice has talked for a while."""
        if any(v.get("is_opponent") for v in voices) or self._identify_requested:
            return
        threshold = self.config.agent.identify_after_s
        for v in voices:
            key = v.get("voice_group") or v.get("label")
            if (
                not v.get("speaker")
                and v.get("speech_seconds", 0) >= threshold
                and key not in self._identify_attempted
            ):
                self._identify_attempted.add(key)
                self._identify_requested = True
                self._queue.appendleft({"kind": "identify", "voices": voices})
                self._wake.set()
                return

    async def _worker(self):
        while True:
            if not self._queue:
                self._wake = anyio.Event()
                await self._wake.wait()
                continue
            work = self._queue.popleft()
            self._working = True  # off the queue but not done: _drain must wait
            try:
                if work["kind"] == "identify":
                    await self._identify(work["voices"])
                    self._identify_requested = False
                else:
                    await self._reply(work["point"])
            finally:
                self._working = False

    async def _drain(self):
        """
        Once the stream has ended, briefly finish what's in flight: queued
        replies, and commands the user already typed (a "copy" must not be
        cancelled half-way).
        """

        def busy():
            unhandled = self.terminal.commands_delivered - self._commands_baseline - self._commands_handled
            return bool(self._queue) or self._working or unhandled > 0

        with anyio.move_on_after(self.config.agent.reply_timeout_s):
            while busy():
                await anyio.sleep(0.05)

    # ── episodes ─────────────────────────────────────────────────

    async def _reply(self, point: dict):
        self.echo(
            click.style(
                f"\n💬 [{point['point_id']}] {point.get('opponent', 'opponent')}: ", fg="magenta", bold=True
            )
            + point["text"][:200]
        )
        if self.ctx.policy == "llm":
            await self.episode(
                f"reply:{point['point_id']}",
                self._reply_brief(point),
                [
                    "get_point_evidence",
                    "research_search",
                    "suggest_counters",
                    "verify_message",
                    "format_for_chat",
                    "propose_replies",
                    "note",
                ],
                model=self.reply_model,
                limits=EpisodeLimits(
                    max_steps=6, timeout_s=self.config.agent.reply_timeout_s, max_tokens=900
                ),
            )
        else:
            await self._reply_by_rules(point)

    async def _identify(self, voices: list[dict]):
        self.tracer.say(click.style("\n🔎 Someone unrecognised is talking — working out who…", fg="magenta"))
        if self.ctx.policy == "llm":
            await self.episode(
                "identify",
                self._identify_brief(voices),
                [
                    "list_voices",
                    "research_subjects",
                    "research_find_identity",
                    "research_add_subject",
                    "enroll_voice",
                    "set_opponent",
                    "ask_user",
                    "note",
                ],
            )
        else:
            await self._identify_by_rules(voices)

    def _setup_brief(self, start_args: dict) -> str:
        return (
            f"Stream: {self.source}\n"
            f"Opponent: {self.opponent_hint or 'not given — ask the user who the viewer disagrees with, or set it later'}\n"
            f"Viewer's own voice profile: {self.me or 'none'}\n"
            f"Chat reply limit: {self.config.agent.chat_max_chars} characters\n\n"
            "Do this quickly:\n"
            "1. research_profile for the opponent (if known). If their record is thin and they "
            "have confirmed accounts, you may start research_collect in the background.\n"
            f"2. start_live_session with {start_args} (add opponent= if you established it).\n"
            "3. note one line about what you found, then finish."
        )

    def _reply_brief(self, point: dict) -> str:
        context = (
            "\n".join(f"  {line}" for line in point.get("context", [])[-4:]) if point.get("context") else ""
        )
        return (
            f"Live session: {self.session_id}\n"
            f"Opponent: {point.get('opponent')} · research subject: {self.research_subject or 'none'}\n"
            f"Their point [{point['point_id']}]: {point['text']}\n"
            + (f"Recent lines:\n{context}\n" if context else "")
            + f"\nWrite 1-3 chat replies (max {self.config.agent.chat_max_chars} chars) a viewer "
            "could paste. Use get_point_evidence(session_id, point_id) for their record (or "
            "suggest_counters for ready-made ones), then propose_replies. If it's banter, not an "
            "argument, propose nothing and finish."
        )

    def _identify_brief(self, voices: list[dict]) -> str:
        lines = []
        for v in voices[:6]:
            samples = " | ".join(s["text"][:90] for s in v.get("sample_lines", [])[:3])
            lines.append(f"- {v.get('label')} ({fmt_duration(v.get('speech_seconds', 0))}): {samples}")
        return (
            f"Live session: {self.session_id} · stream: {self.source}\n"
            f"Opponent (research subject): {self.research_subject or 'not set'}\n"
            f"Voices so far:\n" + "\n".join(lines) + "\n\n"
            "The opponent isn't recognised yet. Work out which voice is the opponent (what they "
            "argue, names used). Ask the user to confirm (include sample lines in details). If "
            "confirmed: enroll_voice(session_id, utterance_id, name, research_subject=...) and "
            "set_opponent(session_id, opponent=<research subject or voice name>). Then finish."
        )

    # ── rules policy (no model) ──────────────────────────────────

    async def _reply_by_rules(self, point: dict):
        """Without a model: offer their own past words on the topic ("receipts")."""
        evidence = await self.call(
            "get_point_evidence",
            {"session_id": self.session_id, "point_id": point["point_id"], "limit": 8},
            actor="rules",
        )
        if not evidence.ok:
            return
        limit = self.config.agent.chat_max_chars
        replies, quoted = [], set()
        for item in evidence.data.get("evidence", []):
            if not (item.get("first_hand") and item.get("sincere")):
                continue
            quote = item["quote"].strip().rstrip(".")
            if len(quote.split()) < 3 or quote.lower() in quoted:
                continue  # too short, or already offered (claim + excerpt)
            quoted.add(quote.lower())
            when = f" ({item['date'][:7]})" if item.get("date") else ""
            text = f'you said "{quote}"{when} — how does that fit with what you just said?'
            if len(text) > limit:
                continue
            replies.append({"text": text, "refs": [item["ref"]]})
            if len(replies) == 2:
                break
        if replies:
            await self.call(
                "propose_replies", {"point_id": point["point_id"], "replies": replies}, actor="rules"
            )
        else:
            self.echo(click.style("   (nothing relevant in their record for this point)", fg="bright_black"))

    async def _identify_by_rules(self, voices: list[dict]):
        unknown = [v for v in voices if not v.get("speaker") and v.get("sample_lines")]
        if not unknown:
            return
        v = max(unknown, key=lambda x: x.get("speech_seconds", 0))
        name = self.research_subject or self.opponent_hint
        samples = "\n".join(f"   “{s['text']}”" for s in v["sample_lines"][:3])
        answer = await self.call(
            "ask_user",
            {
                "question": f"Is {v['label']} the opponent{f' ({name})' if name else ''}? Type their name "
                "to enroll this voice, or Enter to skip.",
                "details": f"   {fmt_duration(v['speech_seconds'])} of speech:\n{samples}",
            },
            actor="rules",
        )
        chosen = answer.data.get("answer") if answer.ok else None
        if not chosen:
            return
        enrolled = await self.call(
            "enroll_voice",
            {
                "session_id": self.session_id,
                "utterance_id": v["sample_lines"][0]["utterance_id"],
                "name": chosen,
                **({"research_subject": self.research_subject} if self.research_subject else {}),
            },
            actor="rules",
        )
        if enrolled.ok:
            await self.call(
                "set_opponent",
                {"session_id": self.session_id, "opponent": self.research_subject or chosen},
                actor="rules",
            )

    # ── replies: verification + display ──────────────────────────

    def _observe(self, name, args, result):
        super()._observe(name, args, result)
        # Trust what the session resolved, not the name that was passed in: an
        # opponent can be a voice profile with no research record behind it.
        if name in ("set_opponent", "start_live_session") and result.ok and "research_subject" in result.data:
            self.research_subject = result.data["research_subject"]

    def _propose_replies_tool(self) -> LocalTool:
        async def propose_replies(args):
            point_id = str(args.get("point_id") or "")
            raw = args.get("replies") or []
            if not isinstance(raw, list) or not raw:
                raise LocalToolError("replies must be a non-empty list of {text, refs}")
            accepted, rejected = [], []
            for item in raw[:3]:
                text = str(item.get("text") or "").strip() if isinstance(item, dict) else str(item)
                refs = item.get("refs") or [] if isinstance(item, dict) else []
                check = await self._verify(text, refs)
                if check["ok"]:
                    number = len(self.replies) + 1
                    reply = {
                        "number": number,
                        "point_id": point_id,
                        "text": check["text"],
                        "refs": refs,
                        "sources": check.get("verified_quotes", []),
                        "warnings": check.get("warnings", []),
                    }
                    self.replies.append(reply)
                    accepted.append({"number": number, "text": reply["text"]})
                    self._show(reply)
                else:
                    entry = {"point_id": point_id, "text": text, "issues": check["issues"]}
                    self.rejected.append(entry)
                    rejected.append({"text": text, "issues": check["issues"]})
            self.tracer.event("replies_proposed", point_id=point_id, accepted=accepted, rejected=rejected)
            return {
                "accepted": accepted,
                "rejected": rejected,
                "hint": "Fix or drop rejected replies and propose again." if rejected else "",
            }

        return LocalTool(
            "propose_replies",
            "Offer chat replies to the user. Each is fact-checked first; rejected ones come back "
            "with the reasons.",
            schema(
                {
                    "point_id": {"type": "string"},
                    "replies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Evidence refs the reply relies on",
                                },
                            },
                            "required": ["text"],
                        },
                    },
                },
                ["point_id", "replies"],
            ),
            propose_replies,
        )

    async def _verify(self, text: str, refs: list[str]) -> dict:
        """Our own check — we don't take the agent's word that a reply is accurate."""
        formatted = await self.call("format_for_chat", {"text": text})
        text = formatted.data["text"] if formatted.ok else text
        if not text:
            return {"ok": False, "issues": ["empty reply"], "text": text}
        if not self.research_subject:
            if has_quote_marks(text):
                return {
                    "ok": False,
                    "text": text,
                    "issues": ["quotes can't be checked: the opponent has no research record"],
                }
            return {"ok": True, "text": text, "warnings": ["unchecked: no research record"]}
        check = await self.call(
            "verify_message", {"opponent": self.research_subject, "message": text, "cited_refs": refs}
        )
        if not check.ok:
            return {"ok": False, "text": text, "issues": [str(check.data)]}
        return {**check.data, "text": text}

    def _show(self, reply: dict):
        self.echo(click.style(f"   [{reply['number']}] ", fg="green", bold=True) + reply["text"])
        for source in reply["sources"][:2]:
            where = " · ".join(x for x in (source.get("date"), source.get("url")) if x)
            self.echo(click.style(f"       ↳ “{source['quote'][:80]}” {where}", fg="bright_black"))

    async def _commands(self, scope):
        async for line in self.terminal.commands():
            try:
                if await self._command(line.strip().lower(), scope):
                    return
            finally:
                self._commands_handled += 1

    async def _command(self, command: str, scope) -> bool:
        """Handle one typed command; True means stop."""
        if command in ("q", "quit", "stop"):
            self.stop_requested = True
            self.tracer.event("user_command", command="quit")
            scope.cancel()
            return True
        if command in ("r", "replies"):
            for reply in self.replies[-6:]:
                self._show(reply)
        elif command.isdigit():
            await self._copy(int(command))
        elif command:
            self.echo("   Type a reply number to copy it, r = list replies, q = stop")
        return False

    async def _copy(self, number: int):
        reply = next((r for r in self.replies if r["number"] == number), None)
        if reply is None:
            self.echo(f"   No reply [{number}]")
            return
        result = await self.call("copy_to_clipboard", {"text": reply["text"]}, actor="user")
        copied = result.ok and result.data.get("copied")
        self.tracer.event("user_command", command="copy", number=number, copied=bool(copied))
        if copied:
            self.copied.append(reply)
            self.echo(click.style(f"   📋 Copied [{number}] — paste it into chat", fg="green"))
        else:
            detail = result.data.get("detail") if result.ok else result.data
            self.echo(
                click.style(
                    f"   Couldn't copy ({detail}). Reply [{number}]:\n   {reply['text']}", fg="yellow"
                )
            )

    # ── report ───────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "command": self.command,
            "source": self.source,
            "policy": self.ctx.policy,
            "opponent": self.research_subject,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "lines_heard": self.lines_heard,
            "opponent_heard": self.opponent_heard,
            "points": [
                {"point_id": pid, "t": p.get("t"), "text": p["text"]} for pid, p in self.points.items()
            ],
            "points_skipped": self._skipped_points,
            "replies": self.replies,
            "rejected_replies": self.rejected,
            "copied": [r["number"] for r in self.copied],
            "stats": self.stats(),
        }


def render_markdown(report: dict) -> str:
    out = [
        f"# Debate copilot — {report['source']}",
        "",
        f"- Policy: `{report['policy']}` · opponent: {report['opponent'] or '—'} · "
        f"lines heard: {report['lines_heard']} · session `{report['session_id']}` "
        f"({report['session_status']})",
        f"- Arguments detected: {len(report['points'])} ({report['points_skipped']} skipped as stale) · "
        f"replies offered: {len(report['replies'])} · rejected by fact-check: "
        f"{len(report['rejected_replies'])} · copied: {len(report['copied'])}",
        "",
    ]
    by_point: dict[str, list[dict]] = {}
    for r in report["replies"]:
        by_point.setdefault(r["point_id"], []).append(r)
    out += ["## Arguments and replies", ""]
    if not report["points"]:
        out.append("_No arguments from the opponent were detected._")
    for p in report["points"]:
        out.append(f"### [{p['point_id']}] “{p['text'][:300]}”")
        for r in by_point.get(p["point_id"], []):
            mark = " 📋 copied" if r["number"] in report["copied"] else ""
            out.append(f"- **[{r['number']}]** {r['text']}{mark}")
            for s in r.get("sources", []):
                out.append(
                    f"  - source: “{s['quote']}” ({s.get('date') or 'undated'}) {s.get('url') or ''}".rstrip()
                )
        if not by_point.get(p["point_id"]):
            out.append("- _no reply offered_")
        out.append("")
    if report["rejected_replies"]:
        out += ["## Replies rejected by the fact-check", ""]
        for r in report["rejected_replies"]:
            out.append(f"- “{r['text']}” — {'; '.join(r['issues'])}")
        out.append("")
    s = report["stats"]
    out += [
        "## Agent activity",
        "",
        f"- {s['llm_calls']} model call(s), {s['tool_calls']} tool call(s) ({s['tool_errors']} failed), "
        f"{s['input_tokens'] + s['output_tokens']:,} tokens, {fmt_duration(s['elapsed_s'])}",
        f"- Episodes: {len(s['episodes'])}",
        "- Full step-by-step log: `trace.jsonl`",
    ]
    return "\n".join(out) + "\n"
