"""
Shared machinery for goal-driven agents.

A goal agent has:
  - a context (config, MCP toolbox, tracer, terminal, model + policy, budget)
  - episodes it runs with a model ("llm" policy) or deterministic rules
    ("rules" policy — no model needed, same tools, same traces)
  - an orchestrator loop that follows the live session over MCP and wakes
    the agent only when there's something to decide
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio
import click

from kickvoice.agent.episode import EpisodeLimits, EpisodeResult, TokenBudget, run_episode
from kickvoice.agent.terminal import Terminal
from kickvoice.agent.toolbox import LocalTool, LocalToolError, ToolBox, ToolResult, schema
from kickvoice.agent.toolcalling import ToolLLM
from kickvoice.agent.trace import RunDir, Tracer
from kickvoice.config import Config
from kickvoice.console import fmt_clock

TERMINAL_STATES = ("finished", "stopped", "error")


class AgentFailed(RuntimeError):
    pass


@dataclass
class AgentContext:
    config: Config
    toolbox: ToolBox
    tracer: Tracer
    terminal: Terminal
    run_dir: RunDir
    policy: str  # "llm" | "rules"
    llm: ToolLLM | None
    budget: TokenBudget
    echo: Callable[[str], None] = click.echo


class GoalAgent:
    command = "agent"
    system_prompt = ""

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx
        self.config = ctx.config
        self.toolbox = ctx.toolbox
        self.tracer = ctx.tracer
        self.terminal = ctx.terminal
        self.echo = ctx.echo
        self.session_id: str | None = None
        self.session_status = "not started"
        self.stop_requested = False
        self.episodes: list[EpisodeResult] = []
        for tool in self.common_tools():
            self.toolbox.add_local(tool)
        self.toolbox.observers.append(self._observe)

    # ── entry point ──────────────────────────────────────────────

    async def run(self) -> dict:
        raise NotImplementedError

    # ── episodes ─────────────────────────────────────────────────

    async def episode(
        self,
        goal: str,
        brief: str,
        tools: list[str],
        model: str | None = None,
        limits: EpisodeLimits | None = None,
    ) -> EpisodeResult:
        if self.ctx.policy != "llm" or self.ctx.llm is None:
            raise AgentFailed("episode() needs the llm policy")
        agent_cfg = self.config.agent
        result = await run_episode(
            goal=goal,
            llm=self.ctx.llm,
            model=model or self.planning_model,
            toolbox=self.toolbox,
            tracer=self.tracer,
            system=self.system_prompt,
            brief=brief,
            tools=tools,
            budget=self.ctx.budget,
            limits=limits
            or EpisodeLimits(max_steps=agent_cfg.max_steps, timeout_s=agent_cfg.episode_timeout_s),
        )
        self.episodes.append(result)
        if result.outcome == "budget":
            self.tracer.say(
                click.style(
                    "⚠ Token budget for this run is used up "
                    f"(agent.max_run_tokens={agent_cfg.max_run_tokens}).",
                    fg="yellow",
                )
            )
        elif result.outcome == "error":
            self.tracer.say(click.style(f"⚠ Agent step failed: {result.error}", fg="yellow"))
        return result

    @property
    def planning_model(self) -> str:
        return self.config.agent.model or self.config.llm.analysis_model

    @property
    def reply_model(self) -> str:
        return self.config.agent.reply_model or self.config.llm.live_model

    # ── tools ────────────────────────────────────────────────────

    async def call(
        self, name: str, args: dict | None = None, actor: str = "orchestrator", **kw
    ) -> ToolResult:
        return await self.toolbox.call(name, args, actor=actor, **kw)

    def _observe(self, name: str, args: dict, result: ToolResult):
        """Keep track of the live session whoever (agent or rules) started it."""
        if name == "start_live_session" and result.ok and isinstance(result.data, dict):
            self.session_id = result.data.get("session_id", self.session_id)
            self.session_status = result.data.get("status", self.session_status)

    def common_tools(self) -> list[LocalTool]:
        async def ask_user(args):
            question = str(args.get("question") or "").strip()
            if not question:
                raise LocalToolError("question is required")
            details = str(args.get("details") or "").strip()
            if details:
                self.echo(click.style(details, fg="bright_black"))
            self.tracer.event(
                "user_question", question=question, details=details, options=args.get("options")
            )
            answer = await self.terminal.ask(question, args.get("options"))
            self.tracer.event("user_answer", question=question, answer=answer)
            if answer is None:
                return {
                    "answer": None,
                    "user_available": False,
                    "note": "No user available (non-interactive run). Do not act on this; "
                    "record it as pending.",
                }
            return {"answer": answer, "user_available": True}

        async def note(args):
            message = str(args.get("message") or "").strip()
            if message:
                self.tracer.say(click.style(f"🤖 {message}", fg="magenta"))
            return {"ok": True}

        async def finish(args):
            self.tracer.event("agent_summary", summary=args.get("summary", ""))
            return {"ok": True}

        return [
            LocalTool(
                "ask_user",
                "Ask the person running kickvoice a question and wait for their answer. "
                "Use for every approval. Put evidence (sample lines, clip path, your "
                "reasoning) in `details`.",
                schema(
                    {
                        "question": {"type": "string"},
                        "details": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    ["question"],
                ),
                ask_user,
            ),
            LocalTool(
                "note",
                "Tell the user briefly what you're doing or found.",
                schema({"message": {"type": "string"}}, ["message"]),
                note,
            ),
            LocalTool(
                "finish",
                "End this task, with a short summary of what you did.",
                schema({"summary": {"type": "string"}}, ["summary"]),
                finish,
            ),
        ]

    # ── following the live session ───────────────────────────────

    async def ensure_session(self, args: dict):
        """Start the live session ourselves if the agent didn't."""
        if self.session_id:
            return
        if self.ctx.policy == "llm":  # in rules mode this is simply how it starts
            self.tracer.say("   (the agent didn't start the live session — starting it)")
        result = await self.call("start_live_session", args)
        if not result.ok:
            raise AgentFailed(f"Couldn't start the live session: {result.data}")

    async def follow(self, on_event: Callable[[dict], Any], duration_s: float | None) -> None:
        """Poll the live session until it ends, the time is up, or the user stops it."""
        cursor = 0
        started = self.tracer.elapsed()

        def summarize(d: dict) -> dict:  # keep the trace compact: events are traced one by one
            return {
                "events": len(d.get("events", [])),
                "status": d.get("status"),
                "next_cursor": d.get("next_cursor"),
            }

        while not self.stop_requested:
            if duration_s and self.tracer.elapsed() - started >= duration_s:
                self.tracer.say(f"⏱  {fmt_clock(duration_s)} reached.")
                break
            result = await self.call(
                "get_live_updates",
                {
                    "session_id": self.session_id,
                    "cursor": cursor,
                    "wait_seconds": self.config.agent.poll_wait_s,
                },
                summarize=summarize,
            )
            if not result.ok:
                raise AgentFailed(f"Lost the live session: {result.data}")
            for event in result.data.get("events", []):
                self.tracer.event("live_event", **event)
                if event.get("type") == "status":
                    self.session_status = event.get("status", self.session_status)
                outcome = on_event(event)
                if hasattr(outcome, "__await__"):
                    await outcome
            cursor = result.data.get("next_cursor", cursor)
            self.session_status = result.data.get("status", self.session_status)
            if self.session_status in TERMINAL_STATES:
                if self.session_status == "error":
                    self.tracer.say(
                        click.style(f"⚠ Live session error: {result.data.get('error')}", fg="red")
                    )
                break

    async def stop_session(self):
        if self.session_id and self.session_status not in TERMINAL_STATES:
            result = await self.call("stop_live_session", {"session_id": self.session_id})
            if result.ok:
                self.session_status = result.data.get("status", "stopped")

    async def until_interrupted(self, body: Callable[[], Any]) -> bool:
        """Run `body`; Ctrl+C ends it (not the run), so we can still clean up and report."""
        interrupted = False
        async with anyio.create_task_group() as tg:

            async def watch_signals():
                nonlocal interrupted
                try:
                    with anyio.open_signal_receiver(signal.SIGINT) as signals:
                        async for _ in signals:
                            interrupted = True
                            self.stop_requested = True
                            tg.cancel_scope.cancel()
                            return
                except (NotImplementedError, RuntimeError, ValueError):
                    return  # no signal handling here (e.g. not the main thread)

            tg.start_soon(watch_signals)
            await body()
            tg.cancel_scope.cancel()
        if interrupted:
            self.tracer.say("\n⏹  Stopped by you.")
        return interrupted

    def print_line(self, event: dict, highlight: bool = False):
        who = event.get("speaker", "?")
        colour = (
            "cyan" if highlight else ("green" if event.get("attribution") == "confident" else "bright_black")
        )
        self.echo(
            click.style(f"[{fmt_clock(event.get('t', 0))}] {who}: ", fg=colour, bold=highlight)
            + event.get("text", "")
        )

    def stats(self) -> dict:
        s = self.tracer.stats.to_dict()
        s["episodes"] = [{"goal": e.goal, "outcome": e.outcome, "steps": e.steps} for e in self.episodes]
        s["elapsed_s"] = round(self.tracer.elapsed(), 1)
        return s
