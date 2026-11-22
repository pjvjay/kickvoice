"""
Spawn a goal agent for a CLI command and produce its outputs.

    run folder  ← config.json (exact settings used)
    MCP server  ← spawned: `python -m kickvoice --config <run>/config.json mcp`
    agent       ← goal (extract-profiles | monitor) × policy (llm | rules)
    outputs     → trace.jsonl (live), report.md, report.json, clips/
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anyio
import click

from kickvoice.agent.episode import TokenBudget
from kickvoice.agent.goals import extract as extract_goal
from kickvoice.agent.goals import monitor as monitor_goal
from kickvoice.agent.goals.base import AgentContext, AgentFailed
from kickvoice.agent.terminal import Terminal
from kickvoice.agent.toolbox import ToolBox, connect_mcp
from kickvoice.agent.toolcalling import ToolLLM, make_tool_llm
from kickvoice.agent.trace import RunDir, Tracer
from kickvoice.config import Config

if sys.version_info < (3, 11):  # pragma: no cover - Python 3.10
    from exceptiongroup import BaseExceptionGroup

logger = logging.getLogger(__name__)

POLICIES = ("llm", "rules")


@dataclass
class RunOutcome:
    run_dir: RunDir
    report: dict
    ok: bool = True
    error: str | None = None


def run_agent(
    command: str,
    config: Config,
    options: dict,
    policy: str = "llm",
    *,
    llm: ToolLLM | None = None,
    terminal: Terminal | None = None,
    service=None,
    echo: Callable[[str], None] = click.echo,
) -> RunOutcome:
    """
    Run the agent for `command` ("extract-profiles" or "monitor").

    `service` connects to an in-process MCP server instead of spawning one
    (tests); `llm` overrides the configured model client.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if policy == "llm" and llm is None:
        llm = make_tool_llm(config)  # raises LLMError with setup guidance
    run_dir = RunDir.create(Path(config.data_dir) / "runs", command, options["source"])
    config.save(run_dir.config)
    tracer = Tracer(run_dir.trace)
    tracer.event("run_start", command=command, policy=policy, options=options, run_dir=str(run_dir.path))
    echo(click.style(f"📁 Run folder: {run_dir.path}", fg="bright_black"))

    outcome = RunOutcome(run_dir, {})

    async def main():
        async with connect_mcp(run_dir.config, run_dir.server_log, service) as client:
            toolbox = ToolBox(client, tracer)
            await toolbox.load()
            ctx = AgentContext(
                config=config,
                toolbox=toolbox,
                tracer=tracer,
                terminal=terminal or Terminal(echo=echo),
                run_dir=run_dir,
                policy=policy,
                llm=llm,
                budget=TokenBudget(config.agent.max_run_tokens),
                echo=echo,
            )
            tracer.echo = echo
            agent = _make_agent(command, ctx, options)
            outcome.report = await agent.run()

    try:
        anyio.run(main)
    except KeyboardInterrupt:
        outcome.ok, outcome.error = False, "interrupted before the agent was ready"
    except Exception as wrapped:
        error = root_cause(wrapped)  # task groups wrap errors in ExceptionGroups
        if isinstance(error, AgentFailed):
            outcome.ok, outcome.error = False, str(error)
        else:  # unexpected: keep the traceback in the log
            logger.error("Agent run failed", exc_info=error)
            outcome.ok, outcome.error = False, f"{type(error).__name__}: {error}"
            if service is None and _connection_lost(error):
                outcome.error = _server_stopped(run_dir.server_log, outcome.error)
        tracer.event("run_error", error=outcome.error)
    _write_outputs(command, outcome, tracer)
    return outcome


def root_cause(error: BaseException) -> BaseException:
    """Unwrap single-exception ExceptionGroups (as raised by anyio task groups)."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _connection_lost(error: BaseException) -> bool:
    """The MCP connection dropped (the server process went away), as opposed to a tool failing."""
    from mcp import MCPError
    from mcp_types import CONNECTION_CLOSED

    return isinstance(error, MCPError) and error.code == CONNECTION_CLOSED


def _server_stopped(server_log: Path, error: str, tail: int = 2, width: int = 200) -> str:
    """
    One line saying the spawned MCP server died, with the end of its log.

    "MCPError: Connection closed" on its own says nothing about why; the
    reason (a crash in a native library, a missing dependency) is whatever
    the server last wrote to stderr.
    """
    try:
        lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    last = [line.strip()[:width] for line in lines if line.strip()][-tail:]
    message = f"kickvoice's MCP server stopped unexpectedly ({error})"
    if last:
        message += ". Its log ends: " + " | ".join(last)
    return f"{message}. Full log: {server_log}"


def _make_agent(command: str, ctx: AgentContext, options: dict):
    if command == "extract-profiles":
        return extract_goal.ExtractProfilesAgent(
            ctx,
            source=options["source"],
            duration_s=options.get("duration_s"),
            min_speech_s=options.get("min_speech_s", ctx.config.extract.min_speaker_speech_s),
        )
    if command == "monitor":
        return monitor_goal.MonitorAgent(
            ctx,
            source=options["source"],
            opponent=options.get("opponent"),
            me=options.get("me"),
            duration_s=options.get("duration_s"),
        )
    raise ValueError(f"No agent for command '{command}'")


def _write_outputs(command: str, outcome: RunOutcome, tracer: Tracer):
    report = dict(outcome.report) if outcome.report else {"command": command}
    report.setdefault(
        "stats", tracer.stats.to_dict() | {"episodes": [], "elapsed_s": round(tracer.elapsed(), 1)}
    )
    report["ok"], report["error"] = outcome.ok, outcome.error
    report["outputs"] = {"trace": str(outcome.run_dir.trace), "report": str(outcome.run_dir.report_md)}
    outcome.report = report
    outcome.run_dir.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    renderer = extract_goal.render_markdown if command == "extract-profiles" else monitor_goal.render_markdown
    try:
        markdown = renderer(report)
    except (KeyError, TypeError):  # the run failed before producing a full report
        markdown = (
            f"# {command}\n\nThe run ended early: {outcome.error or 'no results'}.\n\nSee `trace.jsonl`.\n"
        )
    if outcome.error:
        markdown = f"> ⚠ Run ended with an error: {outcome.error}\n\n" + markdown
    outcome.run_dir.report_md.write_text(markdown, encoding="utf-8")
    tracer.event("run_end", ok=outcome.ok, error=outcome.error, stats=tracer.stats.to_dict())
    tracer.close()
