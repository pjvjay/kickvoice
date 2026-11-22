"""
One agent episode: a model works towards a goal by calling tools until it's
done or hits a limit.

Long-running agents (a multi-hour stream) are run as many short episodes —
"identify this voice", "answer this argument" — each starting from a fresh,
compact brief. That keeps every model call small and the total cost bounded,
instead of one conversation that grows for hours.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import anyio

from kickvoice.agent.toolbox import ToolBox
from kickvoice.agent.toolcalling import ToolLLM
from kickvoice.agent.trace import Tracer

MAX_RESULT_CHARS_FOR_MODEL = 6000  # tool output fed back to the model per call
FINISH = "finish"


@dataclass
class EpisodeLimits:
    max_steps: int = 10  # model turns
    timeout_s: float = 120.0
    max_tokens: int = 1500  # per model response


class TokenBudget:
    """Shared cap on total tokens for a whole run (thread-safe)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def spend(self, tokens: int):
        with self._lock:
            self.used += tokens

    @property
    def exhausted(self) -> bool:
        return self.limit > 0 and self.used >= self.limit


@dataclass
class EpisodeResult:
    goal: str
    outcome: str  # done | finished | max_steps | timeout | budget | error
    final_text: str = ""
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None


async def run_episode(
    *,
    goal: str,
    llm: ToolLLM,
    model: str,
    toolbox: ToolBox,
    tracer: Tracer,
    system: str,
    brief: str,
    tools: list[str],
    limits: EpisodeLimits | None = None,
    budget: TokenBudget | None = None,
) -> EpisodeResult:
    """Run the model-and-tools loop for one goal. Never raises for model/tool failures."""
    limits = limits or EpisodeLimits()
    result = EpisodeResult(goal=goal, outcome="done")
    specs = toolbox.specs([*tools, FINISH] if FINISH in toolbox.local else tools)
    messages: list[dict] = [{"role": "user", "text": brief}]
    tracer.event("episode_start", goal=goal, model=model, tools=[s.name for s in specs], brief=brief)
    deadline = time.monotonic() + limits.timeout_s

    while True:
        if budget is not None and budget.exhausted:
            result.outcome = "budget"
            break
        if result.steps >= limits.max_steps:
            result.outcome = "max_steps"
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result.outcome = "timeout"
            break

        result.steps += 1
        tracer.event("llm_request", goal=goal, model=model, step=result.steps, messages=len(messages))
        try:
            with anyio.fail_after(remaining):
                turn = await anyio.to_thread.run_sync(
                    lambda: llm.respond(system, messages, specs, model, limits.max_tokens),
                    abandon_on_cancel=True,
                )
        except TimeoutError:
            result.outcome = "timeout"
            break
        except Exception as e:  # model/API failure: end the episode cleanly
            result.outcome, result.error = "error", str(e)
            tracer.event("llm_error", goal=goal, error=str(e))
            break
        if budget is not None:
            budget.spend(turn.input_tokens + turn.output_tokens)
        tracer.event(
            "llm_response",
            goal=goal,
            step=result.steps,
            model=turn.model,
            text=turn.text,
            stop_reason=turn.stop_reason,
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            latency_s=round(turn.latency_s, 2),
        )
        messages.append({"role": "assistant", "turn": turn})
        if turn.text:
            result.final_text = turn.text
        if not turn.tool_calls:
            break  # the model considers the goal met

        finished = False
        for call in turn.tool_calls:
            result.tool_calls.append(call.name)
            if call.parse_error:
                content, is_error = f"Error: {call.parse_error}", True
            elif call.name not in {s.name for s in specs}:
                content, is_error = f"Error: tool '{call.name}' isn't available for this goal", True
            else:
                outcome = await toolbox.call(call.name, call.arguments, actor=goal)
                content, is_error = outcome.text, not outcome.ok
                finished = finished or (call.name == FINISH and outcome.ok)
            if len(content) > MAX_RESULT_CHARS_FOR_MODEL:
                content = content[:MAX_RESULT_CHARS_FOR_MODEL] + " …[truncated]"
            messages.append(
                {
                    "role": "tool",
                    "call_id": call.id,
                    "name": call.name,
                    "content": content,
                    "is_error": is_error,
                }
            )
        if finished:
            result.outcome = "finished"
            break

    tracer.event(
        "episode_end",
        goal=goal,
        outcome=result.outcome,
        steps=result.steps,
        tool_calls=result.tool_calls,
        error=result.error,
        final_text=result.final_text,
    )
    return result
