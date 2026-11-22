"""
Tool-calling language models for agents.

One internal conversation format, two wire formats:
  - Anthropic Messages API "tool_use" / "tool_result" blocks
  - OpenAI-compatible Chat Completions "tools" / "tool_calls" (OpenAI,
    Ollama, vLLM, LM Studio, …)

Internal messages:
    {"role": "user", "text": str}
    {"role": "assistant", "turn": AssistantTurn}
    {"role": "tool", "call_id": str, "name": str, "content": str, "is_error": bool}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from kickvoice.config import Config
from kickvoice.research.llm import LLMError, _api_key, _post


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None  # the model sent arguments that weren't valid JSON


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ToolLLM(Protocol):
    def respond(
        self, system: str, messages: list[dict], tools: list[ToolSpec], model: str, max_tokens: int = 1500
    ) -> AssistantTurn: ...


class AnthropicToolLLM:
    def __init__(self, config: Config):
        cfg = config.llm
        self.base_url = (cfg.base_url or "https://api.anthropic.com").rstrip("/")
        self.key_env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        self.timeout = cfg.timeout_s

    def respond(self, system, messages, tools, model, max_tokens=1500) -> AssistantTurn:
        started = time.monotonic()
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self.to_wire(messages),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
            ],
        }
        data = _post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": _api_key(self.key_env),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body=body,
            timeout=self.timeout,
        )
        turn = AssistantTurn(
            stop_reason=data.get("stop_reason", ""),
            model=data.get("model", model),
            latency_s=time.monotonic() - started,
        )
        for block in data.get("content", []):
            if block.get("type") == "text":
                turn.text += block.get("text", "")
            elif block.get("type") == "tool_use":
                turn.tool_calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = data.get("usage", {})
        turn.input_tokens, turn.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return turn

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                turn: AssistantTurn = m["turn"]
                content: list[dict] = [{"type": "text", "text": turn.text}] if turn.text else []
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in turn.tool_calls
                ]
                wire.append({"role": "assistant", "content": content})
            else:
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["call_id"],
                    "content": m["content"],
                    "is_error": bool(m.get("is_error")),
                }
                # Consecutive tool results go in one user message.
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        return wire


class OpenAIToolLLM:
    def __init__(self, config: Config):
        cfg = config.llm
        self.base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.key_env = cfg.api_key_env or "OPENAI_API_KEY"
        self.timeout = cfg.timeout_s

    def respond(self, system, messages, tools, model, max_tokens=1500) -> AssistantTurn:
        started = time.monotonic()
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.key_env, "").strip()
        if key:
            headers["authorization"] = f"Bearer {key}"
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *self.to_wire(messages)],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
                }
                for t in tools
            ],
        }
        data = _post(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e
        turn = AssistantTurn(
            text=message.get("content") or "",
            stop_reason=choice.get("finish_reason", ""),
            model=data.get("model", model),
            latency_s=time.monotonic() - started,
        )
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args, error = (json.loads(raw) if isinstance(raw, str) else raw), None
                if not isinstance(args, dict):
                    args, error = {}, "arguments must be a JSON object"
            except json.JSONDecodeError as e:
                args, error = {}, f"arguments weren't valid JSON: {e}"
            turn.tool_calls.append(ToolCall(call.get("id", ""), fn.get("name", ""), args, error))
        usage = data.get("usage") or {}
        turn.input_tokens, turn.output_tokens = (
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return turn

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                turn: AssistantTurn = m["turn"]
                entry: dict = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in turn.tool_calls
                    ]
                wire.append(entry)
            else:
                wire.append({"role": "tool", "tool_call_id": m["call_id"], "content": m["content"]})
        return wire


def make_tool_llm(config: Config) -> ToolLLM:
    provider = config.llm.provider.lower()
    if provider == "anthropic":
        _api_key(config.llm.api_key_env or "ANTHROPIC_API_KEY")  # fail fast
        return AnthropicToolLLM(config)
    if provider in ("openai", "openai-compatible", "ollama"):
        return OpenAIToolLLM(config)
    raise LLMError(f"Unknown llm.provider '{config.llm.provider}' (use anthropic or openai)")
