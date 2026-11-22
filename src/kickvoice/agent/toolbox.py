"""
The agent's tools: kickvoice's MCP tools plus a few local ones, behind one
calling convention, with every call traced.

MCP tools come from a real MCP connection — normally `kickvoice mcp`
spawned as a subprocess over stdio (exactly how Claude Desktop/Code would
use it); tests connect in-process. Local tools are things only the agent
runner can do: ask the person at the terminal, show them reply options,
record a decision.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kickvoice.agent.toolcalling import ToolSpec
from kickvoice.agent.trace import Tracer


@dataclass
class ToolResult:
    ok: bool
    data: Any  # parsed JSON for successful calls, else the error text
    text: str  # what the model sees


@dataclass
class LocalTool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], Awaitable[dict]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


class LocalToolError(Exception):
    """A local tool refused the call; the message goes back to the model."""


def schema(properties: dict, required: Iterable[str] = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


class ToolBox:
    def __init__(self, client, tracer: Tracer, local_tools: Iterable[LocalTool] = ()):
        self.client = client
        self.tracer = tracer
        self.local: dict[str, LocalTool] = {t.name: t for t in local_tools}
        self.mcp: dict[str, ToolSpec] = {}
        # Called after every tool call, whoever made it (agent or orchestrator).
        self.observers: list[Callable[[str, dict, ToolResult], None]] = []

    async def load(self) -> None:
        listed = await self.client.list_tools()
        self.mcp = {t.name: ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools}
        self.tracer.event("tools_loaded", mcp=sorted(self.mcp), local=sorted(self.local))

    def add_local(self, tool: LocalTool) -> None:
        self.local[tool.name] = tool

    def specs(self, names: Iterable[str]) -> list[ToolSpec]:
        specs = []
        for name in names:
            if name in self.local:
                specs.append(self.local[name].spec)
            elif name in self.mcp:
                specs.append(self.mcp[name])
            else:
                raise KeyError(f"No tool '{name}' (MCP server tools: {sorted(self.mcp)})")
        return specs

    async def call(
        self,
        name: str,
        args: dict | None = None,
        actor: str = "agent",
        summarize: Callable[[Any], Any] | None = None,
    ) -> ToolResult:
        """Call a tool. `summarize` shortens what's written to the trace (e.g. polls)."""
        args = args or {}
        self.tracer.event("tool_call", tool=name, actor=actor, args=args)
        started = time.monotonic()
        result = await (self._call_local(name, args) if name in self.local else self._call_mcp(name, args))
        traced = summarize(result.data) if (summarize and result.ok) else result.data
        self.tracer.event(
            "tool_result",
            tool=name,
            actor=actor,
            is_error=not result.ok,
            duration_s=round(time.monotonic() - started, 3),
            result=traced,
        )
        for observer in self.observers:
            observer(name, args, result)
        return result

    async def _call_local(self, name: str, args: dict) -> ToolResult:
        try:
            data = await self.local[name].fn(args)
        except (LocalToolError, KeyError, TypeError, ValueError) as e:
            return ToolResult(False, str(e), f"Error: {e}")
        return ToolResult(True, data, json.dumps(data, ensure_ascii=False, default=str))

    async def _call_mcp(self, name: str, args: dict) -> ToolResult:
        if name not in self.mcp:
            return ToolResult(False, f"Unknown tool '{name}'", f"Error: unknown tool '{name}'")
        response = await self.client.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in (response.content or []))
        if response.is_error:
            return ToolResult(False, text, f"Error: {text}")
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = text
        return ToolResult(True, data, text)


@asynccontextmanager
async def connect_mcp(config_path: Path, server_log: Path, service=None):
    """
    Connect to kickvoice's MCP server.

    By default it's spawned as `python -m kickvoice --config <run config> mcp`
    over stdio; its stderr goes to `server_log`. Pass an AgentService to
    connect in-process instead (tests, embedding).
    """
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if service is not None:
        from kickvoice.agent.server import build_server

        async with Client(build_server(service), cache=None) as client:
            yield client
        return

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kickvoice", "--config", str(config_path), "mcp"],
        # The SDK only forwards a minimal "safe" environment by default; the
        # server needs API keys, PATH (ffmpeg, streamlink, yt-dlp) and HOME.
        env=dict(os.environ),
    )
    with open(server_log, "a", encoding="utf-8") as errlog:
        async with Client(stdio_client(params, errlog=errlog), cache=None) as client:
            yield client
