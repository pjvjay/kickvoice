"""
Run folders and trace logs for agent runs.

Every `extract-profiles` / `monitor` agent run gets its own folder:

    ~/.kickvoice/runs/20250601-201500_monitor_somechannel/
        trace.jsonl       every step: model calls, tool calls + results,
                          questions to the user, decisions, outputs
        report.md / .json what the run produced
        config.json       the exact configuration used (reproducibility)
        mcp_server.log    the MCP server's own log
        clips/            voice clips exported for review (extract-profiles)

The trace is append-only JSONL written as things happen, so a crashed or
interrupted run still leaves a complete record up to that point.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kickvoice.pipelines.session import source_slug

MAX_FIELD_CHARS = 4000


@dataclass
class RunDir:
    path: Path

    @classmethod
    def create(cls, root: Path, command: str, source: str) -> RunDir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(root) / f"{stamp}_{command}_{source_slug(source)}"
        path, n = base, 2
        while path.exists():
            path, n = base.with_name(f"{base.name}-{n}"), n + 1
        path.mkdir(parents=True)
        return cls(path)

    @property
    def trace(self) -> Path:
        return self.path / "trace.jsonl"

    @property
    def report_md(self) -> Path:
        return self.path / "report.md"

    @property
    def report_json(self) -> Path:
        return self.path / "report.json"

    @property
    def config(self) -> Path:
        return self.path / "config.json"

    @property
    def server_log(self) -> Path:
        return self.path / "mcp_server.log"

    @property
    def clips(self) -> Path:
        return self.path / "clips"


@dataclass
class TraceStats:
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    episodes: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "episodes": self.episodes,
            "by_tool": dict(sorted(self.by_tool.items(), key=lambda kv: -kv[1])),
        }


class Tracer:
    """Thread-safe JSONL trace writer that also keeps running totals."""

    def __init__(self, path: Path, echo: Callable[[str], None] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.stats = TraceStats()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 (closed in close())

    def event(self, kind: str, **data: Any) -> dict:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "t": round(time.monotonic() - self._started, 3),
            "kind": kind,
            **{k: _clip(v) for k, v in data.items()},
        }
        with self._lock:
            self._count(kind, data)
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        return record

    def say(self, message: str):
        """A human-readable progress line (also traced)."""
        self.event("note", message=message)
        if self.echo:
            self.echo(message)

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def _count(self, kind: str, data: dict):
        s = self.stats
        if kind == "llm_response":
            s.llm_calls += 1
            s.input_tokens += int(data.get("input_tokens") or 0)
            s.output_tokens += int(data.get("output_tokens") or 0)
        elif kind == "tool_result":
            s.tool_calls += 1
            s.by_tool[data.get("tool", "?")] = s.by_tool.get(data.get("tool", "?"), 0) + 1
            if data.get("is_error"):
                s.tool_errors += 1
        elif kind == "episode_start":
            s.episodes += 1


def read_trace(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clip(value: Any) -> Any:
    """Keep traces readable: long strings are truncated (their length is noted)."""
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS] + f"… [{len(value)} chars]"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > MAX_FIELD_CHARS:
            return text[:MAX_FIELD_CHARS] + f"… [{len(text)} chars]"
    return value
