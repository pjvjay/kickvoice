"""
The person at the terminal, as seen by an agent run.

One reader owns stdin. A line answers the pending question if the agent
asked one; otherwise it's a command (e.g. "2" = copy reply 2, "q" = quit).
Without a terminal (piped, CI) questions return None, so agents record
their proposals as pending instead of acting on them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import anyio
import click


class Terminal:
    def __init__(self, interactive: bool | None = None, echo: Callable[[str], None] = click.echo):
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.echo = echo
        self._commands_send, self._commands = anyio.create_memory_object_stream[str](100)
        self._answer: tuple[anyio.Event, list[str]] | None = None
        self._lock = anyio.Lock()
        self._reader_running = False
        # Command lines delivered so far. Consumers count what they've handled;
        # comparing the two is race-free (a line handed straight to a waiting
        # receiver never shows up in the stream's buffer).
        self.commands_delivered = 0

    async def ask(self, question: str, options: list[str] | None = None) -> str | None:
        """Ask and wait for an answer (None without an interactive user)."""
        if not self.interactive:
            return None
        async with self._lock:  # one question at a time
            hint = f" [{' / '.join(options)}]" if options else ""
            self.echo(click.style(f"\n❓ {question}{hint}", fg="cyan", bold=True))
            self.echo(click.style("   > ", fg="cyan") + "(type your answer and press Enter)")
            if not self._reader_running:
                # Nobody else is reading stdin (e.g. extract-profiles): read the answer here.
                line = await anyio.to_thread.run_sync(sys.stdin.readline, abandon_on_cancel=True)
                return line.strip() if line else None  # EOF → no answer
            event, box = anyio.Event(), []
            self._answer = (event, box)
            await event.wait()
            self._answer = None
            return box[0].strip() if box else None

    def feed(self, line: str) -> None:
        """Route one input line (called by the reader, or directly in tests)."""
        if self._answer is not None:
            event, box = self._answer
            box.append(line)
            event.set()
        else:
            self.commands_delivered += 1
            self._commands_send.send_nowait(line.strip())

    async def commands(self):
        """Async iterator over command lines typed while no question is pending."""
        async for line in self._commands:
            yield line

    async def read_stdin(self) -> None:
        """Reader task: feed stdin lines until EOF."""
        if not self.interactive:
            return
        self._reader_running = True
        try:
            while True:
                line = await anyio.to_thread.run_sync(sys.stdin.readline, abandon_on_cancel=True)
                if not line:
                    return
                self.feed(line.rstrip("\n"))
        finally:
            self._reader_running = False


class ScriptedTerminal(Terminal):
    """For tests and unattended runs: canned answers, recorded questions."""

    def __init__(self, answers: list[str | None] | None = None, interactive: bool = True):
        super().__init__(interactive=interactive, echo=lambda _m: None)
        self.answers = list(answers or [])
        self.questions: list[str] = []

    async def ask(self, question: str, options: list[str] | None = None) -> str | None:
        self.questions.append(question)
        if not self.interactive:
            return None
        return self.answers.pop(0) if self.answers else None

    async def read_stdin(self) -> None:
        """Input comes from feed(), never from the real stdin."""
        return
