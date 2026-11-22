"""
The person at the terminal, as seen by an agent run.

One reader owns stdin. A line answers the pending question if the agent
asked one; otherwise it's a command (e.g. "2" = copy reply 2, "q" = quit).
Without a terminal (piped, CI) questions return None, so agents record
their proposals as pending instead of acting on them.

Stdin is read in short polls rather than with a blocking `readline()`: a
thread parked in a blocking read can't be interrupted, and Python waits for
it at exit, so a finished run would hang until the user pressed Enter.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from collections.abc import Callable

import anyio
import click

POLL_S = 0.2  # longest a stdin poll waits, so also the longest one outlives its run
CAN_SELECT_FILES = sys.platform != "win32"  # Windows' select() only accepts sockets


class StdinLines:
    """
    Lines from stdin, waiting at most `POLL_S` per call.

    Reads the file descriptor directly once `select()` says there's input
    (in a terminal, that's when a whole line has been entered), so no call
    blocks for long. Where stdin can't be polled (no real descriptor, as in
    tests and embedded consoles, or on Windows), it falls back to a blocking
    `readline()`.
    """

    def __init__(self) -> None:
        self._pending = b""  # bytes read but not yet returned as a line
        self._eof = False
        # One poll at a time: a poll abandoned by a cancelled wait may still be
        # running when the next starts, and two reads for one line would leave
        # the second blocked.
        self._lock = threading.Lock()

    def poll(self, timeout: float = POLL_S) -> str | None:
        """
        The next line without its newline, or None if none arrived within
        `timeout` seconds. Raises EOFError once stdin is closed.
        """
        with self._lock:
            return self._poll(timeout)

    def _poll(self, timeout: float) -> str | None:
        line = self._take_line()
        if line is not None:
            return line
        if self._eof:
            raise EOFError
        fd = _pollable_fd(sys.stdin)
        if fd is None:
            try:
                text = sys.stdin.readline()
            except (OSError, ValueError):  # closed
                text = ""
            if not text:
                self._eof = True
                raise EOFError
            return text.rstrip("\r\n")
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                return None
            chunk = os.read(fd, 4096)
        except (OSError, ValueError):  # stdin was closed under us
            chunk = b""
        if chunk:
            self._pending += chunk
            return self._take_line()
        self._eof = True
        if self._pending:  # a last line with no newline before EOF
            rest, self._pending = self._pending, b""
            return self._decode(rest)
        raise EOFError

    def _take_line(self) -> str | None:
        line, newline, rest = self._pending.partition(b"\n")
        if not newline:
            return None
        self._pending = rest
        return self._decode(line)

    @staticmethod
    def _decode(line: bytes) -> str:
        encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
        return line.decode(encoding, errors="replace").rstrip("\r")


def _pollable_fd(stream) -> int | None:
    """The stream's file descriptor if `select()` can wait on it, else None."""
    if not CAN_SELECT_FILES:
        return None
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError):  # io.UnsupportedOperation is both of the last two
        return None


class Terminal:
    def __init__(self, interactive: bool | None = None, echo: Callable[[str], None] = click.echo):
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.echo = echo
        self._commands_send, self._commands = anyio.create_memory_object_stream[str](100)
        self._answer: tuple[anyio.Event, list[str]] | None = None
        self._lock = anyio.Lock()
        self._reader_running = False
        self._stdin = StdinLines()
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
                line = await self._next_line()
                return line.strip() if line is not None else None  # EOF → no answer
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
            while (line := await self._next_line()) is not None:
                self.feed(line)
        finally:
            self._reader_running = False

    async def _next_line(self) -> str | None:
        """
        Wait for one line from stdin (None at EOF).

        Cancelling this doesn't leave a thread stuck in a read (wherever stdin
        can be polled): each poll returns within POLL_S, so the process can
        exit as soon as a run ends.
        """
        while True:
            try:
                line = await anyio.to_thread.run_sync(self._stdin.poll, abandon_on_cancel=True)
            except EOFError:
                return None
            if line is not None:
                return line


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
