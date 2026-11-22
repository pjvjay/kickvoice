"""The terminal's stdin handling: lines arrive, and reading never keeps a finished run alive."""

import io
import os
import subprocess
import sys
import textwrap
import threading
import time

import anyio
import pytest

import kickvoice.agent.terminal as terminal_module
from kickvoice.agent.terminal import POLL_S, StdinLines, Terminal

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="stdin can't be polled on Windows")


@pytest.fixture
def typed(monkeypatch):
    """A real, pollable stdin (pytest's own isn't one). Returns the end a 'user' types into."""
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r", encoding="utf-8")
    keyboard = os.fdopen(write_fd, "wb", buffering=0)
    monkeypatch.setattr(sys, "stdin", stdin)
    yield keyboard
    keyboard.close()  # also releases any reader still blocked on it
    stdin.close()


# ── StdinLines ───────────────────────────────────────────────────


@posix_only
def test_poll_gives_up_quickly_when_nothing_is_typed(typed):
    started = time.monotonic()
    assert StdinLines().poll() is None
    assert time.monotonic() - started < POLL_S + 1.0


@posix_only
def test_poll_returns_whole_lines_one_at_a_time(typed):
    lines = StdinLines()
    typed.write(b"hel")
    assert lines.poll(0.05) is None  # no newline yet: not a line
    typed.write(b"lo\nsecond\r\nthi")
    assert lines.poll(0.05) == "hello"
    assert lines.poll(0.05) == "second"  # already read; "\r\n" endings are stripped too
    assert lines.poll(0.05) is None
    typed.write("rd é".encode())
    assert lines.poll(0.05) is None
    typed.close()
    assert lines.poll(0.05) == "third é"  # a last line without a newline still counts
    with pytest.raises(EOFError):
        lines.poll(0.05)
    with pytest.raises(EOFError):  # and it stays closed
        lines.poll(0.05)


@posix_only
def test_poll_treats_a_failing_stdin_as_end_of_input(typed, monkeypatch):
    def gone(*_a):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(terminal_module.select, "select", gone)
    with pytest.raises(EOFError):
        StdinLines().poll(0.05)


def test_poll_treats_a_closed_stdin_as_end_of_input(typed):
    sys.stdin.close()
    with pytest.raises(EOFError):
        StdinLines().poll(0.05)


def test_poll_reads_streams_without_a_file_descriptor(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("one\r\n\ntwo"))
    lines = StdinLines()
    assert [lines.poll(), lines.poll(), lines.poll()] == ["one", "", "two"]
    with pytest.raises(EOFError):
        lines.poll()


def test_poll_falls_back_to_readline_where_files_cannot_be_polled(monkeypatch, typed):
    monkeypatch.setattr(terminal_module, "CAN_SELECT_FILES", False)  # as on Windows
    typed.write(b"typed on windows\n")
    assert StdinLines().poll() == "typed on windows"


@posix_only
def test_overlapping_polls_do_not_both_read(typed):
    """A poll abandoned by a cancelled wait can overlap the next; only one may read the line."""
    lines, results = StdinLines(), []
    pollers = [threading.Thread(target=lambda: results.append(lines.poll(1.0))) for _ in range(2)]
    for poller in pollers:
        poller.start()
    time.sleep(0.1)  # both are waiting
    typed.write(b"only one line\n")
    for poller in pollers:
        poller.join(timeout=5)
    assert not any(p.is_alive() for p in pollers), "a poll is blocked in read()"
    assert sorted(results, key=str) == [None, "only one line"]


# ── Terminal ─────────────────────────────────────────────────────


@posix_only
def test_reader_delivers_lines_and_leaves_no_thread_behind(typed):
    """Regression: the reader's thread stayed blocked in readline() after the run, so Python couldn't exit."""
    before = set(threading.enumerate())

    async def run():
        terminal, got = Terminal(interactive=True, echo=lambda _m: None), []
        async with anyio.create_task_group() as tg:
            tg.start_soon(terminal.read_stdin)
            typed.write(b"2\n")
            async for command in terminal.commands():
                got.append(command)
                break
            await anyio.sleep(POLL_S * 2)  # the reader is waiting for input again
            tg.cancel_scope.cancel()  # the run is over
        return got

    assert anyio.run(run) == ["2"]
    deadline = time.monotonic() + 5
    while (
        left := [t for t in set(threading.enumerate()) - before if t.is_alive()]
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not left, f"still waiting on stdin: {left}"


@posix_only
def test_ask_reads_the_answer_itself_when_no_reader_runs(typed):
    async def run():
        terminal = Terminal(interactive=True, echo=lambda _m: None)
        typed.write(b"  Jane \n")
        first = await terminal.ask("Who is voice 1?")
        typed.close()
        return first, await terminal.ask("Who is voice 2?")  # EOF → no answer

    assert anyio.run(run) == ("Jane", None)


@posix_only
def test_an_unanswered_question_can_be_cancelled(typed):
    """E.g. Ctrl+C while extract-profiles waits for a name: the wait ends, nothing stays blocked."""
    before = set(threading.enumerate())

    async def run():
        terminal = Terminal(interactive=True, echo=lambda _m: None)
        with anyio.move_on_after(POLL_S) as scope:
            await terminal.ask("Who is voice 1?")
        return scope.cancelled_caught

    assert anyio.run(run) is True
    deadline = time.monotonic() + 5
    while (
        left := [t for t in set(threading.enumerate()) - before if t.is_alive()]
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not left


# ── a real process on a real terminal ────────────────────────────

PROGRAM = textwrap.dedent(
    """
    import anyio
    from kickvoice.agent.terminal import Terminal

    async def main():
        terminal = Terminal(echo=lambda _m: None)
        assert terminal.interactive, "stdin should be a terminal here"
        async with anyio.create_task_group() as tg:
            tg.start_soon(terminal.read_stdin)
            await anyio.sleep(0.5)  # the run: nothing is typed
            tg.cancel_scope.cancel()

    anyio.run(main)
    print("run finished", flush=True)
    """
)


@posix_only
def test_process_exits_after_the_run_without_enter_being_pressed():
    """Regression: `monitor` printed its report, then stayed alive until the user pressed Enter."""
    import pty

    keyboard, terminal = pty.openpty()  # the child's stdin is a terminal nobody types into
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", PROGRAM],
            stdin=terminal,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.write(keyboard, b"\n")  # what the user had to do
            proc.kill()
            proc.communicate()
            pytest.fail("the process was still waiting on stdin 30 s after its run finished")
    finally:
        os.close(terminal)
        os.close(keyboard)
    assert proc.returncode == 0, err
    assert out.strip() == "run finished"
