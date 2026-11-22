"""
Terminal output helpers.

One place for message styling and duration formatting, so pipelines just say
*what* happened and the presentation stays consistent. Click strips colour
automatically when output isn't a terminal.
"""
from __future__ import annotations

import click

# When kickvoice runs as an MCP server over stdio, stdout carries the protocol
# itself, so every human-readable message must go to stderr instead.
_to_stderr = False


def use_stderr(enabled: bool = True):
    global _to_stderr
    _to_stderr = enabled


def echo(message: str = "", **kwargs):
    kwargs.setdefault("err", _to_stderr)
    click.echo(message, **kwargs)


def info(message: str):
    click.echo(message, err=_to_stderr)


def success(message: str):
    click.secho(f"✅ {message}", fg="green", err=_to_stderr)


def warn(message: str):
    click.secho(f"⚠  {message}", fg="yellow", err=_to_stderr)


def error(message: str):
    click.secho(f"❌ {message}", fg="red", err=True)


def hint(message: str):
    click.secho(message, fg="bright_black", err=_to_stderr)


def heading(message: str):
    click.secho(message, bold=True, err=_to_stderr)


def table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None):
    """Print a simple left-aligned text table."""
    widths = widths or [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 if rows else len(str(h)) + 2
        for i, h in enumerate(headers)
    ]
    line = "".join(str(h).ljust(w) for h, w in zip(headers, widths)).rstrip()
    click.secho(line, bold=True, err=_to_stderr)
    click.echo("-" * max(len(line), sum(widths) - 2), err=_to_stderr)
    for row in rows:
        click.echo("".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip(), err=_to_stderr)


def fmt_duration(seconds: float) -> str:
    """62 → '1m 02s', 3700 → '1h 01m', 9 → '9s'."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_clock(seconds: float) -> str:
    """Offset within a capture: 62 → '01:02', 3700 → '1:01:40'."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
