"""
Terminal output helpers.

One place for message styling and duration formatting, so pipelines just say
*what* happened and the presentation stays consistent. Click strips colour
automatically when output isn't a terminal.
"""

from __future__ import annotations

import click


def echo(message: str = "", **kwargs):
    click.echo(message, **kwargs)


def info(message: str):
    click.echo(message)


def success(message: str):
    click.secho(f"✅ {message}", fg="green")


def warn(message: str):
    click.secho(f"⚠  {message}", fg="yellow")


def error(message: str):
    click.secho(f"❌ {message}", fg="red", err=True)


def hint(message: str):
    click.secho(message, fg="bright_black")


def heading(message: str):
    click.secho(message, bold=True)


def table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None):
    """Print a simple left-aligned text table."""
    widths = widths or [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 if rows else len(str(h)) + 2
        for i, h in enumerate(headers)
    ]
    line = "".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()
    click.secho(line, bold=True)
    click.echo("-" * max(len(line), sum(widths) - 2))
    for row in rows:
        click.echo("".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


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
