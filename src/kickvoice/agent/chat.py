"""
Chat-ready replies: formatting, fact-checking, clipboard.

A reply goes into a live chat where you can't edit or add sources, so before
one is offered it must (1) read like a normal chat message and fit the
length limit, and (2) not put words in the opponent's mouth: every quote in
it must be verbatim from their first-hand record, and every year it mentions
must be backed by dated evidence.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from kickvoice.research.extract import find_quote
from kickvoice.research.store import FIRST_HAND, Evidence

_MARKDOWN = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*]\s+)", re.M)
_LABEL = re.compile(r"^\s*(counter|reply|response|chat|message)\s*\d*\s*[:\-–]\s*", re.I)
_QUOTED = re.compile(r"[\"“”]([^\"“”]{8,})[\"“”]")
_YEAR = re.compile(r"\b(19[89]\d|20\d\d)\b")
_URL = re.compile(r"https?://\S+")
_ATTRIBUTION = re.compile(
    r"\byou(?:'ve| have)? (?:said|tweeted|posted|wrote|claimed|argued|admitted)\b", re.I
)


def fit_to_chat(text: str, max_chars: int) -> str:
    """Plain, single-line chat text within `max_chars` (cut at a sentence if possible)."""
    text = _MARKDOWN.sub("", text.strip())  # markdown first, so "**Reply 1:**" → "Reply 1:"
    text = re.sub(r"\s+", " ", text).strip()
    text = _LABEL.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()  # the whole message wrapped in quotes
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sentence_end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if sentence_end >= max_chars * 0.3:  # a complete thought beats a cut-off one
        return cut[: sentence_end + 1].strip()
    return cut.rsplit(" ", 1)[0].rstrip(",;:-—") if " " in cut else cut


@dataclass
class Verification:
    ok: bool
    length: int
    max_chars: int
    issues: list[str] = field(default_factory=list)  # must fix before sending
    warnings: list[str] = field(default_factory=list)  # worth a look
    verified_quotes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "length": self.length,
            "max_chars": self.max_chars,
            "issues": self.issues,
            "warnings": self.warnings,
            "verified_quotes": self.verified_quotes,
        }


def verify_message(
    message: str, evidence: list[Evidence], max_chars: int, opponent: str = "them"
) -> Verification:
    """
    Check a reply against the opponent's record before it's offered.

    `evidence` should contain the items the message relies on (the refs it
    cites and/or search hits for its quotes).
    """
    issues, warnings, verified = [], [], []
    first_hand = [e for e in evidence if e.authorship in FIRST_HAND]

    for quote in _QUOTED.findall(message):
        match = None
        for ev in first_hand:
            found = _quoted_in(quote, ev.quote) or (
                _quoted_in(quote, ev.text) if ev.text != ev.quote else None
            )
            if found:
                match = (ev, found)
                break
        if match is None:
            reported = any(_quoted_in(quote, e.quote) for e in evidence if e.authorship not in FIRST_HAND)
            issues.append(
                f'"{quote[:60]}" isn\'t something {opponent} said in their own words'
                + (
                    " (it's from a third-party source — don't present it as their quote)"
                    if reported
                    else " — not found in their record; paraphrase without quote marks or drop it"
                )
            )
        else:
            ev, found = match
            verified.append({"quote": found, "ref": ev.ref, "date": ev.published_at, "url": ev.url})

    dates = {(e.published_at or "")[:4] for e in evidence if e.published_at}
    for year in sorted(set(_YEAR.findall(message))):
        if year not in dates:
            issues.append(f"mentions {year}, but none of the evidence is from {year}")

    if _ATTRIBUTION.search(message) and not first_hand:
        issues.append(
            f"says '{_ATTRIBUTION.search(message).group(0)}' but cites none of {opponent}'s own statements"
        )
    if len(message) > max_chars:
        issues.append(f"too long for chat: {len(message)} > {max_chars} characters")
    if _URL.search(message):
        warnings.append("contains a link — many stream chats block or hold links")
    if "\n" in message.strip():
        warnings.append("multiple lines — chat will flatten or reject it")
    return Verification(
        ok=not issues,
        length=len(message),
        max_chars=max_chars,
        issues=issues,
        warnings=warnings,
        verified_quotes=verified,
    )


def _quoted_in(quote: str, source: str) -> str | None:
    """
    Exact (case/punctuation-insensitive) match — no word changes. An
    ellipsis may join fragments that each appear exactly, in order.
    """
    fragments = [f.strip() for f in re.split(r"\.\.\.|…", quote) if f.strip()]
    if len(fragments) <= 1:
        return find_quote(quote, source, exact=True)
    found, position = [], 0
    for fragment in fragments:
        if len(fragment.split()) < 3:
            return None  # "…so…" style fragments can't be verified
        hit = find_quote(fragment, source[position:], exact=True)
        if hit is None:
            return None
        position = source.index(hit, position) + len(hit)
        found.append(hit)
    return " … ".join(found)


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Copy to the system clipboard. Returns (ok, how / why not)."""
    system = platform.system()
    candidates: list[list[str]]
    if system == "Darwin":
        candidates = [["pbcopy"]]
    elif system == "Windows":
        candidates = [["clip"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    last_error: str | None = None
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                check=True,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, cmd[0]
        except (subprocess.SubprocessError, OSError) as e:
            last_error = str(e)
    tools = ", ".join(c[0] for c in candidates)
    return False, (
        f"clipboard tool failed: {last_error}"
        if last_error
        else f"no clipboard tool found (install one of: {tools})"
    )
