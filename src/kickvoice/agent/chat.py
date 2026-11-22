"""
Chat-ready replies: formatting, fact-checking, clipboard.

A reply goes into a live chat where you can't edit or add sources, so before
one is offered it must (1) read like a normal chat message and fit the
length limit, and (2) not put words in the opponent's mouth: every quote in
it must be verbatim from their first-hand record (one unbroken passage, in
whatever kind of quote marks), and every year it mentions must be backed by
dated evidence.
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
# Quote marks that are never apostrophes: any of them opens or closes a quote.
_DOUBLE_MARKS = frozenset('"“”„‟«»＂')
# Single marks double as apostrophes, so their position decides (see quoted_spans).
_SINGLE_MARKS = frozenset("'‘’‚‛＇")
MIN_QUOTE_CHARS = 8  # shorter spans are scare quotes or single words, not quotations
_ELLIPSIS = re.compile(r"\.{2,}|…")
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

    for span in quoted_spans(message):
        quote, match = span.text, None
        for ev in first_hand:
            found = _quoted_in(quote, ev.quote) or (
                _quoted_in(quote, ev.text) if ev.text != ev.quote else None
            )
            if found:
                match = (ev, found)
                break
        if match is None:
            reported = any(_quoted_in(quote, e.quote) for e in evidence if e.authorship not in FIRST_HAND)
            if reported:
                why = " (it's from a third-party source — don't present it as their quote)"
            elif _ELLIPSIS.search(quote.strip(" .…")):
                why = (
                    ' — a quote can\'t skip words with "…": it has to be one unbroken passage '
                    "from their record; quote a single part or paraphrase without quote marks"
                )
            else:
                why = " — not found in their record; paraphrase without quote marks or drop it"
            if span.single and not reported:  # an apostrophe may have been read as a quote mark
                why += " (if the quote starts or ends elsewhere, put it in double quote marks)"
            issues.append(f'"{quote[:60]}" isn\'t something {opponent} said in their own words' + why)
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
    The source's own words for `quote`, if its words appear there as one
    unbroken run (case and punctuation aside), else None.

    An ellipsis is just punctuation here, so it can't stand in for words
    that were left out: "are … fine" is not a quote of "are not fine", and
    two sentences can't be spliced into one.
    """
    return find_quote(quote, source, exact=True)


@dataclass(frozen=True)
class QuotedSpan:
    text: str
    single: bool = False  # in single quote marks, which can be confused with apostrophes


def quoted_spans(message: str, min_chars: int = MIN_QUOTE_CHARS) -> list[QuotedSpan]:
    """
    Every passage `message` puts in quote marks, at least `min_chars` long.

    Double marks (straight, curly, low, guillemets) pair up left to right.
    Single marks are also apostrophes, so one only opens a quote when it
    doesn't follow a letter or digit ("don't", "Jane's"), and only closes one
    when it doesn't run into a word. A plural possessive ("the streamers'
    rights") looks exactly like a closing mark; when a quote could end in
    several places, the longest reading is the one returned, so a misquote
    can't hide behind the first apostrophe. Marks inside a span belong to it.

    This errs on the side of seeing a quote: "back in '21 … the streamers'
    view" reads as one. Such a reply is rejected with a hint, never passed
    unchecked.
    """
    spans, i = [], 0
    while i < len(message):
        end, single = None, False
        if message[i] in _DOUBLE_MARKS:
            end = next((k for k in range(i + 1, len(message)) if message[k] in _DOUBLE_MARKS), None)
        elif _opens_single(message, i):
            ends = _single_ends(message, i)
            end, single = (ends[-1] if ends else None), True
        if end is None:
            i += 1
            continue
        if end - i - 1 >= min_chars:
            spans.append(QuotedSpan(message[i + 1 : end], single))
        i = end + 1
    return spans


def has_quote_marks(text: str) -> bool:
    """True if `text` puts anything at all in quote marks."""
    return any(ch in _DOUBLE_MARKS for ch in text) or bool(quoted_spans(text, min_chars=1))


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _opens_single(message: str, i: int) -> bool:
    if message[i] not in _SINGLE_MARKS:
        return False
    before = message[i - 1] if i else " "
    after = message[i + 1] if i + 1 < len(message) else " "
    return not _is_word_char(before) and not after.isspace()


def _closes_single(message: str, k: int) -> bool:
    if message[k] not in _SINGLE_MARKS or message[k - 1].isspace():
        return False
    return k + 1 == len(message) or not _is_word_char(message[k + 1])


def _single_ends(message: str, start: int) -> list[int]:
    """Where the single-quoted span opened at `start` could end, up to the next opening mark."""
    ends = []
    for k in range(start + 2, len(message)):
        if _closes_single(message, k):
            ends.append(k)
        elif _opens_single(message, k):
            break
    return ends


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
