"""
Chat-ready replies: checks a draft before you paste it into stream chat.

The agent writes the reply; this makes sure it fits and is honest:
  - one line, no markdown, within the chat's length limit
  - anything in quotation marks must really appear in the evidence it cites
    (fabricated or "tidied" quotes are the fastest way to lose an argument)
  - links flagged where chats usually block them
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from kickvoice.research.extract import find_quote

_QUOTED = re.compile(r"[\"“”]([^\"“”]{12,}?)[\"“”]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


@dataclass
class ChatCheck:
    ok: bool
    message: str                 # cleaned, ready to paste
    length: int
    max_chars: int
    issues: list[str] = field(default_factory=list)      # must fix before posting
    warnings: list[str] = field(default_factory=list)    # worth a look
    verified_quotes: list[str] = field(default_factory=list)


def clean_message(text: str) -> str:
    """Strip markdown and layout a chat box can't show; collapse to one line."""
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'`":
        t = t[1:-1]
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<!\w)[*_](\S(?:.*?\S)?)[*_](?!\w)", r"\1", t)
    t = re.sub(r"^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 \2", t)            # [text](url) → text url
    return re.sub(r"\s+", " ", t).strip()


def check_message(text: str, evidence: list[str], max_chars: int = 500,
                  allow_links: bool = False) -> ChatCheck:
    message = clean_message(text)
    check = ChatCheck(ok=True, message=message, length=len(message), max_chars=max_chars)
    if not message:
        check.issues.append("The message is empty.")
    if len(message) > max_chars:
        check.issues.append(f"Too long for chat: {len(message)} chars, limit {max_chars} "
                            f"(cut {len(message) - max_chars}).")
    for quote in _QUOTED.findall(message):
        if len(quote.split()) < 3:
            continue                                   # scare quotes, not a citation
        if any(find_quote(quote, source) for source in evidence):
            check.verified_quotes.append(quote)
        elif evidence:
            check.issues.append(f'Quote not found in the cited evidence: "{quote[:80]}". '
                                "Use their exact words or paraphrase without quotation marks.")
        else:
            check.issues.append(f'Quote with no evidence cited: "{quote[:80]}". '
                                "Pass the evidence refs it comes from, or paraphrase.")
    if _URL.search(message) and not allow_links:
        check.warnings.append("Contains a link — many chats block or auto-mod links.")
    if message.count("@") > 2:
        check.warnings.append("Mentions several people — can read as spam.")
    check.ok = not check.issues
    return check


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Copy to the system clipboard using whatever the OS provides."""
    candidates = []
    if sys.platform == "darwin":
        candidates.append(["pbcopy"])
    elif sys.platform.startswith("win"):
        candidates.append(["clip"])
    else:
        candidates += [["wl-copy"], ["xclip", "-selection", "clipboard"],
                       ["xsel", "--clipboard", "--input"], ["clip.exe"]]   # clip.exe: WSL
    for cmd in candidates:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5,
                               capture_output=True)
                return True, cmd[0]
            except (subprocess.SubprocessError, OSError) as e:
                return False, f"{cmd[0]} failed: {e}"
    return False, "no clipboard tool found (install wl-clipboard or xclip on Linux)"
