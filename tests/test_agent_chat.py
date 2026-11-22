"""Chat-ready replies: formatting, fact-checking, clipboard."""

import subprocess

import pytest

from kickvoice.agent import chat
from kickvoice.agent.chat import copy_to_clipboard, fit_to_chat, verify_message
from kickvoice.research.store import ABOUT, SELF, Evidence


def ev(quote, authorship=SELF, date="2021-03-01", kind="claim", id_=1):
    return Evidence(kind, id_, "paraphrase", quote, authorship, "https://x.com/j/status/1", date, "x", "")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**Counter 1:** You said the *opposite* in 2021.", "You said the *opposite* in 2021."),
        ('"rent control shrinks supply, you know that"', "rent control shrinks supply, you know that"),
        ("Reply: line one\n\nline two", "line one line two"),
        ("- `bullet` point", "bullet point"),
    ],
)
def test_fit_to_chat_cleans(raw, expected):
    assert fit_to_chat(raw, 300) == expected


def test_fit_to_chat_trims_at_sentence_or_word():
    text = "First sentence is short. " + "Second sentence goes on and on " * 10
    assert fit_to_chat(text, 60) == "First sentence is short."
    out = fit_to_chat("word " * 100, 53)
    assert len(out) <= 53 and not out.endswith(" ") and out.endswith("word")


def test_verified_first_hand_quote_passes():
    v = verify_message(
        'you literally said "rent control reduces housing supply" lol',
        [ev("Rent control reduces housing supply. Build more.")],
        300,
        "Jane",
    )
    assert v.ok, v.issues
    assert v.verified_quotes[0]["quote"] == "Rent control reduces housing supply"


def test_fabricated_quote_is_an_issue():
    v = verify_message(
        'you said "rent control is always evil"', [ev("Rent control reduces housing supply.")], 300, "Jane"
    )
    assert not v.ok and "not found in their record" in v.issues[0]


def test_third_party_words_cant_be_quoted_as_theirs():
    v = verify_message(
        'you said "she has opposed rent control"',
        [ev("She has opposed rent control.", authorship=ABOUT)],
        300,
        "Jane",
    )
    assert not v.ok and "third-party" in v.issues[0]


def test_unsupported_year_and_attribution():
    v = verify_message("back in 2019 you argued the opposite", [ev("x y z", date="2021-03-01")], 300, "Jane")
    assert any("2019" in i for i in v.issues)
    v = verify_message("you tweeted the opposite", [ev("x y z", authorship=ABOUT)], 300, "Jane")
    assert any("cites none of Jane's own statements" in i for i in v.issues)


def test_length_and_warnings():
    v = verify_message("a" * 301, [], 300)
    assert not v.ok and "too long" in v.issues[0]
    v = verify_message("see https://example.com", [], 300)
    assert v.ok and "link" in v.warnings[0]


def test_plain_reasoning_passes():
    assert verify_message("ok but who pays for that?", [], 300).ok


def test_clipboard_uses_available_tool(monkeypatch):
    calls = []
    monkeypatch.setattr(chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(chat.shutil, "which", lambda name: name if name == "xclip" else None)
    monkeypatch.setattr(chat.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw["input"])))
    assert copy_to_clipboard("héllo") == (True, "xclip")
    assert calls == [(["xclip", "-selection", "clipboard"], "héllo".encode())]


def test_clipboard_reports_missing_or_failing_tool(monkeypatch):
    monkeypatch.setattr(chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(chat.shutil, "which", lambda name: None)
    ok, detail = copy_to_clipboard("x")
    assert not ok and "wl-copy" in detail

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(chat.shutil, "which", lambda name: name)
    monkeypatch.setattr(chat.subprocess, "run", boom)
    ok, detail = copy_to_clipboard("x")
    assert not ok and "failed" in detail


@pytest.mark.parametrize(
    "message",
    [
        'you said "we should abolish the minimum wage"',  # source: raise
        'you said "we should raise minimum wages"',  # word changed
        'you said "we should raise the wage for everyone"',  # "minimum" dropped
    ],
)
def test_one_word_changes_are_misquotes(message):
    source = ev("Honestly I think we should raise the minimum wage for everyone.")
    v = verify_message(message, [source], 300)
    assert not v.ok, message


def test_verbatim_excerpt_is_fine():
    source = ev("Honestly I think we should raise the minimum wage for everyone.")
    assert verify_message('you said "raise the minimum wage"', [source], 300).ok


def test_exact_quote_ignoring_case_and_punctuation():
    source = ev("Honestly, I think we should RAISE the minimum wage for everyone.")
    assert verify_message('you said "i think we should raise the minimum wage"', [source], 300).ok


def test_ellipsis_joins_exact_fragments_in_order():
    source = ev("We should raise the minimum wage. Also, housing supply has to grow fast.")
    assert verify_message(
        'you said "we should raise the minimum wage ... housing supply has to grow"', [source], 300
    ).ok
    assert not verify_message(
        'you said "housing supply has to grow ... we should raise the minimum wage"', [source], 300
    ).ok
