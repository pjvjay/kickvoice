"""Chat-ready replies: formatting, fact-checking, clipboard."""

import subprocess

import pytest

from kickvoice.agent import chat
from kickvoice.agent.chat import (
    copy_to_clipboard,
    fit_to_chat,
    has_quote_marks,
    quoted_spans,
    verify_message,
)
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


# ── quotes: one unbroken passage ─────────────────────────────────


@pytest.mark.parametrize("gap", [" … ", " ... ", "…", " [...] ", " .. ", " — "])
def test_an_ellipsis_cannot_drop_a_word(gap):
    """Regression: "are … fine" was accepted as a quote of "are not fine"."""
    source = ev("Sponsored streams are not fine for streamers who hide them.")
    v = verify_message(f'you said "sponsored streams are{gap}fine for streamers"', [source], 300, "Jane")
    assert not v.ok and not v.verified_quotes
    assert "sponsored streams are" in v.issues[0]


def test_skipped_words_get_their_own_explanation():
    source = ev("Sponsored streams are not fine for streamers who hide them.")
    v = verify_message('you said "sponsored streams are … fine for streamers"', [source], 300, "Jane")
    assert "one unbroken passage" in v.issues[0] and "not found" not in v.issues[0]
    made_up = verify_message('you said "sponsored streams are great"', [source], 300, "Jane")
    assert "not found in their record" in made_up.issues[0]


@pytest.mark.parametrize(
    "quote",
    [
        "we should raise the minimum wage ... housing supply has to grow",  # two sentences spliced
        "housing supply has to grow ... we should raise the minimum wage",  # and out of order
        "we should raise ... minimum wage",  # one word dropped
    ],
)
def test_an_ellipsis_cannot_join_separate_parts(quote):
    source = ev("We should raise the minimum wage. Also, housing supply has to grow fast.")
    assert not verify_message(f'you said "{quote}"', [source], 300).ok


@pytest.mark.parametrize(
    "quote",
    [
        "we should raise the minimum wage…",  # trails off
        "... housing supply has to grow fast",  # picks up mid-way
        "…we should raise the minimum wage...",
        "the minimum wage. Also, housing supply",  # runs across the sentence break, nothing skipped
    ],
)
def test_an_ellipsis_around_an_unbroken_passage_is_fine(quote):
    source = ev("We should raise the minimum wage. Also, housing supply has to grow fast.")
    v = verify_message(f'you said "{quote}"', [source], 300)
    assert v.ok, v.issues


def test_an_ellipsis_they_wrote_themselves_is_just_punctuation():
    source = ev("Well... I think rent control works, actually.")
    v = verify_message('you said "well... I think rent control works"', [source], 300)
    assert v.ok and v.verified_quotes[0]["quote"] == "Well... I think rent control works"


# ── quotes: every kind of quote mark ─────────────────────────────

MARKS = [
    ('"', '"'),
    ("“", "”"),
    ("'", "'"),
    ("‘", "’"),
    ("«", "»"),
    ("„", "“"),
    ("‚", "‘"),
    ("’", "’"),
    ("＂", "＂"),
]


@pytest.mark.parametrize("opening,closing", MARKS)
def test_made_up_quotes_are_caught_in_any_quote_marks(opening, closing):
    """Regression: only double quote marks were looked at, so 'single-quoted' inventions passed."""
    source = ev("Rent control reduces housing supply.")
    v = verify_message(f"you said {opening}rent control is always evil{closing} lol", [source], 300, "Jane")
    assert not v.ok and "not found in their record" in v.issues[0]


@pytest.mark.parametrize("opening,closing", MARKS)
def test_real_quotes_pass_in_any_quote_marks(opening, closing):
    source = ev("Rent control reduces housing supply. Build more.")
    v = verify_message(f"you said {opening}rent control reduces housing supply{closing} lol", [source], 300)
    assert v.ok, v.issues
    assert [q["quote"] for q in v.verified_quotes] == ["Rent control reduces housing supply"]


@pytest.mark.parametrize(
    "message",
    [
        "don't say it's fine, that's not what Jane's record says",
        "the streamers' rights aren't the point, y'all",
        "back in '21 that wasn’t their view",
        "rock 'n' roll isn’t dead",
    ],
)
def test_apostrophes_are_not_quote_marks(message):
    v = verify_message(message, [ev("Rent control reduces housing supply.")], 300)
    assert v.ok and not v.verified_quotes, v.issues
    assert quoted_spans(message) == []


def test_single_quoted_quotes_may_contain_apostrophes():
    source = ev("Honestly I don't think it's fine for anyone.")
    v = verify_message("you said 'I don't think it's fine' and now this?", [source], 300)
    assert v.ok and v.verified_quotes[0]["quote"] == "I don't think it's fine"
    assert not verify_message("you said 'I do think it's fine' and now this?", [source], 300).ok


def test_a_possessive_inside_single_quotes_does_not_end_the_check_early():
    """The apostrophe in "streamers'" looks like a closing mark; the whole quote still has to match."""
    source = ev("I support the streamers' right to organise.")
    exact = verify_message("you said 'I support the streamers' right to organise'", [source], 300)
    assert exact.ok and exact.verified_quotes[0]["quote"] == "I support the streamers' right to organise"
    changed = verify_message("you said 'I support the streamers' right to strike'", [source], 300)
    assert not changed.ok  # "I support the streamers" alone would have matched


def test_unclear_single_quotes_fail_safe_and_say_how_to_fix_it():
    source = ev("Rent control reduces housing supply.")
    unclear = "you said 'rent control reduces housing supply' but the streamers' fans disagree"
    v = verify_message(unclear, [source], 300)
    assert not v.ok and "double quote marks" in v.issues[0]
    assert verify_message(unclear.replace("'rent", '"rent').replace("supply'", 'supply"'), [source], 300).ok
    # An apostrophe pair that only looks like a quote is rejected too, rather than passed unchecked.
    looks_quoted = verify_message("back in '21 that wasn't the streamers' view", [source], 300)
    assert not looks_quoted.ok and "double quote marks" in looks_quoted.issues[0]


def test_quote_marks_inside_a_quote_belong_to_it():
    source = ev("I don't think 'free markets' work for housing.")
    v = verify_message("you said \"I don't think 'free markets' work\" yesterday", [source], 300)
    assert v.ok and len(v.verified_quotes) == 1, v.issues
    nested = verify_message("you said 'I don't think \"free markets\" work for housing'", [source], 300)
    assert nested.ok and len(nested.verified_quotes) == 1, nested.issues


def test_short_quoted_words_do_not_shift_the_pairing():
    source = ev("Rent control reduces housing supply.")
    v = verify_message('"lol" you said "rent control reduces housing supply" though', [source], 300)
    assert v.ok and len(v.verified_quotes) == 1, v.issues
    assert [s.text for s in quoted_spans('"no" and then "rent control is evil" and \'ok\'')] == [
        "rent control is evil"
    ]


def test_two_quotes_are_checked_separately():
    source = ev("Rent control reduces housing supply. Build more homes instead.")
    both = "you said 'rent control reduces housing supply' and 'build more homes instead'"
    assert len(verify_message(both, [source], 300).verified_quotes) == 2
    one_wrong = "you said 'rent control reduces housing supply' and 'build fewer homes instead'"
    v = verify_message(one_wrong, [source], 300)
    assert not v.ok and len(v.verified_quotes) == 1 and "build fewer homes" in v.issues[0]


def test_third_party_words_in_single_quotes_are_still_third_party():
    v = verify_message(
        "you said 'she has opposed rent control'",
        [ev("She has opposed rent control.", authorship=ABOUT)],
        300,
        "Jane",
    )
    assert not v.ok and "third-party" in v.issues[0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("who pays for it?", False),
        ("that's not what Jane's fans' polls say", False),
        ('you said "no"', True),
        ("you said 'no'", True),
        ("you said ‘never’ though", True),
        ("«jamais», you said", True),
        ('an unclosed " mark', True),
    ],
)
def test_has_quote_marks(text, expected):
    assert has_quote_marks(text) is expected
