"""Agent behaviour at the edges: Ctrl+C, time limits, stream errors, slow models, fact-check paths."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest

from conftest import ScriptedToolLLM, conversation, fake_pipeline_factory, voice_vector
from kickvoice.agent.runner import run_agent
from kickvoice.agent.service import AgentService
from kickvoice.agent.terminal import ScriptedTerminal, Terminal
from kickvoice.agent.trace import read_trace
from kickvoice.research.store import Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore
from test_agent_cli import SITECUSTOMIZE, TESTS, fast

URL = "https://kick.com/somechannel"


@pytest.fixture
def cfg(config):
    config.debate.chunk_duration_s = 5.0
    config.debate.min_point_words = 5
    config.agent.poll_wait_s = 0.3
    config.agent.refine_interval_s = 0.5
    config.agent.reply_timeout_s = 5.0
    return config


def go(
    cfg,
    command,
    options,
    turns,
    delay=0.05,
    policy="rules",
    llm=None,
    terminal=None,
    clipboard=None,
    factory=None,
):
    service = AgentService(
        cfg,
        pipeline_factory=factory or fake_pipeline_factory(conversation(turns), delay),
        clipboard=clipboard or (lambda t: (True, "fake")),
    )
    lines = []
    outcome = run_agent(
        command,
        cfg,
        {"source": URL, **options},
        policy=policy,
        llm=llm,
        terminal=terminal or ScriptedTerminal([]),
        service=service,
        echo=lines.append,
    )
    service.close()
    return outcome, lines


def jane(cfg, claims=(("w220 w220 w220 w220 is my honest view", "self", True),)):
    profiles = ProfileStore(cfg)
    profiles.add_profile("Jane", voice_vector(220), speech_seconds=300)
    profiles.add_profile("Me", voice_vector(550), speech_seconds=300)
    with ResearchStore(cfg.research_db) as store:
        p = store.add_person("Jane Pundit", voice_profile="Jane")
        for i, (text, authorship, _sincere) in enumerate(claims):
            doc = Document(p.id, "x", "post", str(i), text, authorship, published_at="2022-02-02")
            store.upsert_document(doc, [Chunk(text)])
        for chunk, doc in store.chunks_to_analyze(p.id):
            store.add_claims(
                [
                    Claim(
                        p.id,
                        doc.id,
                        chunk.id,
                        "w220",
                        "claim",
                        doc.text,
                        authorship=doc.authorship,
                        sincere=dict((t, s) for t, _, s in claims)[doc.text],
                        published_at="2022-02-02",
                    )
                ]
            )


# ── time limit, stream errors ────────────────────────────────────


def test_duration_limit_stops_listening(cfg):
    """monitor enforces --duration itself (the live session has no limit)."""
    outcome, lines = go(cfg, "monitor", {"duration_s": 1.0}, [(220, 5)] * 20, delay=0.5)
    assert outcome.ok and outcome.report["session_status"] == "stopped"
    assert any("reached" in e.get("message", "") for e in read_trace(outcome.run_dir.trace))


def test_stream_error_is_reported_and_run_still_finishes(cfg):
    def broken_factory(stream_url, file_path, config, on_utterance, audio_path=None):
        raise RuntimeError("streamlink failed: error: No playable streams found")

    outcome, lines = go(cfg, "monitor", {"opponent": None}, [], factory=broken_factory)
    assert outcome.ok and outcome.report["session_status"] == "error"
    assert any("No playable streams" in line for line in lines)


# ── monitor: slow model → stale points skipped ───────────────────


def test_slow_replies_skip_stale_points(cfg):
    jane(cfg)
    cfg.debate.max_queued_points = 1

    def slow_script(system, messages, tools):
        if "start_live_session" in tools:
            if not any(m["role"] == "tool" for m in messages):
                return "", [("start_live_session", {"source": URL, "opponent": "Jane Pundit", "me": "Me"})]
            return "ok", []
        time.sleep(1.5)  # a slow model
        return "nothing useful", []

    outcome, _ = go(
        cfg,
        "monitor",
        {},
        [(220, 5), (550, 2)] * 6,
        delay=0.05,
        policy="llm",
        llm=ScriptedToolLLM(slow_script),
    )
    assert outcome.report["points_skipped"] >= 1
    assert any(e["kind"] == "point_skipped" for e in read_trace(outcome.run_dir.trace))


# ── monitor: the fact-check and rules paths ──────────────────────


def test_rules_receipts_skip_unusable_evidence(cfg):
    jane(
        cfg,
        claims=[
            ("w220 w220 w220 w220 w220 is a joke", "self", False),  # insincere
            ("w220 w220 w220 w220 reported", "about", True),  # not first-hand
            ("w220 " * 70, "self", True),
        ],
    )  # too long for chat
    outcome, lines = go(cfg, "monitor", {"opponent": "Jane Pundit", "me": "Me"}, [(220, 5), (550, 3)])
    # the joke is excluded as a claim AND as a raw excerpt; reported words aren't theirs
    assert outcome.report["replies"] == []
    assert any("nothing relevant in their record" in line for line in lines)


def propose(cfg, replies, opponent_has_record=True):
    """Run one reply episode whose model proposes `replies`; return the tool result it saw."""
    if opponent_has_record:
        jane(cfg)
    else:
        ProfileStore(cfg).add_profile("Jane", voice_vector(220))
        ProfileStore(cfg).add_profile("Me", voice_vector(550))
    seen = []

    def script(system, messages, tools):
        if "start_live_session" in tools:
            if not any(m["role"] == "tool" for m in messages):
                opp = "Jane Pundit" if opponent_has_record else "Jane"
                return "", [("start_live_session", {"source": URL, "opponent": opp, "me": "Me"})]
            return "ok", []
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            point = messages[0]["text"].split("Their point [")[1].split("]")[0]
            return "", [("propose_replies", {"point_id": point, "replies": replies})]
        seen.append(
            json.loads(tool_msgs[-1]["content"])
            if not tool_msgs[-1]["is_error"]
            else tool_msgs[-1]["content"]
        )
        return "done", []

    outcome, lines = go(cfg, "monitor", {}, [(220, 5), (550, 3)], policy="llm", llm=ScriptedToolLLM(script))
    return seen[0], outcome, lines


def test_propose_replies_rejects_bad_input(cfg):
    result, _, _ = propose(cfg, [])
    assert "non-empty list" in result


def test_empty_reply_is_rejected(cfg):
    result, outcome, _ = propose(cfg, [{"text": "**   **"}])
    assert result["rejected"][0]["issues"] == ["empty reply"]


def test_without_a_record_quotes_are_rejected_and_plain_replies_flagged(cfg):
    result, outcome, lines = propose(
        cfg,
        [
            {"text": 'you said "w220 forever"'},
            {"text": "you said 'w220 forever'"},
            {"text": "who pays for it?"},
        ],
        opponent_has_record=False,
    )
    assert [r["text"] for r in result["rejected"]] == ['you said "w220 forever"', "you said 'w220 forever'"]
    assert all("can't be checked" in r["issues"][0] for r in result["rejected"])
    assert outcome.report["replies"][0]["warnings"] == ["unchecked: no research record"]


def test_copy_failure_shows_the_text(cfg):
    jane(cfg)
    terminal = ScriptedTerminal([])

    def echo_hook(line):
        if "[1] " in line:
            terminal.feed("1")

    service = AgentService(
        cfg,
        pipeline_factory=fake_pipeline_factory(conversation([(220, 5), (550, 3)]), 0.05),
        clipboard=lambda t: (False, "no clipboard tool found"),
    )
    lines = []

    def echo(line):
        lines.append(line)
        echo_hook(line)

    outcome = run_agent(
        "monitor",
        cfg,
        {"source": URL, "opponent": "Jane Pundit", "me": "Me"},
        policy="rules",
        terminal=terminal,
        service=service,
        echo=echo,
    )
    service.close()
    assert outcome.report["copied"] == []
    assert any("Couldn't copy (no clipboard tool found)" in line for line in lines)


def test_rules_identify_without_candidates_or_answer(cfg):
    ProfileStore(cfg).add_profile("Me", voice_vector(550), speech_seconds=300)
    cfg.agent.identify_after_s = 3
    outcome, _ = go(
        cfg, "monitor", {"me": "Me"}, [(220, 6), (550, 3)] * 2, delay=0.2, terminal=ScriptedTerminal([""])
    )  # user skips the question
    assert outcome.ok and ProfileStore(cfg).find_profile("Jane") is None
    assert "No arguments from the opponent were detected" in outcome.run_dir.report_md.read_text()


# ── terminal ─────────────────────────────────────────────────────


def test_terminal_reads_real_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("2\nq\n"))

    async def main():
        t = Terminal(interactive=True, echo=lambda _m: None)
        await t.read_stdin()
        got = []
        async for command in t.commands():
            got.append(command)
            if len(got) == 2:
                break
        return got

    assert anyio.run(main) == ["2", "q"]


# ── Ctrl+C on a real process ─────────────────────────────────────


@pytest.mark.slow
def test_ctrl_c_stops_cleanly_and_still_reports(config, tmp_path):
    fast(config)
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        SITECUSTOMIZE.format(tests=str(TESTS), turns=[(220, 5)] * 40).replace("delay=0.05", "delay=0.5")
    )
    cfg_path = tmp_path / "config.json"
    config.save(cfg_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(hook), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kickvoice",
            "--config",
            str(cfg_path),
            "monitor",
            "--stream",
            URL,
            "--policy",
            "rules",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    run_root = Path(config.data_dir) / "runs"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # wait until it's actually listening
        traces = list(run_root.glob("*/trace.jsonl")) if run_root.exists() else []
        if traces and '"live_event"' in traces[0].read_text():
            break
        time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    assert "Stopped by you" in out and "Report:" in out
    report = json.loads(next(run_root.glob("*/report.json")).read_text())
    assert report["ok"] and report["session_status"] == "stopped"


def test_reply_to_the_last_point_survives_the_stream_ending(cfg):
    """Regression: the final argument's reply was cancelled mid-way when the stream ended."""
    jane(cfg)

    def slow_then_propose(system, messages, tools):
        if "start_live_session" in tools:
            if not any(m["role"] == "tool" for m in messages):
                return "", [("start_live_session", {"source": URL, "opponent": "Jane Pundit", "me": "Me"})]
            return "ok", []
        if not any(m["role"] == "tool" for m in messages):
            time.sleep(1.0)  # the stream has long finished by now
            point = messages[0]["text"].split("Their point [")[1].split("]")[0]
            return "", [("propose_replies", {"point_id": point, "replies": [{"text": "who pays for it?"}]})]
        return "done", []

    outcome, _ = go(
        cfg, "monitor", {}, [(220, 5)], delay=0.0, policy="llm", llm=ScriptedToolLLM(slow_then_propose)
    )
    assert [r["text"] for r in outcome.report["replies"]] == ["who pays for it?"]


def test_real_terminal_answers_without_a_reader_task(monkeypatch):
    """Regression: extract-profiles (no stdin reader task) hung forever on its first question."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("Jane\n"))

    async def main():
        t = Terminal(interactive=True, echo=lambda _m: None)
        first = await t.ask("Who is voice 1?")
        second = await t.ask("Who is voice 2?")  # EOF → no answer
        return first, second

    assert anyio.run(main) == ("Jane", None)


def test_extract_profiles_with_the_real_terminal(cfg, monkeypatch):
    """The whole extract-profiles run answering on (simulated) stdin, as a user at a keyboard."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("Host Person\n\n"))
    outcome, _ = go(
        cfg,
        "extract-profiles",
        {"min_speech_s": 8},
        [(220, 6), (550, 5)] * 3,
        terminal=Terminal(interactive=True, echo=lambda _m: None),
    )
    assert [d["decision"] for d in outcome.report["decisions"]] == ["created", "skipped"]
    assert ProfileStore(cfg).find_profile("Host Person") is not None


def test_command_typed_as_the_stream_ends_is_still_handled(cfg):
    """Regression: a line handed straight to the waiting command task was invisible to shutdown."""
    jane(cfg)
    copied = []
    service = AgentService(
        cfg,
        pipeline_factory=fake_pipeline_factory(conversation([(220, 5)]), 0.0),
        clipboard=lambda t: (copied.append(t) or True, "fake"),
    )
    for _ in range(5):  # repeat: this used to fail intermittently
        copied.clear()
        terminal, typed = ScriptedTerminal([]), []

        def echo(line, terminal=terminal, typed=typed):
            if "   [1] " in line and not typed:  # typed the moment the reply appears
                typed.append(1)
                terminal.feed("1")

        outcome = run_agent(
            "monitor",
            cfg,
            {"source": URL, "opponent": "Jane Pundit"},
            policy="rules",
            terminal=terminal,
            service=service,
            echo=echo,
        )
        assert outcome.report["copied"] == [1] and len(copied) == 1
    service.close()
