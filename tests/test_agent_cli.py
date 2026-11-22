"""The two user-facing commands in agent mode: option handling, and full runs as real processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import SR, voice_vector
from kickvoice.agent.runner import RunOutcome
from kickvoice.agent.trace import RunDir, read_trace
from kickvoice.audio.ffmpeg import write_wav
from kickvoice.cli import cli
from kickvoice.research.store import Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore

TESTS = Path(__file__).parent
URL = "https://kick.com/somechannel"


@pytest.fixture
def invoke(config, tmp_path):
    cfg = tmp_path / "config.json"
    config.save(cfg)

    def _invoke(*args, expect=0):
        result = CliRunner().invoke(cli, ["--config", str(cfg), *args], obj={})
        assert result.exit_code == expect, result.output + repr(result.exception)
        return result

    return _invoke


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Replace the runner: record what the CLI asks for."""
    import kickvoice.agent.runner as runner

    calls = []

    def fake_run_agent(command, config, options, policy="llm", **kw):
        calls.append({"command": command, "options": options, "policy": policy, "config": config})
        return RunOutcome(RunDir.create(tmp_path / "runs", command, options["source"]), {})

    monkeypatch.setattr(runner, "run_agent", fake_run_agent)
    return calls


def test_monitor_spawns_agent_with_options(invoke, captured):
    out = invoke(
        "monitor",
        "--stream",
        URL,
        "--opponent",
        "Jane",
        "--me",
        "Me",
        "--duration",
        "60",
        "--policy",
        "rules",
        "--model",
        "m-1",
        "--max-tokens",
        "5000",
    ).output
    call = captured[0]
    assert call["command"] == "monitor" and call["policy"] == "rules"
    assert call["options"] == {"source": URL, "opponent": "Jane", "me": "Me", "duration_s": 60.0}
    assert call["config"].agent.model == "m-1" and call["config"].agent.reply_model == "m-1"
    assert call["config"].agent.max_run_tokens == 5000
    assert "Report:" in out and "Trace:" in out


def test_extract_profiles_spawns_agent(invoke, captured, tmp_path):
    invoke("extract-profiles", "--stream", URL, "--min-speech", "12")
    assert captured[0]["options"] == {"source": URL, "duration_s": 300.0, "min_speech_s": 12.0}
    audio = tmp_path / "vod.wav"
    write_wav(audio, b"\x00\x00" * SR, SR)
    invoke("extract-profiles", "--file", str(audio))
    assert captured[1]["options"]["duration_s"] is None  # whole file
    assert captured[1]["policy"] == "llm"


def test_missing_model_key_explains_alternatives(invoke, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = invoke("monitor", "--stream", URL, expect=1)
    assert "ANTHROPIC_API_KEY" in result.output and "--policy rules" in result.output


def test_failed_run_exits_nonzero(invoke, monkeypatch, tmp_path):
    import kickvoice.agent.runner as runner

    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *a, **k: RunOutcome(
            RunDir.create(tmp_path / "runs", "monitor", URL), {}, ok=False, error="stream offline"
        ),
    )
    result = invoke("monitor", "--stream", URL, expect=1)
    assert "stopped early: stream offline" in result.output


def test_agent_extra_missing(invoke, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_runner(name, *args, **kwargs):
        if name == "kickvoice.agent.runner":
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_runner)
    result = invoke("monitor", "--stream", URL, expect=1)
    assert "kickvoice[agent]" in result.output


# ── the real thing: separate processes, real MCP over stdio ──────

SITECUSTOMIZE = """
import sys
sys.path.insert(0, {tests!r})
import kickvoice.agent.live as live
from conftest import conversation, fake_pipeline_factory
live.default_pipeline_factory = fake_pipeline_factory(conversation({turns!r}), delay=0.05)
"""


def run_cli(config, tmp_path, turns, *args):
    """`python -m kickvoice …` as a user would run it (its MCP server is another process)."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(SITECUSTOMIZE.format(tests=str(TESTS), turns=turns))
    cfg = tmp_path / "config.json"
    config.save(cfg)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(hook), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run(
        [sys.executable, "-m", "kickvoice", "--config", str(cfg), *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    runs = sorted((Path(config.data_dir) / "runs").iterdir())
    return proc, runs[-1]


def fast(config):
    config.debate.chunk_duration_s = 5.0
    config.debate.min_point_words = 5
    config.agent.poll_wait_s = 0.5
    config.agent.refine_interval_s = 0.5
    config.agent.min_enroll_speech_s = 5.0
    return config


@pytest.mark.slow
def test_monitor_command_end_to_end(config, tmp_path):
    fast(config)
    ProfileStore(config).add_profile("Jane", voice_vector(220), speech_seconds=300)
    ProfileStore(config).add_profile("Me", voice_vector(550), speech_seconds=300)
    with ResearchStore(config.research_db) as store:
        p = store.add_person("Jane Pundit", voice_profile="Jane")
        doc = Document(
            p.id,
            "x",
            "post",
            "1",
            "w220 w220 w220 w220 is what I said",
            "self",
            url="https://x.com/jane/status/1",
            published_at="2021-05-05",
        )
        store.upsert_document(doc, [Chunk(doc.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        store.add_claims(
            [
                Claim(
                    p.id,
                    doc.id,
                    chunk.id,
                    "w220",
                    "w220 stance",
                    doc.text,
                    published_at="2021-05-05",
                    url=doc.url,
                )
            ]
        )

    proc, run_dir = run_cli(
        config,
        tmp_path,
        [(220, 5), (550, 4), (220, 5)],
        "monitor",
        "--stream",
        URL,
        "--opponent",
        "Jane Pundit",
        "--me",
        "Me",
        "--policy",
        "rules",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Debate copilot (rules)" in proc.stdout and "Report:" in proc.stdout
    report = json.loads((run_dir / "report.json").read_text())
    assert report["ok"] and len(report["points"]) == 2
    assert report["replies"] and report["replies"][0]["sources"][0]["date"] == "2021-05-05"
    assert (run_dir / "mcp_server.log").exists() and (run_dir / "config.json").exists()
    trace = read_trace(run_dir / "trace.jsonl")
    assert {"run_start", "tools_loaded", "live_event", "replies_proposed", "run_end"} <= {
        e["kind"] for e in trace
    }
    assert "start_live_session" in {e["tool"] for e in trace if e["kind"] == "tool_call"}


@pytest.mark.slow
def test_extract_profiles_command_end_to_end(config, tmp_path):
    fast(config)
    proc, run_dir = run_cli(
        config,
        tmp_path,
        [(220, 6), (550, 5)] * 3,
        "extract-profiles",
        "--stream",
        URL,
        "--policy",
        "rules",
        "--min-speech",
        "8",
        "--duration",
        "60",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((run_dir / "report.json").read_text())
    # not a terminal → nobody to approve → nothing created, everything pending
    assert [d["decision"] for d in report["decisions"]] == ["pending", "pending"]
    assert len(list((run_dir / "clips").glob("*.wav"))) == 2
    assert "Pending approval: voice 1, voice 2" in (run_dir / "report.md").read_text()
