"""CLI wiring: argument parsing, dispatch, error handling and exit codes."""
import json

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import SR, FakeEncoder, conversation, tone
from kickvoice.audio.ffmpeg import write_wav
from kickvoice.audio.vad import EnergyVAD
from kickvoice.cli import cli
from kickvoice.speakers.profiles import ProfileStore


@pytest.fixture
def run(config, tmp_path):
    """Invoke the CLI against an isolated config file."""
    cfg_path = tmp_path / "config.json"
    config.save(cfg_path)
    runner = CliRunner()

    def _run(*args, input=None, expect_ok=True):
        result = runner.invoke(cli, ["--config", str(cfg_path), *args], input=input, obj={})
        if expect_ok and result.exit_code != 0:
            raise AssertionError(
                f"exit {result.exit_code}: {result.output}\n{result.exception!r}")
        return result
    return _run


def unit(seed):
    v = np.random.default_rng(seed).normal(size=192)
    return v / np.linalg.norm(v)


def test_help_lists_commands(run):
    out = run("--help").output
    for command in ("enroll", "monitor", "extract-profiles", "review", "profiles"):
        assert command in out


def test_version(run):
    assert "kickvoice" in run("--version").output


def test_profiles_empty_then_populated(run, config):
    assert "No enrolled speakers" in run("profiles", "list").output
    ProfileStore(config).add_profile("xQc", unit(1), n_segments=7, speech_seconds=125,
                                     source_urls=["https://youtu.be/a"])
    out = run("profiles", "list").output
    assert "xQc" in out and "2m 05s" in out


def test_profiles_info_and_missing(run, config):
    ProfileStore(config).add_profile("xQc", unit(1), notes="hi", speech_seconds=60)
    out = run("profiles", "info", "xqc").output      # case-insensitive lookup
    assert "Speaker: xQc" in out and "hi" in out
    missing = run("profiles", "info", "nobody", expect_ok=False)
    assert missing.exit_code == 1 and "not found" in missing.output


def test_profiles_remove_confirmation(run, config):
    ProfileStore(config).add_profile("xQc", unit(1))
    assert "Cancelled" in run("profiles", "remove", "xQc", input="n\n").output
    assert ProfileStore(config).count == 1
    assert "Removed" in run("profiles", "remove", "xQc", "--yes").output
    assert ProfileStore(config).count == 0


def test_profiles_export_import_roundtrip(run, config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("Amy", unit(1), speech_seconds=100)
    store.add_profile("Bob", unit(2), speech_seconds=50)
    pack = tmp_path / "pack.json"

    assert "Exported 2" in run("profiles", "export", "--out", str(pack)).output
    assert json.loads(pack.read_text())["format"] == "kickvoice-profile-pack"

    # selected profiles, comma-separated
    one = tmp_path / "one.json"
    run("profiles", "export", "--out", str(one), "--name", "Amy,Bob")
    assert len(json.loads(one.read_text())["profiles"]) == 2

    config.profiles_dir = tmp_path / "fresh"
    config.save(tmp_path / "config.json")
    assert "Added: Amy, Bob" in run("profiles", "import", str(pack)).output
    assert "Skipped" in run("profiles", "import", str(pack)).output
    assert "Merged: Amy, Bob" in run(
        "profiles", "import", str(pack), "--on-conflict", "merge").output


def test_profiles_import_rejects_foreign_pack(run, config, tmp_path):
    pack = tmp_path / "p.json"
    pack.write_text(json.dumps({
        "format": "kickvoice-profile-pack", "version": 1,
        "embedding_model": "some/other-model", "embedding_dim": 192, "profiles": [],
    }))
    result = run("profiles", "import", str(pack), expect_ok=False)
    assert result.exit_code == 1 and "different models" in result.output


def test_enroll_requires_a_source(run):
    result = run("enroll", "Nobody", expect_ok=False)
    assert result.exit_code == 2 and "at least one --file or --youtube" in result.output


def test_enroll_from_file(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.enroll as enroll_mod

    clip = tmp_path / "clip.wav"
    write_wav(clip, conversation([(220, 5), (220, 4)]), SR)
    monkeypatch.setattr(enroll_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(enroll_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))

    out = run("enroll", "Streamer", "--file", str(clip)).output
    assert "Enrolled 'Streamer'" in out and "2 segments" in out
    assert ProfileStore(config).get_profile("Streamer").speech_seconds > 8


def test_enroll_reports_no_speech(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.enroll as enroll_mod

    quiet = tmp_path / "quiet.wav"
    write_wav(quiet, b"\x00\x00" * SR * 3, SR)
    monkeypatch.setattr(enroll_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(enroll_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))

    result = run("enroll", "Streamer", "--file", str(quiet), expect_ok=False)
    assert result.exit_code == 1 and "No speech detected" in result.output


@pytest.mark.parametrize("command", ["monitor", "extract-profiles"])
@pytest.mark.parametrize("args", [[], ["--stream", "u", "--file", "f"]])
def test_stream_or_file_required(run, command, args, tmp_path):
    if "--file" in args:
        (tmp_path / "f").write_bytes(b"x")
        args = ["--stream", "u", "--file", str(tmp_path / "f")]
    result = run(command, *args, expect_ok=False)
    assert result.exit_code == 2 and "exactly one" in result.output


def test_extract_profiles_runs_pipeline_and_defers_review(run, config, tmp_path, monkeypatch):
    """--no-review stops before the interactive step and points at `review`."""
    import kickvoice.pipelines.extract as extract_mod

    audio = tmp_path / "stream.wav"
    write_wav(audio, conversation([(220, 6), (550, 5), (220, 6), (550, 5)]), SR)
    monkeypatch.setattr(extract_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(extract_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))

    out = run("extract-profiles", "--file", str(audio),
              "--min-speech", "8", "--no-review").output
    assert "Found 2 speaker(s)" in out and "Review later with" in out

    sessions = run("review", "--list").output
    assert "2" in sessions and "stream" in sessions


def test_review_with_nothing_pending(run):
    assert "Nothing to review" in run("review").output


def test_review_unknown_session(run):
    result = run("review", "--session", "does-not-exist", expect_ok=False)
    assert result.exit_code == 1 and "not found" in result.output


def test_monitor_without_profiles_explains(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.monitor as monitor_mod

    audio = tmp_path / "a.wav"
    write_wav(audio, tone(220, 3), SR)
    monkeypatch.setattr(monitor_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(monitor_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))
    out = run("monitor", "--file", str(audio)).output
    assert "No speaker profiles enrolled" in out and "extract-profiles" in out


def test_monitor_detects_enrolled_speaker(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.monitor as monitor_mod
    from conftest import voice_vector

    ProfileStore(config).add_profile("Host", voice_vector(220), speech_seconds=300)
    audio = tmp_path / "a.wav"
    write_wav(audio, conversation([(220, 6), (550, 6)]), SR)
    monkeypatch.setattr(monitor_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(monitor_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))

    out = run("monitor", "--file", str(audio)).output
    assert "Host detected" in out and "Unknown speaker" in out
    assert "Match rate" in out


# ── backwards compatibility with the pre-Click flags ─────────────

def test_old_style_name_flags_still_work(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.enroll as enroll_mod

    clip = tmp_path / "clip.wav"
    write_wav(clip, conversation([(220, 5), (220, 4)]), SR)
    monkeypatch.setattr(enroll_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(enroll_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))

    # old: enroll --name X --file Y
    assert "Enrolled 'Streamer'" in run(
        "enroll", "--name", "Streamer", "--file", str(clip)).output
    assert "Speaker: Streamer" in run("profiles", "info", "--name", "Streamer").output
    assert "Removed" in run("profiles", "remove", "--name", "Streamer", "--yes").output


def test_old_style_youtube_comma_list(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.enroll as enroll_mod
    from test_enroll import FakeDownloader

    downloader = FakeDownloader(tone(220, 6))
    monkeypatch.setattr(enroll_mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(enroll_mod, "get_vad", lambda **kw: EnergyVAD(config.audio))
    monkeypatch.setattr(enroll_mod, "download_audio", downloader)

    run("enroll", "Streamer", "--youtube", "https://youtu.be/A,https://youtu.be/B")
    assert [u for u, _ in downloader.calls] == ["https://youtu.be/A", "https://youtu.be/B"]


def test_old_style_pack_file_flag(run, config, tmp_path):
    ProfileStore(config).add_profile("Amy", unit(1))
    pack = tmp_path / "p.json"
    run("profiles", "export", "--out", str(pack))
    config.profiles_dir = tmp_path / "fresh"
    config.save(tmp_path / "config.json")
    assert "Added: Amy" in run("profiles", "import", "--file", str(pack)).output


def test_name_given_twice_is_rejected(run):
    result = run("enroll", "A", "--name", "B", "--file", "x", expect_ok=False)
    assert result.exit_code == 2


# ── transcribe / transcripts ─────────────────────────────────────

@pytest.fixture
def fake_transcription(config, monkeypatch):
    """Swap the models inside the transcription pipeline for fakes."""
    import kickvoice.pipelines.transcribe as mod
    from conftest import FakeRecognizer

    monkeypatch.setattr(mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(mod, "get_vad", lambda **kw: EnergyVAD(config.audio))
    monkeypatch.setattr(mod, "FasterWhisperRecognizer", lambda config=None: FakeRecognizer())


@pytest.fixture
def transcribed(run, config, tmp_path, fake_transcription):
    from conftest import voice_vector

    ProfileStore(config).add_profile("Host", voice_vector(220), speech_seconds=300)
    audio = tmp_path / "vod.wav"
    write_wav(audio, conversation([(220, 5), (550, 4), (220, 5)]), SR)
    result = run("transcribe", "--file", str(audio))
    return result.output


def test_transcribe_prints_attributed_lines(transcribed):
    # live lines are provisional: unmatched voices print as "(unknown)"
    assert "Host: w220" in transcribed and "(unknown): w550" in transcribed
    assert "2 distinct voice(s), 1 matched" in transcribed
    assert "confident: 2" in transcribed and "unknown: 1" in transcribed
    assert "transcripts show" in transcribed


def test_transcripts_list_show_speakers(run, transcribed):
    listing = run("transcripts", "list").output
    assert "vod" in listing and "Host" in listing
    session = [l.split()[0] for l in listing.splitlines() if "vod" in l][0]

    shown = run("transcripts", "show", session).output
    # after the whole-session check, the unmatched voice has a group number
    assert shown.count("w220") > 2 and "(voice 2): w550" in shown
    only_host = run("transcripts", "show", session, "--speaker", "host").output
    assert "(voice" not in only_host
    confident = run("transcripts", "show", session, "--min-attribution", "confident").output
    assert "w550" not in confident

    speakers = run("transcripts", "speakers").output
    assert "Host" in speakers and "(unknown)" in speakers      # summary pools unknowns


def test_transcripts_search(run, transcribed):
    out = run("transcripts", "search", "w220").output
    assert "Host (confident)" in out and "2 match(es)" in out
    assert "No matches" in run("transcripts", "search", "w550").output   # unknown hidden
    assert "w550" in run("transcripts", "search", "w550", "--min-attribution", "unknown").output


def test_transcripts_export_jsonl_and_txt(run, transcribed, tmp_path):
    jsonl = tmp_path / "out.jsonl"
    assert "Exported 3" in run("transcripts", "export", "--out", str(jsonl)).output
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    assert [r["speaker"] for r in rows] == ["Host", None, "Host"]
    assert {"said_at", "source", "attribution", "start_s", "text"} <= set(rows[0])

    txt = tmp_path / "host.txt"
    run("transcripts", "export", "--out", str(txt), "--speaker", "Host",
        "--min-attribution", "confident")
    assert txt.read_text().count("Host:") == 2


def test_transcripts_reattribute_after_enrolling(run, config, transcribed, tmp_path):
    from conftest import voice_vector

    ProfileStore(config).add_profile("Guest", voice_vector(550), speech_seconds=300)
    assert "1 line(s) changed" in run("transcripts", "reattribute").output
    out = tmp_path / "after.jsonl"
    run("transcripts", "export", "--out", str(out))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [(r["speaker"], r["attribution"]) for r in rows] == [
        ("Host", "confident"), ("Guest", "confident"), ("Host", "confident")]
    assert "1 line(s) changed" not in run("transcripts", "reattribute").output   # idempotent


def test_transcripts_reattribute_refuses_other_model(run, config, tmp_path, transcribed):
    config.embedding.model_source = "other/model"
    config.save(tmp_path / "config.json")
    result = run("transcripts", "reattribute", expect_ok=False)
    assert result.exit_code == 1 and "aren't comparable" in result.output


def test_transcripts_delete(run, transcribed):
    session = [l.split()[0] for l in run("transcripts", "list").output.splitlines()
               if "vod" in l][0]
    assert "Cancelled" in run("transcripts", "delete", session, input="n\n").output
    assert "Deleted" in run("transcripts", "delete", session, "-y").output
    assert "No transcripts yet" in run("transcripts", "list").output


def test_transcripts_unknown_session(run):
    result = run("transcripts", "show", "nope", expect_ok=False)
    assert result.exit_code == 1 and "not found" in result.output


def test_transcribe_options_reach_config(run, config, tmp_path, monkeypatch):
    import kickvoice.pipelines.transcribe as mod

    seen = {}

    class Capturing:
        def __init__(self, stream_url=None, file_path=None, config=None, on_utterance=None):
            seen.update(vars(config.transcribe), quiet=on_utterance is None)

        def run(self, max_duration_s=None):
            seen["duration"] = max_duration_s

    monkeypatch.setattr(mod, "TranscriptionPipeline", Capturing)
    audio = tmp_path / "a.wav"
    write_wav(audio, tone(220, 1), SR)
    run("transcribe", "--file", str(audio), "--model", "medium", "--language", "en",
        "--device", "cuda", "--drop-unknown", "--quiet", "--duration", "60")
    assert seen["asr_model"] == "medium" and seen["language"] == "en"
    assert seen["asr_device"] == "cuda" and seen["asr_compute_type"] == "float16"
    assert seen["keep_unattributed"] is False and seen["quiet"] and seen["duration"] == 60


def test_transcribe_without_faster_whisper_explains(run, config, tmp_path, monkeypatch):
    import builtins

    import kickvoice.pipelines.transcribe as mod

    monkeypatch.setattr(mod, "SpeakerEncoder", lambda config=None: FakeEncoder())
    monkeypatch.setattr(mod, "get_vad", lambda **kw: EnergyVAD(config.audio))
    real_import = builtins.__import__

    def no_whisper(name, *args, **kwargs):
        if name.startswith("faster_whisper"):
            raise ImportError("no faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_whisper)
    audio = tmp_path / "a.wav"
    write_wav(audio, conversation([(220, 3)]), SR)
    result = run("transcribe", "--file", str(audio), expect_ok=False)
    assert result.exit_code == 1 and "pip install" in result.output
