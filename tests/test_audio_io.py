"""Stream capture (streamlink + ffmpeg), downloads (yt-dlp) and ffmpeg helpers, via real subprocesses."""

import os
import stat
import sys
import textwrap

import pytest

from conftest import SR, conversation, tone
from kickvoice.audio import capture as capture_mod
from kickvoice.audio.capture import StreamCapture
from kickvoice.audio.download import download_audio
from kickvoice.audio.ffmpeg import (
    AudioError,
    decode_to_pcm,
    open_pcm_stream,
    pcm_to_tensor,
    require_tool,
    write_wav,
)


def fake_tool(bin_dir, name, body):
    """Create an executable `name` (a small Python script) on a private PATH dir."""
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def bin_dir(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")
    return d


@pytest.fixture
def stream_file(tmp_path):
    path = tmp_path / "stream.wav"
    write_wav(path, conversation([(220, 4), (550, 4)]), SR)
    return path


# ── streamlink + ffmpeg ──────────────────────────────────────────


def test_stream_capture_resolves_and_decodes(config, bin_dir, stream_file):
    fake_tool(
        bin_dir,
        "streamlink",
        f"""
        import sys
        assert sys.argv[1:] == ["--stream-url", "https://kick.com/chan", "best"], sys.argv
        print({str(stream_file)!r})
    """,
    )
    cap = StreamCapture("https://kick.com/chan", config)
    chunks = list(cap.capture_chunks(chunk_duration_s=2, chunk_stride_s=2))
    total = sum(len(c) for c in chunks) / (SR * 2)
    assert total == pytest.approx(9.8, abs=0.1) and all(len(c) == SR * 4 for c in chunks[:-1])
    assert cap._process is None  # cleaned up after the stream ended


@pytest.mark.parametrize(
    "stdout,stderr,expected",
    [
        ("error: No playable streams found on this URL", "", "No playable streams"),  # streamlink style
        ("", "boom", "boom"),
        ("", "", "exit code 1"),
    ],
)
def test_stream_capture_reports_streamlink_errors(config, bin_dir, stdout, stderr, expected):
    fake_tool(
        bin_dir,
        "streamlink",
        f"""
        import sys
        print({stdout!r}); print({stderr!r}, file=sys.stderr); sys.exit(1)
    """,
    )
    with pytest.raises(AudioError, match=expected):
        list(StreamCapture("https://kick.com/offline", config).capture_chunks())


def test_stream_capture_stop_terminates_ffmpeg(config, bin_dir, tmp_path):
    long = tmp_path / "long.wav"
    write_wav(long, tone(220, 60), SR)
    fake_tool(bin_dir, "streamlink", f"print({str(long)!r})")
    cap = StreamCapture("https://kick.com/chan", config)
    gen = cap.capture_chunks(chunk_duration_s=1, chunk_stride_s=1)
    next(gen)
    process = cap._process
    cap.stop()
    assert process.poll() is not None and cap._process is None
    cap.stop()  # idempotent
    gen.close()


def test_stream_capture_kills_stubborn_process(config, monkeypatch):
    class Stubborn:
        def __init__(self):
            self.killed = False
            self.waits = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise capture_mod.subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            self.killed = True

    cap = StreamCapture("x", config)
    cap._process = proc = Stubborn()
    cap.stop()
    assert proc.killed


def test_missing_streamlink(config, monkeypatch):
    monkeypatch.setattr(
        capture_mod,
        "require_tool",
        lambda name, hint="": (_ for _ in ()).throw(AudioError(f"'{name}' not found")),
    )
    with pytest.raises(AudioError, match="streamlink"):
        StreamCapture("https://kick.com/x", config).resolve_url()


def test_chunk_duration_must_be_positive(config, stream_file):
    with pytest.raises(ValueError):
        list(capture_mod.FileCapture(stream_file, config).capture_chunks(0, 0))
    with pytest.raises(AudioError):
        capture_mod.FileCapture(stream_file.with_name("nope.wav"), config)


# ── yt-dlp ───────────────────────────────────────────────────────


def test_download_audio_follows_output_template(bin_dir, tmp_path):
    fake_tool(
        bin_dir,
        "yt-dlp",
        """
        import sys, pathlib
        args = sys.argv[1:]
        assert "--extract-audio" in args and args[-1] == "https://youtu.be/x"
        out = pathlib.Path(args[args.index("--output") + 1].replace("%(ext)s", "wav"))
        out.write_bytes(b"RIFF")
    """,
    )
    path = download_audio("https://youtu.be/x", tmp_path / "dl")
    assert path.name == "audio.wav" and path.read_bytes() == b"RIFF"


@pytest.mark.parametrize(
    "body,match",
    [
        ("import sys; print('ERROR: video unavailable', file=sys.stderr); sys.exit(1)", "video unavailable"),
        ("pass", "no audio file"),
    ],
)
def test_download_audio_failures(bin_dir, tmp_path, body, match):
    fake_tool(bin_dir, "yt-dlp", body)
    with pytest.raises(AudioError, match=match):
        download_audio("https://youtu.be/x", tmp_path / "dl")


def test_download_audio_timeout(bin_dir, tmp_path):
    fake_tool(bin_dir, "yt-dlp", "import time; time.sleep(10)")
    with pytest.raises(AudioError, match="timed out"):
        download_audio("https://youtu.be/x", tmp_path / "dl", timeout=0.5)


# ── ffmpeg helpers ───────────────────────────────────────────────


def test_require_tool(monkeypatch):
    assert require_tool("ffmpeg")
    with pytest.raises(AudioError, match="not found. Install it"):
        require_tool("definitely-not-a-tool-xyz", "Install it")
    with pytest.raises(AudioError, match="on your PATH"):
        require_tool("definitely-not-a-tool-xyz")


def test_decode_timeout_and_empty(config, stream_file, monkeypatch, tmp_path):
    import subprocess

    import kickvoice.audio.ffmpeg as ff

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr(ff.subprocess, "run", slow)
    with pytest.raises(AudioError, match="timed out"):
        decode_to_pcm(stream_file, config.audio)
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    with pytest.raises(AudioError, match="No audio found"):
        decode_to_pcm(stream_file, config.audio)


def test_open_pcm_stream_and_tensor(config, stream_file):
    proc = open_pcm_stream(stream_file, config.audio)
    data = proc.stdout.read()
    proc.wait()
    assert len(data) >= SR * 2 * 8
    t = pcm_to_tensor(tone(440, 0.1))
    assert t.dtype.is_floating_point and t.shape[0] == int(SR * 0.1)
