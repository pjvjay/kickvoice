"""Enrollment pipeline, including the 'downloads every video twice' regression."""

from pathlib import Path

import numpy as np
import pytest

from conftest import SR, FakeEncoder, conversation, tone, voice_vector
from kickvoice.audio.ffmpeg import write_wav
from kickvoice.audio.vad import EnergyVAD
from kickvoice.pipelines.enroll import Enroller, is_url
from kickvoice.speakers.profiles import ProfileStore

VOICE = 220


class FakeDownloader:
    """Stands in for yt-dlp; records every call."""

    def __init__(self, pcm: bytes):
        self.pcm = pcm
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, url: str, out_dir: Path) -> Path:
        self.calls.append((url, Path(out_dir)))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audio.wav"
        write_wav(path, self.pcm, SR)
        return path


def make_enroller(config, downloader=None):
    return Enroller(
        config=config,
        encoder=FakeEncoder(),
        vad=EnergyVAD(config.audio),
        store=ProfileStore(config),
        downloader=downloader or FakeDownloader(tone(VOICE, 6)),
    )


def test_each_url_is_downloaded_exactly_once(config):
    """Regression: enroll used to call yt-dlp twice for every URL."""
    downloader = FakeDownloader(tone(VOICE, 6))
    enroller = make_enroller(config, downloader)
    urls = ["https://youtu.be/AAA", "https://youtu.be/BBB"]

    enroller.enroll("Streamer", urls)

    assert [url for url, _ in downloader.calls] == urls  # once each, in order
    assert len({d for _, d in downloader.calls}) == 2  # separate work dirs


def test_enroll_from_youtube_wrapper_downloads_once(config):
    downloader = FakeDownloader(tone(VOICE, 6))
    make_enroller(config, downloader).enroll_from_youtube("S", ["https://youtu.be/AAA"])
    assert len(downloader.calls) == 1


def test_enroll_mixed_sources_builds_one_profile(config, tmp_path):
    clip = tmp_path / "clip.wav"
    write_wav(clip, conversation([(VOICE, 5), (VOICE, 4)]), SR)
    downloader = FakeDownloader(tone(VOICE, 6))
    enroller = make_enroller(config, downloader)

    seen = []
    result = enroller.enroll(
        "Streamer",
        [clip, "https://youtu.be/AAA"],
        notes="two sources",
        on_progress=lambda s, i, n: seen.append((i, n)),
    )

    assert len(downloader.calls) == 1  # the local file is not downloaded
    assert seen == [(1, 2), (2, 2)]
    assert result.n_segments == 3  # 2 from the file + 1 from the URL
    assert result.speech_seconds > 10

    stored = ProfileStore(config).get_profile("Streamer")
    assert stored.notes == "two sources"
    assert stored.speech_seconds == pytest.approx(result.speech_seconds, abs=0.1)
    assert stored.embedding_model == config.embedding.model_source
    voiceprint = ProfileStore(config).get_embedding("Streamer")
    assert voiceprint @ voice_vector(VOICE) > 0.95
    assert np.isclose(np.linalg.norm(voiceprint), 1.0)


def test_enroll_without_sources_or_speech(config, tmp_path):
    with pytest.raises(ValueError, match="No audio sources"):
        make_enroller(config).enroll("X", [])

    quiet = tmp_path / "quiet.wav"
    write_wav(quiet, b"\x00\x00" * SR * 3, SR)
    with pytest.raises(ValueError, match="No speech detected"):
        make_enroller(config).enroll("X", [quiet])
    assert ProfileStore(config).count == 0


def test_temp_downloads_are_cleaned_up(config):
    downloader = FakeDownloader(tone(VOICE, 6))
    make_enroller(config, downloader).enroll("S", ["https://youtu.be/AAA"])
    assert not downloader.calls[0][1].exists()  # temp dir removed after enrolling


@pytest.mark.parametrize(
    "source,expected",
    [
        ("https://youtu.be/AAA", True),
        ("http://x.com/a", True),
        ("/tmp/clip.wav", False),
        ("clip.wav", False),
    ],
)
def test_is_url(source, expected):
    assert is_url(source) is expected
