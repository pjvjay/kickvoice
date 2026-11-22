import json
import wave

import pytest

from conftest import FakeCapture, FakeEncoder, conversation, voice_vector
from kickvoice.pipelines.extract import ProfileExtractor
from kickvoice.pipelines.session import ExtractionSession, channel_name, find_session, source_slug
from kickvoice.audio.vad import EnergyVAD
from kickvoice.speakers.profiles import ProfileStore

HOST, GUEST, CAMEO = 220, 550, 880   # three synthetic "voices"


def talk_show():
    # host ~30s, guest ~18s, cameo 3s (below min speech → dropped)
    return conversation([(HOST, 6), (GUEST, 4), (HOST, 5), (GUEST, 5), (HOST, 7),
                         (CAMEO, 3), (HOST, 6), (GUEST, 5), (HOST, 6), (GUEST, 4)])


def make(config, pcm, store=None, url="https://kick.com/SomeChannel"):
    lines = []
    ex = ProfileExtractor(
        stream_url=url, config=config, capture=FakeCapture(pcm),
        vad=EnergyVAD(config.audio), encoder=FakeEncoder(),
        store=store or ProfileStore(config), out=lines.append,
    )
    return ex, lines


def test_discovers_each_voice(config):
    ex, lines = make(config, talk_show())
    session = ex.run(max_duration_s=0)

    assert [c.id for c in session.candidates] == ["speaker_1", "speaker_2"]
    host, guest = session.candidates
    assert 27 <= host.speech_seconds <= 31 and 15 <= guest.speech_seconds <= 19
    assert host.first_heard_s < guest.first_heard_s
    # centroid really is that voice
    assert session.embedding(host) @ voice_vector(HOST) > 0.95
    assert session.embedding(guest) @ voice_vector(GUEST) > 0.95
    # review clip exists, is valid audio, and is ~sample_clip_s long
    with wave.open(str(session.sample_path(host))) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert 15 <= w.getnframes() / 16000 <= 26
    # persisted
    data = json.loads((session.path / "session.json").read_text())
    assert len(data["candidates"]) == 2 and data["source"].endswith("SomeChannel")
    assert session.id.endswith("_somechannel")
    assert any("Found 2 speaker(s)" in l for l in lines)


def test_non_overlapping_blocks_requested_and_capture_closed(config):
    ex, _ = make(config, talk_show())
    ex.run(max_duration_s=0)
    assert ex.capture.requested == (10.0, 10.0)
    assert ex.capture.closed


def test_duration_limit_stops_capture(config):
    ex, _ = make(config, talk_show())
    ex.run(max_duration_s=20)
    assert ex.captured_seconds == 20
    assert ex.capture.closed


def test_matches_known_profile(config):
    store = ProfileStore(config)
    store.add_profile("TheHost", voice_vector(HOST), speech_seconds=600)
    ex, lines = make(config, talk_show(), store=store)
    host, guest = ex.run(max_duration_s=0).candidates
    assert host.matched and host.best_match == "TheHost" and host.best_similarity > 0.9
    assert not guest.matched and guest.best_match == "TheHost"
    assert any("✅ TheHost" in l for l in lines)


def test_no_speech_returns_none(config):
    ex, lines = make(config, b"\x00\x00" * 16000 * 25)
    assert ex.run(max_duration_s=0) is None
    assert not config.extractions_dir.exists() or not any(config.extractions_dir.iterdir())


def test_too_little_speech_returns_none(config):
    config.extract.min_speaker_speech_s = 60
    ex, lines = make(config, talk_show())
    assert ex.run(max_duration_s=0) is None
    assert any("--min-speech" in l for l in lines)


def test_keyboard_interrupt_keeps_what_was_captured(config):
    class InterruptingCapture(FakeCapture):
        def capture_chunks(self, chunk_duration_s=None, chunk_stride_s=None):
            gen = super().capture_chunks(chunk_duration_s, chunk_stride_s)
            for i, chunk in enumerate(gen):
                if i == 4:
                    raise KeyboardInterrupt
                yield chunk

    lines = []
    ex = ProfileExtractor(stream_url="https://kick.com/x", config=config,
                          capture=InterruptingCapture(talk_show()),
                          vad=EnergyVAD(config.audio), encoder=FakeEncoder(),
                          store=ProfileStore(config), out=lines.append)
    session = ex.run(max_duration_s=0)
    assert session is not None and len(session.candidates) >= 1
    assert any("Stopped listening after 40s" in l for l in lines)


@pytest.mark.parametrize("url,expected", [
    ("https://kick.com/SomeChannel", "SomeChannel"),
    ("kick.com/xqc?foo=1", "xqc"),
    ("https://www.kick.com/adin/videos/abc-123", "adin"),
    ("https://kick.com/video/3f2a", None),
    ("https://twitch.tv/foo", None),
    ("/tmp/vod.mp4", None),
])
def test_channel_name(url, expected):
    assert channel_name(url) == expected


def test_source_slug_for_files():
    assert source_slug("/tmp/My VOD.mp4") == "my-vod"


def test_find_session(config):
    root = config.extractions_dir
    a = ExtractionSession.create(root, "https://kick.com/a", "m")
    a.save()
    assert find_session(root, None) is None            # no pending candidates
    assert find_session(root, a.id).id == a.id
    assert find_session(root, str(a.path)).id == a.id
    assert find_session(root, "nope") is None
