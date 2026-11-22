"""Transcription pipeline end to end (fake ASR + fake voices, real plumbing)."""

import numpy as np
import pytest

from conftest import (
    FakeCapture,
    FakeEncoder,
    FakeRecognizer,
    conversation,
    silence,
    voice_vector,
)
from kickvoice.audio.vad import EnergyVAD
from kickvoice.pipelines.transcribe import TranscriptionPipeline, speaker_label
from kickvoice.speakers.profiles import ProfileStore
from kickvoice.transcripts.attribution import CONFIDENT, PROBABLE, UNKNOWN
from kickvoice.transcripts.refine import refine_session
from kickvoice.transcripts.store import TranscriptStore

HOST, GUEST, STRANGER = 220, 550, 880


@pytest.fixture
def profiles(config):
    store = ProfileStore(config)
    store.add_profile("Host", voice_vector(HOST), speech_seconds=600)
    store.add_profile("Guest", voice_vector(GUEST), speech_seconds=600)
    return store


@pytest.fixture
def transcripts(config):
    with TranscriptStore(config.transcripts_db) as store:
        yield store


def run(config, pcm, profiles, transcripts, recognizer=None, duration=0, **kw):
    lines = []
    pipeline = TranscriptionPipeline(
        stream_url="https://kick.com/SomeChannel",
        config=config,
        capture=FakeCapture(pcm),
        vad=EnergyVAD(config.audio),
        encoder=FakeEncoder(),
        recognizer=recognizer or FakeRecognizer(),
        profiles=profiles,
        transcripts=transcripts,
        on_utterance=lines.append,
        out=lambda _: None,
        **kw,
    )
    session = pipeline.run(max_duration_s=duration)
    return pipeline, session, lines


def voice_of(text):
    """The pitch the fake ASR heard for every word in a line."""
    return {int(t[1:]) for t in text.split()}


def test_each_line_attributed_to_the_right_speaker(config, profiles, transcripts):
    pcm = conversation([(HOST, 4), (GUEST, 3), (HOST, 5), (STRANGER, 3), (GUEST, 4)])
    _, session, lines = run(config, pcm, profiles, transcripts)

    expected = {HOST: "Host", GUEST: "Guest", STRANGER: None}
    stored = transcripts.utterances(session.id)
    assert len(stored) == 5 and [u.text for u in stored] == [u.text for u in lines]
    for u in stored:
        (voice,) = voice_of(u.text)  # every line is one voice
        assert u.speaker == expected[voice]
        assert u.attribution == (UNKNOWN if voice == STRANGER else CONFIDENT)
        assert u.embedding is not None and u.embedding.shape == (192,)
    # timestamps are relative to the capture start: gap 0.6s before each turn
    starts, t = [], 0.0
    for secs in (4, 3, 5, 3, 4):
        t += 0.6
        starts.append(t)
        t += secs
    assert [u.start_s for u in stored] == pytest.approx(starts, abs=0.1)


def test_utterance_across_block_boundary_is_not_split(config, profiles, transcripts):
    """Blocks are 10s; the host talks from ~7.6s to ~13.6s."""
    config.transcribe.chunk_duration_s = 10.0
    pcm = conversation([(GUEST, 6.4), (HOST, 6.0), (GUEST, 4)], gap=0.6)
    pipeline, session, _ = run(config, pcm, profiles, transcripts)

    host = [u for u in transcripts.utterances(session.id) if u.speaker == "Host"]
    assert len(host) == 1
    assert host[0].start_s < 10 < host[0].end_s
    assert host[0].duration == pytest.approx(6.0, abs=0.3)
    assert pipeline.stats.audio_seconds == pytest.approx(len(pcm) / 32000, abs=0.05)


def test_hallucinated_words_in_silence_are_dropped(config, profiles, transcripts):
    pcm = conversation([(HOST, 3), (GUEST, 3)], gap=2.0)
    _, session, _ = run(config, pcm, profiles, transcripts, recognizer=FakeRecognizer(hallucinate=True))
    texts = " ".join(u.text for u in transcripts.utterances(session.id))
    assert "thanks" not in texts and "w220" in texts


def test_short_turn_is_at_most_probable(config, profiles, transcripts):
    config.transcribe.min_speech_s = 0.5
    pcm = conversation([(HOST, 4), (GUEST, 0.8), (HOST, 4)], gap=1.5)
    _, session, _ = run(config, pcm, profiles, transcripts)
    short = [u for u in transcripts.utterances(session.id) if u.duration < 1.5]
    assert short and all(u.attribution != CONFIDENT for u in short)


def test_drop_unknown(config, profiles, transcripts):
    config.transcribe.keep_unattributed = False
    pcm = conversation([(HOST, 4), (STRANGER, 4)])
    pipeline, session, _ = run(config, pcm, profiles, transcripts)
    assert {u.speaker for u in transcripts.utterances(session.id)} == {"Host"}
    assert pipeline.stats.dropped_unattributed == 1


def test_duration_limit(config, profiles, transcripts):
    config.transcribe.chunk_duration_s = 5.0
    pcm = conversation([(HOST, 4)] * 8)
    pipeline, session, _ = run(config, pcm, profiles, transcripts, duration=10)
    assert transcripts.get_session(session.id).captured_seconds == 10
    assert all(u.start_s < 12 for u in transcripts.utterances(session.id))


def test_no_profiles_then_reattribute_after_enrolling(config, transcripts):
    """Transcribe first, enroll later: stored embeddings make it recoverable."""
    empty = ProfileStore(config)
    pcm = conversation([(HOST, 4), (GUEST, 4), (HOST, 4)])
    _, session, _ = run(config, pcm, empty, transcripts)
    before = transcripts.utterances(session.id)
    assert {u.attribution for u in before} == {UNKNOWN}
    # ...but the two voices are still told apart
    assert [u.voice_group for u in before] == [1, 2, 1]

    empty.add_profile("Host", voice_vector(HOST))
    names, matrix = empty.embedding_matrix()
    result = refine_session(transcripts, session.id, names, matrix, config)
    assert (result.changed, result.unchanged) == (2, 1)
    after = transcripts.utterances(session.id)
    assert [u.speaker for u in after] == ["Host", None, "Host"]
    assert after[0].attribution == CONFIDENT
    # idempotent
    assert refine_session(transcripts, session.id, names, matrix, config).changed == 0


def test_soundalike_guest_is_not_given_the_hosts_name(config, profiles, transcripts):
    """
    The failure mode seen on real meeting audio: an unenrolled guest on the
    same mic scores above the match threshold against the host's profile.
    Line by line they'd be called "Host"; the whole-session check gives the
    profile to the host's own (closer) voice group and leaves the guest unknown.
    """
    SOUNDALIKE = 330
    host = voice_vector(HOST)
    other = voice_vector(999)
    other -= (other @ host) * host
    other /= np.linalg.norm(other)
    lookalike = 0.7 * host + np.sqrt(1 - 0.7**2) * other  # cos(host) = 0.70
    config.extract.cluster_distance_threshold = 0.2  # calibrated to this "model"

    lines = []
    pipeline = TranscriptionPipeline(
        stream_url="https://kick.com/x",
        config=config,
        capture=FakeCapture(conversation([(HOST, 4), (SOUNDALIKE, 4), (HOST, 4), (SOUNDALIKE, 4)])),
        vad=EnergyVAD(config.audio),
        encoder=FakeEncoder(noise=0.05, voices={SOUNDALIKE: lookalike}),
        recognizer=FakeRecognizer(),
        profiles=profiles,
        transcripts=transcripts,
        on_utterance=lines.append,
        out=lambda _: None,
    )
    session = pipeline.run(max_duration_s=0)

    # live (line by line) the soundalike was mistaken for the host...
    assert [u.speaker for u in lines] == ["Host", "Host", "Host", "Host"]
    assert len({round(u.start_s) for u in lines}) == 4  # ...but kept as separate turns
    # ...the whole-session check fixes it
    final = transcripts.utterances(session.id)
    assert [u.speaker for u in final] == ["Host", None, "Host", None]
    host_group, guest_group = final[0].voice_group, final[1].voice_group
    assert final[2].voice_group == host_group and final[3].voice_group == guest_group
    assert host_group != guest_group
    assert speaker_label(final[1]) == f"(voice {guest_group})"
    assert pipeline.stats.refine.changed == 2


def test_two_unknown_people_get_separate_voice_groups(config, profiles, transcripts):
    pcm = conversation([(HOST, 4), (880, 4), (660, 4), (880, 4)])
    _, session, _ = run(config, pcm, profiles, transcripts)
    final = transcripts.utterances(session.id)
    labels = [speaker_label(u) for u in final]
    assert labels[0] == "Host"
    assert labels[1] == labels[3] != labels[2]  # 880 twice, 660 once
    assert all(lbl.startswith("(voice") for lbl in labels[1:])


def test_short_line_keeps_label_only_if_speaker_was_found(config, profiles, transcripts):
    from kickvoice.transcripts.store import Utterance

    session = transcripts.create_session("s", "src")
    long_host = Utterance(session_id="s", start_s=0, end_s=4, text="a", embedding=voice_vector(HOST))
    short_host = Utterance(
        session_id="s", start_s=5, end_s=5.6, text="b", speaker="Host", attribution=PROBABLE, similarity=0.7
    )
    short_guest = Utterance(
        session_id="s", start_s=6, end_s=6.6, text="c", speaker="Guest", attribution=PROBABLE, similarity=0.7
    )
    transcripts.add_utterances([long_host, short_host, short_guest])
    names, matrix = profiles.embedding_matrix()
    refine_session(transcripts, session.id, names, matrix, config)
    got = transcripts.utterances(session.id)
    assert [(u.speaker, u.attribution) for u in got] == [
        ("Host", CONFIDENT),
        ("Host", PROBABLE),
        (None, UNKNOWN),
    ]  # no Guest voice found


def test_session_metadata(config, profiles, transcripts):
    _, session, _ = run(config, conversation([(HOST, 4)]), profiles, transcripts)
    stored = transcripts.get_session(session.id)
    assert stored.source == "https://kick.com/SomeChannel"
    assert stored.asr_model == "fake-asr"
    assert stored.embedding_model == config.embedding.model_source
    assert stored.settings["profiles"] == ["Host", "Guest"]
    assert session.id.endswith("_somechannel")


def test_silence_only(config, profiles, transcripts):
    rec = FakeRecognizer()
    pipeline, session, _ = run(config, silence(25), profiles, transcripts, recognizer=rec)
    assert transcripts.utterances(session.id) == []
    assert rec.calls == []  # ASR never runs on pure silence


def test_keyboard_interrupt_saves_what_was_done(config, profiles, transcripts):
    class Interrupting(FakeCapture):
        def capture_chunks(self, chunk_duration_s=None, chunk_stride_s=None):
            for i, chunk in enumerate(super().capture_chunks(chunk_duration_s, chunk_stride_s)):
                if i == 2:
                    raise KeyboardInterrupt
                yield chunk

    config.transcribe.chunk_duration_s = 5.0
    pcm = conversation([(HOST, 4)] * 6)
    pipeline = TranscriptionPipeline(
        stream_url="https://kick.com/x",
        config=config,
        capture=Interrupting(pcm),
        vad=EnergyVAD(config.audio),
        encoder=FakeEncoder(),
        recognizer=FakeRecognizer(),
        profiles=profiles,
        transcripts=transcripts,
        on_utterance=None,
        out=lambda _: None,
    )
    session = pipeline.run(max_duration_s=0)
    stored = transcripts.utterances(session.id)
    assert stored and all(u.speaker == "Host" for u in stored)
    assert transcripts.get_session(session.id).captured_seconds == 10


def test_merged_voice_group_weak_lines_are_not_confident(config, profiles, transcripts):
    """
    The other real failure mode: two similar voices' turns (separate lines,
    split by pauses) land in ONE group because the clustering can't separate
    them, and the group matches the host. The guest's lines individually
    score much lower against the host profile, so they must not be confident.

    (If the voices are so close that the turns themselves aren't split, the
    guest's words end up inside the host's line — no later stage can undo
    that. That's the embedding model's limit, documented in the README.)
    """
    SOUNDALIKE = 330
    host = voice_vector(HOST)
    other = voice_vector(999)
    other -= (other @ host) * host
    other /= np.linalg.norm(other)
    lookalike = 0.68 * host + np.sqrt(1 - 0.68**2) * other  # cos(host) = 0.68
    config.extract.cluster_distance_threshold = 0.7  # loose: groups merge

    encoder = FakeEncoder(voices={SOUNDALIKE: lookalike})
    pipeline = TranscriptionPipeline(
        stream_url="https://kick.com/x",
        config=config,
        capture=FakeCapture(conversation([(HOST, 4), (SOUNDALIKE, 4), (HOST, 4)], gap=1.5)),
        vad=EnergyVAD(config.audio),
        encoder=encoder,
        recognizer=FakeRecognizer(),
        profiles=profiles,
        transcripts=transcripts,
        on_utterance=None,
        out=lambda _: None,
    )
    session = pipeline.run(max_duration_s=0)
    final = transcripts.utterances(session.id)
    assert len({u.voice_group for u in final}) == 1  # merged
    assert [(u.speaker, u.attribution) for u in final] == [
        ("Host", CONFIDENT),
        ("Host", PROBABLE),
        ("Host", CONFIDENT),
    ]
    assert final[1].similarity < 0.75 < final[0].similarity  # line's own score stored


def test_model_failure_happens_before_capture_or_session(config, profiles, transcripts):
    from kickvoice.transcripts.asr import ASRUnavailable

    class Broken(FakeRecognizer):
        def load(self):
            raise ASRUnavailable("no model")

    capture = FakeCapture(conversation([(HOST, 4)]))
    pipeline = TranscriptionPipeline(
        stream_url="https://kick.com/x",
        config=config,
        capture=capture,
        vad=EnergyVAD(config.audio),
        encoder=FakeEncoder(),
        recognizer=Broken(),
        profiles=profiles,
        transcripts=transcripts,
        on_utterance=None,
        out=lambda _: None,
    )
    with pytest.raises(ASRUnavailable):
        pipeline.run()
    assert transcripts.list_sessions() == []  # no empty session left behind
    assert capture.requested is None  # stream never opened
