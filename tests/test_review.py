import numpy as np
import pytest

from conftest import FakeCapture, FakeEncoder, ScriptedInput, conversation, voice_vector
from kickvoice.audio.vad import EnergyVAD
from kickvoice.pipelines.extract import ProfileExtractor
from kickvoice.pipelines.review import Reviewer
from kickvoice.pipelines.session import CREATED, MERGED, PENDING, SKIPPED, ExtractionSession
from kickvoice.speakers.profiles import ProfileStore

HOST, GUEST = 220, 550


@pytest.fixture
def session(config):
    pcm = conversation([(HOST, 6), (GUEST, 5), (HOST, 6), (GUEST, 5), (HOST, 6)])
    ex = ProfileExtractor(
        stream_url="https://kick.com/SomeChannel",
        config=config,
        capture=FakeCapture(pcm),
        vad=EnergyVAD(config.audio),
        encoder=FakeEncoder(),
        store=ProfileStore(config),
        out=lambda _: None,
    )
    return ex.run(max_duration_s=0)


def review(session, config, answers, store=None, player=None):
    store = store or ProfileStore(config)
    inp, out = ScriptedInput(answers), []
    summary = Reviewer(session, config, store=store, input_fn=inp, out=out.append, player=player).run()
    return summary, store, inp, "\n".join(out)


def test_create_with_suggested_channel_name_then_skip(config, session):
    # speaker_1: n, Enter (accept suggested "SomeChannel"); speaker_2: Enter (skip)
    summary, store, inp, out = review(session, config, ["n", "", ""])
    assert summary.created == ["SomeChannel"] and summary.skipped == 1
    assert "[SomeChannel]" in inp.prompts[1]
    assert store.get_embedding("SomeChannel") @ voice_vector(HOST) > 0.95
    p = store.get_profile("SomeChannel")
    assert p.source_urls == ["https://kick.com/SomeChannel"] and p.speech_seconds > 0
    reloaded = ExtractionSession.load(session.path)
    assert [c.status for c in reloaded.candidates] == [CREATED, SKIPPED]
    assert reloaded.candidates[0].profile_name == "SomeChannel"


def test_default_is_skip_and_nothing_saved(config, session):
    summary, store, _, _ = review(session, config, ["", ""])
    assert store.count == 0 and summary.skipped == 2


def test_quit_leaves_rest_pending_and_resume(config, session):
    summary, store, _, out = review(session, config, ["n", "Host", "q"])
    assert summary.created == ["Host"] and summary.remaining == 1
    assert "review --session" in out
    reloaded = ExtractionSession.load(session.path)
    assert [c.status for c in reloaded.candidates] == [CREATED, PENDING]

    # resume: only speaker_2 is offered
    summary2, store2, _, out2 = review(reloaded, config, ["n", "Guest"])
    assert summary2.created == ["Guest"] and summary2.remaining == 0
    assert "Speaker 1 of" not in out2 and "Speaker 2 of 2" in out2
    assert sorted(p.name for p in store2.list_profiles()) == ["Guest", "Host"]


def test_eof_is_treated_as_quit(config, session):
    summary, store, _, _ = review(session, config, [])
    assert summary.remaining == 2 and store.count == 0


def test_accept_match_merges_into_known_profile(config, session):
    store = ProfileStore(config)
    store.add_profile("TheHost", voice_vector(HOST) * 0.9 + voice_vector(999) * 0.1, speech_seconds=100)
    summary, store, _, out = review(session, config, ["a", "s"], store=store)
    assert "Matches known profile: TheHost" in out
    assert summary.merged == ["TheHost"]
    assert store.get_profile("TheHost").speech_seconds > 100
    assert ExtractionSession.load(session.path).candidates[0].status == MERGED


def test_new_profile_is_matched_for_later_candidates(config, session):
    """If the same voice was split in two clusters, approving the first one
    makes the second show up as a match."""
    dup = session.candidates[0]
    np.save(session.path / "dup.npy", session.embedding(dup))
    session.add_candidate(
        embedding=session.embedding(dup),
        sample_pcm=b"\x00\x00" * 10,
        sample_rate=16000,
        speech_seconds=20,
        n_windows=10,
        first_heard_s=30,
        cohesion=0.9,
    )
    summary, store, _, out = review(session, config, ["n", "Host", "s", "a"])
    assert "Speaker 3 of 3" in out
    assert out.count("Matches known profile: Host") == 1
    assert summary.created == ["Host"] and summary.merged == ["Host"]


def test_create_with_existing_name_offers_merge(config, session):
    store = ProfileStore(config)
    store.add_profile("Someone", voice_vector(12345), speech_seconds=50)
    # new profile named "someone" → exists → yes merge; then skip speaker 2
    summary, store, _, _ = review(session, config, ["n", "someone", "y", "s"], store=store)
    assert summary.merged == ["Someone"] and store.count == 1


def test_manual_merge_and_bad_names(config, session):
    store = ProfileStore(config)
    store.add_profile("Alt", voice_vector(777), speech_seconds=50)
    answers = [
        "m",
        "nobody",  # unknown → back to menu
        "x",  # unrecognised
        "m",
        "alt",  # merge (case-insensitive)
        "s",
    ]
    summary, store, _, out = review(session, config, answers, store=store)
    assert "No profile named 'nobody'" in out and "Unrecognised" in out
    assert summary.merged == ["Alt"]


def test_play_uses_player_and_falls_back(config, session):
    played = []
    review(session, config, ["p", "s", "s"], player=lambda p: played.append(p) or True)
    assert played and played[0].name == "speaker_1.wav"
    _, _, _, out = review(ExtractionSession.load(session.path), config, [], player=None)
    # (everything reviewed already — nothing left to play)
    assert "0 speaker(s) awaiting" in out


def test_model_mismatch_refuses(config, session):
    config.embedding.model_source = "other/model"
    summary, store, _, out = review(session, config, ["n", "X"])
    assert "can't be compared" in out and store.count == 0 and summary.remaining == 2
