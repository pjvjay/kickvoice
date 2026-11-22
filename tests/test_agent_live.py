"""Live sessions for agents: the event log, points, voice profiling, enrolment mid-stream."""

import time

import pytest

from conftest import ScriptedLLM, conversation, fake_pipeline_factory, voice_vector
from kickvoice.agent.service import AgentService, ToolError
from kickvoice.research.store import Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore

JANE, ME = 220, 550
URL = "https://kick.com/debate"


@pytest.fixture
def cfg(config):
    config.debate.chunk_duration_s = 5.0
    config.debate.min_point_words = 5
    config.debate.point_pause_s = 0.5
    config.agent.refine_interval_s = 0.5
    config.agent.min_enroll_speech_s = 5.0
    return config


def jane_record(config, voice="Jane"):
    """Research subject 'Jane Pundit' with one verified claim whose words match the fake ASR."""
    with ResearchStore(config.research_db) as store:
        p = store.add_person("Jane Pundit", voice_profile=voice)
        doc = Document(
            p.id,
            "x",
            "post",
            "1",
            "w220 w220 w220 is my view on this",
            "self",
            url="https://x.com/jane/status/1",
            published_at="2022-02-02",
        )
        store.upsert_document(doc, [Chunk(doc.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        (cid,) = store.add_claims(
            [
                Claim(
                    p.id,
                    doc.id,
                    chunk.id,
                    "w220",
                    "Holds the w220 view",
                    "w220 w220 w220 is my view on this",
                    published_at="2022-02-02",
                    url=doc.url,
                )
            ]
        )
    return cid


def drain(service, session_id, timeout=20.0, until=None):
    """Poll like an agent would, until the session ends (or `until(events)` is true)."""
    events, cursor, deadline = [], 0, time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = service.get_live_updates(session_id, cursor, wait_seconds=1.0)
        events += r["events"]
        cursor = r["next_cursor"]
        if until and until(events):
            return events, cursor
        if r["status"] in ("finished", "stopped", "error") and not r["events"]:
            return events, cursor
    raise AssertionError(f"timed out; got {len(events)} events")


def test_live_session_end_to_end(cfg):
    profiles = ProfileStore(cfg)
    profiles.add_profile("Jane", voice_vector(JANE), speech_seconds=300)
    profiles.add_profile("Me", voice_vector(ME), speech_seconds=300)
    cid = jane_record(cfg)
    llm = ScriptedLLM(
        [
            {
                "is_argument": True,
                "their_point": "Jane pushes w220",
                "counters": [
                    {
                        "text": 'you said "w220 w220 w220 is my view" back in 2022, so what changed?',
                        "evidence": [f"C{cid}"],
                    },
                    {"text": 'you said "w220 is evil" in 2019', "evidence": [f"C{cid}"]},
                ],
                "contradictions": [],
            }
        ]
    )
    service = AgentService(
        cfg,
        pipeline_factory=fake_pipeline_factory(
            conversation([(JANE, 5), (ME, 4), (JANE, 5), (ME, 3)]), delay=0.05
        ),
        llm_factory=lambda c: llm,
    )

    started = service.start_live_session(URL, opponent="Jane Pundit")
    assert started["opponent"] == "Jane Pundit (voice: Jane)" and not started["warnings"]
    sid = started["session_id"]
    events, _ = drain(service, sid)

    lines = [e for e in events if e["type"] == "line"]
    points = [e for e in events if e["type"] == "point"]
    assert [line["speaker"] for line in lines] == ["Jane", "Me", "Jane", "Me"]
    assert [line["is_opponent"] for line in lines] == [True, False, True, False]
    assert [p["point_id"] for p in points] == ["p1", "p2"]
    assert any(e["type"] == "voices" for e in events)
    assert events[-1] == {**events[-1], "type": "status", "status": "finished"}
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    evidence = service.get_point_evidence(sid, "p1")
    assert evidence["opponent"] == "Jane Pundit"
    top = evidence["evidence"][0]
    assert top["ref"] == f"C{cid}" and top["first_hand"] and top["date"] == "2022-02-02"
    assert "verify_message" in evidence["rules"]

    s = service.suggest_counters(sid, "p1")
    good, bad = s["replies"]
    assert good["verification"]["ok"], good["verification"]
    assert good["verification"]["verified_quotes"][0]["ref"] == f"C{cid}"
    assert not bad["verification"]["ok"]  # invented quote + unsupported 2019
    assert len(bad["verification"]["issues"]) == 2
    assert "stream's live chat" in llm.calls[0]["system"]  # chat style requested

    voices = service.list_voices(sid)["voices"]
    assert {v["speaker"] for v in voices} == {"Jane", "Me"}
    assert [v["is_opponent"] for v in voices if v["speaker"] == "Jane"] == [True]


def test_unknown_opponent_enrolled_mid_stream(cfg):
    """The agent spots an unrecognised voice, enrols it, and points start flowing."""
    ProfileStore(cfg).add_profile("Me", voice_vector(ME), speech_seconds=300)
    jane_record(cfg, voice=None)
    turns = [(JANE, 6), (ME, 4)] * 6
    service = AgentService(cfg, pipeline_factory=fake_pipeline_factory(conversation(turns), delay=0.4))

    started = service.start_live_session(URL, opponent="Jane Pundit")
    assert "no voice profile" in started["warnings"][0]
    sid = started["session_id"]

    def two_unknown_lines(events):
        return sum(e["type"] == "line" and e["speaker"] == "(unknown)" for e in events) >= 2

    events, cursor = drain(service, sid, until=two_unknown_lines)
    assert not any(e["type"] == "point" for e in events)  # nobody recognised yet

    voices = service.list_voices(sid)["voices"]
    stranger = [v for v in voices if v["speaker"] is None][0]
    assert stranger["speech_seconds"] >= 5
    enrolled = service.enroll_voice(
        sid, stranger["sample_lines"][0]["utterance_id"], "Jane", research_subject="Jane Pundit"
    )
    assert enrolled["voice_profile"] == "Jane" and enrolled["linked_research_subject"] == "Jane Pundit"
    service.set_opponent(sid, "Jane Pundit")

    rest, _ = drain(service, sid)
    points = [e for e in rest if e["type"] == "point"]
    assert points, "no points after enrolment"
    later_jane = [e for e in rest if e["type"] == "line" and e["speaker"] == "Jane"]
    assert later_jane and all(e["is_opponent"] for e in later_jane)
    with ResearchStore(cfg.research_db) as store:
        assert store.get_person("Jane Pundit").voice_profile == "Jane"


def test_enroll_voice_guards(cfg):
    ProfileStore(cfg).add_profile("Me", voice_vector(ME))
    service = AgentService(
        cfg, pipeline_factory=fake_pipeline_factory(conversation([(JANE, 3), (ME, 3)]), delay=0.0)
    )
    sid = service.start_live_session(URL)["session_id"]
    drain(service, sid)
    voices = service.list_voices(sid)["voices"]
    jane_line = [v for v in voices if v["speaker"] is None][0]["sample_lines"][0]["utterance_id"]
    with pytest.raises(ToolError, match="Only 3s of speech"):
        service.enroll_voice(sid, jane_line, "Jane")
    with pytest.raises(ToolError, match="isn't part of this session"):
        service.enroll_voice(sid, 99999, "Jane")
    cfg.agent.min_enroll_speech_s = 1
    with pytest.raises(ToolError, match="already exists"):
        service.enroll_voice(sid, jane_line, "me")
    assert service.enroll_voice(sid, jane_line, "Me", merge_if_exists=True)["merged"]


def test_stop_and_unknown_session(cfg):
    service = AgentService(
        cfg, pipeline_factory=fake_pipeline_factory(conversation([(JANE, 5)] * 20), delay=0.3)
    )
    sid = service.start_live_session(URL, any_speaker=True)["session_id"]
    time.sleep(0.5)
    assert service.stop_live_session(sid)["status"] == "stopped"
    assert service.list_live_sessions()["sessions"][0]["status"] == "stopped"
    with pytest.raises(ToolError, match="No live session"):
        service.get_live_updates("nope")


def test_opponent_resolution(cfg):
    service = AgentService(cfg)
    with pytest.raises(ToolError, match="neither a research subject nor a voice profile"):
        service.start_live_session(URL, opponent="Nobody")
    with pytest.raises(ToolError, match="neither a URL nor an existing file"):
        service.start_live_session("not/a/file.wav")
    ProfileStore(cfg).add_profile("Bob", voice_vector(880))
    opp, warnings = service._resolve_opponent("bob", None, False)
    assert opp.voice == "Bob" and "without research" in warnings[0]


def test_long_poll_waits_then_returns_empty(cfg):
    service = AgentService(
        cfg, pipeline_factory=fake_pipeline_factory(conversation([(JANE, 5)] * 10), delay=5.0)
    )  # very slow stream
    sid = service.start_live_session(URL)["session_id"]
    service.get_live_updates(sid, 0, wait_seconds=0.5)  # the initial status event
    cursor = service.get_live_updates(sid, 0)["next_cursor"]
    started = time.monotonic()
    r = service.get_live_updates(sid, cursor, wait_seconds=0.6)
    assert 0.5 <= time.monotonic() - started < 3 and r["events"] == []
    assert r["next_cursor"] == cursor
    service.stop_live_session(sid)
