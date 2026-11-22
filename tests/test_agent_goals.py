"""
The spawned agents, end to end (in-process MCP server, synthetic voices):
extract-profiles and monitor, each with the rules policy and a scripted model.
"""

import json
import wave

import pytest

from conftest import (
    ScriptedToolLLM,
    conversation,
    episode_brief,
    fake_pipeline_factory,
    last_result,
    voice_vector,
)
from kickvoice.agent.runner import run_agent
from kickvoice.agent.service import AgentService
from kickvoice.agent.terminal import ScriptedTerminal
from kickvoice.agent.trace import read_trace
from kickvoice.research.store import Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore

HOST, GUEST, JANE, ME = 220, 550, 220, 550
URL = "https://kick.com/somechannel"


@pytest.fixture
def cfg(config):
    config.debate.chunk_duration_s = 5.0
    config.debate.min_point_words = 5
    config.debate.point_pause_s = 0.5
    config.agent.poll_wait_s = 0.5
    config.agent.refine_interval_s = 0.5
    config.agent.min_enroll_speech_s = 5.0
    config.agent.identify_after_s = 5.0
    config.agent.reply_timeout_s = 10.0
    return config


def service_for(cfg, turns, delay=0.05, clipboard=None):
    return AgentService(
        cfg,
        pipeline_factory=fake_pipeline_factory(conversation(turns), delay=delay),
        clipboard=clipboard or (lambda t: (True, "fake")),
    )


def run(command, cfg, options, service, policy="rules", llm=None, terminal=None, echo=None):
    lines = []
    outcome = run_agent(
        command,
        cfg,
        {"source": URL, **options},
        policy=policy,
        llm=llm,
        terminal=terminal or ScriptedTerminal([]),
        service=service,
        echo=echo or lines.append,
    )
    service.close()
    return outcome, lines


def trace_of(outcome):
    return read_trace(outcome.run_dir.trace)


# ── extract-profiles ─────────────────────────────────────────────

TALK_SHOW = [(HOST, 6), (GUEST, 5)] * 3


def test_extract_rules_policy_creates_approved_profile(cfg):
    terminal = ScriptedTerminal(["Host Person", ""])  # name the 1st voice, skip the 2nd
    outcome, lines = run(
        "extract-profiles",
        cfg,
        {"duration_s": None, "min_speech_s": 8},
        service_for(cfg, TALK_SHOW),
        terminal=terminal,
    )
    r = outcome.report
    assert outcome.ok and r["profiles_created"] == ["Host Person"]
    assert [d["decision"] for d in r["decisions"]] == ["created", "skipped"]
    assert ProfileStore(cfg).find_profile("Host Person") is not None
    # approval was asked with evidence: sample lines and a playable clip
    assert "Who is voice 1?" in terminal.questions[0]
    clip = r["decisions"][0]["clip"]
    assert clip.startswith(str(outcome.run_dir.clips))
    with wave.open(clip) as w:
        assert w.getnframes() / w.getframerate() >= 5
    # outputs
    md = outcome.run_dir.report_md.read_text()
    assert "Profiles created: Host Person" in md and "Clip:" in md
    assert json.loads(outcome.run_dir.report_json.read_text())["ok"] is True
    kinds = [e["kind"] for e in trace_of(outcome)]
    for kind in (
        "run_start",
        "tools_loaded",
        "tool_call",
        "live_event",
        "user_question",
        "user_answer",
        "decision",
        "run_end",
    ):
        assert kind in kinds, kind
    assert {e["actor"] for e in trace_of(outcome) if e["kind"] == "tool_call"} == {"orchestrator", "rules"}


def test_extract_without_a_user_leaves_everything_pending(cfg):
    outcome, _ = run(
        "extract-profiles",
        cfg,
        {"min_speech_s": 8},
        service_for(cfg, TALK_SHOW),
        terminal=ScriptedTerminal([], interactive=False),
    )
    assert [d["decision"] for d in outcome.report["decisions"]] == ["pending", "pending"]
    assert ProfileStore(cfg).list_profiles() == []  # nothing created unapproved


def test_extract_recognises_known_voices(cfg):
    ProfileStore(cfg).add_profile("Host", voice_vector(HOST), speech_seconds=300)
    outcome, _ = run(
        "extract-profiles",
        cfg,
        {"min_speech_s": 8},
        service_for(cfg, TALK_SHOW),
        terminal=ScriptedTerminal(["Guesty"]),
    )
    decisions = {d["decision"]: d for d in outcome.report["decisions"]}
    assert decisions["recognized"]["name"] == "Host"
    assert decisions["created"]["name"] == "Guesty"


def extract_agent_script(answers_seen):
    """A model that does the identify job the way the prompt asks."""

    def script(system, messages, tools):
        brief = episode_brief(messages)
        if "start_live_session" in tools and "list_voices" not in tools:  # setup
            if last_result(messages, "start_live_session") is None:
                return "Starting.", [("start_live_session", {"source": URL, "keep_audio": True})]
            return "", [("finish", {"summary": "started"})]
        session = brief.split("Live session id: ")[1].split()[0]
        voices = last_result(messages, "list_voices")
        if voices is None:
            return "", [("list_voices", {"session_id": session})]
        target = [v for v in voices["voices"] if not v["speaker"]][0]
        anchor = target["sample_lines"][0]["utterance_id"]
        if last_result(messages, "export_voice_clip") is None:
            return "", [("export_voice_clip", {"session_id": session, "utterance_id": anchor})]
        answer = last_result(messages, "ask_user")
        if answer is None:
            clip = last_result(messages, "export_voice_clip")["path"]
            return "The channel owner introduces herself as Sam.", [
                ("ask_user", {"question": "Is voice 1 Sam (the channel owner)?", "details": f"clip: {clip}"})
            ]
        answers_seen.append(answer)
        if last_result(messages, "enroll_voice") is None:
            return "", [
                ("enroll_voice", {"session_id": session, "utterance_id": anchor, "name": answer["answer"]})
            ]
        if last_result(messages, "record_decision") is None:
            return "", [
                (
                    "record_decision",
                    {
                        "voice": target["label"],
                        "decision": "created",
                        "name": answer["answer"],
                        "confidence": "high",
                        "reasoning": "introduced herself; user confirmed",
                    },
                )
            ]
        return "Done: created one profile.", [("finish", {"summary": "1 created"})]

    return script


def test_extract_llm_policy_agent_does_the_job(cfg):
    answers = []
    llm = ScriptedToolLLM(extract_agent_script(answers))
    outcome, lines = run(
        "extract-profiles",
        cfg,
        {"min_speech_s": 8},
        service_for(cfg, TALK_SHOW),
        policy="llm",
        llm=llm,
        terminal=ScriptedTerminal(["Sam"]),
    )
    r = outcome.report
    assert outcome.ok and r["profiles_created"] == ["Sam"]
    assert r["decisions"][0]["confidence"] == "high"
    assert ProfileStore(cfg).find_profile("Sam")
    assert [e["goal"] for e in r["stats"]["episodes"]] == ["setup", "identify"]
    assert all(e["outcome"] == "finished" for e in r["stats"]["episodes"])
    assert r["stats"]["llm_calls"] == len(llm.calls) and r["stats"]["input_tokens"] > 0
    # the model never got tools outside its goal
    assert "enroll_voice" not in llm.calls[0]["tools"] and "enroll_voice" in llm.calls[-1]["tools"]
    trace = trace_of(outcome)
    assert any(e["kind"] == "llm_response" and e["tool_calls"] for e in trace)
    assert any(e["kind"] == "tool_call" and e["actor"] == "identify" for e in trace)
    assert "2 model call" not in outcome.run_dir.report_md.read_text()  # real numbers rendered


def test_extract_falls_back_if_agent_doesnt_start_the_session(cfg):
    def lazy(system, messages, tools):
        return "I'd rather not.", []

    outcome, lines = run(
        "extract-profiles",
        cfg,
        {"min_speech_s": 8},
        service_for(cfg, TALK_SHOW),
        policy="llm",
        llm=ScriptedToolLLM(lazy),
    )
    assert outcome.ok and outcome.report["session_id"]
    assert any("didn't start the live session" in e.get("message", "") for e in trace_of(outcome))


# ── monitor ──────────────────────────────────────────────────────

DEBATE = [(JANE, 5), (ME, 4), (JANE, 5), (ME, 3)]


def jane_record(cfg, voice="Jane"):
    with ResearchStore(cfg.research_db) as store:
        p = store.add_person("Jane Pundit", voice_profile=voice)
        doc = Document(
            p.id,
            "x",
            "post",
            "1",
            "w220 w220 w220 w220 is my honest view",
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
                    "w220 w220 w220 w220 is my honest view",
                    published_at="2022-02-02",
                    url=doc.url,
                )
            ]
        )
    return cid


def enrolled(cfg):
    profiles = ProfileStore(cfg)
    profiles.add_profile("Jane", voice_vector(JANE), speech_seconds=300)
    profiles.add_profile("Me", voice_vector(ME), speech_seconds=300)


def test_monitor_rules_policy_offers_verified_receipts_and_copies(cfg):
    enrolled(cfg)
    jane_record(cfg)
    copied = []
    terminal = ScriptedTerminal([])

    def echo(line):
        lines.append(line)
        if "[1] " in line and not copied:  # the user picks reply 1
            terminal.feed("1")

    lines = []
    service = service_for(cfg, DEBATE, clipboard=lambda t: (copied.append(t) or True, "fake"))
    outcome, _ = run(
        "monitor", cfg, {"opponent": "Jane Pundit", "me": "Me"}, service, terminal=terminal, echo=echo
    )
    r = outcome.report
    assert outcome.ok and r["opponent"] == "Jane Pundit" and r["opponent_heard"]
    assert [p["point_id"] for p in r["points"]] == ["p1", "p2"]
    first = r["replies"][0]
    assert first["text"].startswith('you said "w220 w220 w220 w220 is my honest view" (2022-02)')
    assert first["sources"][0]["date"] == "2022-02-02"  # verified against the record
    assert r["copied"] == [1] and copied == [first["text"]]
    md = outcome.run_dir.report_md.read_text()
    assert "📋 copied" in md and "### [p1]" in md
    assert any(e["kind"] == "user_command" and e.get("copied") for e in trace_of(outcome))


def reply_agent_script(cid):
    """Propose one good reply and one misquote; fix after the fact-check rejects it."""

    def script(system, messages, tools):
        brief = episode_brief(messages)
        if "get_point_evidence" not in tools:  # setup episode
            if last_result(messages, "research_profile") is None:
                return "", [("research_profile", {"name": "Jane Pundit"})]
            if last_result(messages, "start_live_session") is None:
                return "", [("start_live_session", {"source": URL, "opponent": "Jane Pundit", "me": "Me"})]
            return "", [("note", {"message": "Jane has 1 claim on record."}), ("finish", {"summary": "ok"})]
        session = brief.split("Live session: ")[1].split()[0]
        point = brief.split("Their point [")[1].split("]")[0]
        if last_result(messages, "get_point_evidence") is None:
            return "", [("get_point_evidence", {"session_id": session, "point_id": point})]
        proposed = last_result(messages, "propose_replies")
        if proposed is None:
            return "", [
                (
                    "propose_replies",
                    {
                        "point_id": point,
                        "replies": [
                            {
                                "text": 'you said "w220 w220 w220 w220 is my honest view" in 2022',
                                "refs": [f"C{cid}"],
                            },
                            {"text": 'you literally said "w220 is evil"', "refs": []},
                        ],
                    },
                )
            ]
        if proposed["rejected"] and len([m for m in messages if m.get("name") == "propose_replies"]) == 1:
            return "", [
                (
                    "propose_replies",
                    {"point_id": point, "replies": [{"text": "so which is it, w220 or not?"}]},
                )
            ]
        return "", [("finish", {"summary": "proposed"})]

    return script


def test_monitor_llm_policy_replies_are_fact_checked(cfg):
    enrolled(cfg)
    cid = jane_record(cfg)
    llm = ScriptedToolLLM(reply_agent_script(cid))
    outcome, lines = run(
        "monitor",
        cfg,
        {"opponent": "Jane Pundit", "me": "Me"},
        service_for(cfg, DEBATE),
        policy="llm",
        llm=llm,
    )
    r = outcome.report
    assert outcome.ok
    texts = [x["text"] for x in r["replies"]]
    assert 'you said "w220 w220 w220 w220 is my honest view" in 2022' in texts
    assert "so which is it, w220 or not?" in texts  # the agent's fix
    assert any("w220 is evil" in x["text"] for x in r["rejected_replies"])  # misquote never shown
    assert not any("w220 is evil" in line for line in lines if "[" in line and "]" in line[:8])
    goals = [e["goal"] for e in r["stats"]["episodes"]]
    assert goals[0] == "setup" and goals[1].startswith("reply:p")
    # reply episodes use the fast model
    reply_calls = [c for c in llm.calls if "get_point_evidence" in c["tools"]]
    assert reply_calls and all(c["model"] == cfg.llm.live_model for c in reply_calls)
    assert "rejected by the fact-check" in outcome.run_dir.report_md.read_text()


def test_monitor_llm_identifies_and_enrolls_the_opponent(cfg):
    ProfileStore(cfg).add_profile("Me", voice_vector(ME), speech_seconds=300)
    jane_record(cfg, voice=None)

    def script(system, messages, tools):
        brief = episode_brief(messages)
        if "start_live_session" in tools:
            if last_result(messages, "start_live_session") is None:
                return "", [("start_live_session", {"source": URL, "opponent": "Jane Pundit", "me": "Me"})]
            return "", [("finish", {"summary": "ok"})]
        session = brief.split("Live session: ")[1].split()[0]
        if "enroll_voice" in tools:  # identify episode
            voices = last_result(messages, "list_voices")
            if voices is None:
                return "", [("list_voices", {"session_id": session})]
            stranger = [v for v in voices["voices"] if not v["speaker"]][0]
            if last_result(messages, "ask_user") is None:
                return "", [("ask_user", {"question": "Is the unknown voice Jane Pundit?"})]
            if last_result(messages, "enroll_voice") is None:
                return "", [
                    (
                        "enroll_voice",
                        {
                            "session_id": session,
                            "name": "Jane",
                            "utterance_id": stranger["sample_lines"][0]["utterance_id"],
                            "research_subject": "Jane Pundit",
                        },
                    )
                ]
            if last_result(messages, "set_opponent") is None:
                return "", [("set_opponent", {"session_id": session, "opponent": "Jane Pundit", "me": "Me"})]
            return "", [("finish", {"summary": "enrolled Jane"})]
        return "", [("finish", {"summary": "nothing"})]  # reply episodes

    turns = [(JANE, 6), (ME, 3)] * 5
    outcome, _ = run(
        "monitor",
        cfg,
        {"opponent": "Jane Pundit", "me": "Me"},
        service_for(cfg, turns, delay=0.3),
        policy="llm",
        llm=ScriptedToolLLM(script),
        terminal=ScriptedTerminal(["yes"]),
    )
    r = outcome.report
    assert "identify" in [e["goal"] for e in r["stats"]["episodes"]]
    assert ProfileStore(cfg).find_profile("Jane") is not None
    assert r["points"], "no arguments detected after the opponent was enrolled"
    with ResearchStore(cfg.research_db) as store:
        assert store.get_person("Jane Pundit").voice_profile == "Jane"


def test_monitor_rules_identify_and_quit_command(cfg):
    ProfileStore(cfg).add_profile("Me", voice_vector(ME), speech_seconds=300)
    jane_record(cfg, voice=None)
    terminal = ScriptedTerminal(["Jane"])

    def echo(line):
        if "💬 [p1]" in line:  # first argument answered → user quits
            terminal.feed("r")
            terminal.feed("nonsense")
            terminal.feed("9")
            terminal.feed("q")

    turns = [(JANE, 6), (ME, 3)] * 8
    outcome, _ = run(
        "monitor",
        cfg,
        {"opponent": "Jane Pundit"},
        service_for(cfg, turns, delay=0.3),
        terminal=terminal,
        echo=echo,
    )
    assert outcome.ok and outcome.report["session_status"] == "stopped"
    assert ProfileStore(cfg).find_profile("Jane") is not None
    commands = [e["command"] for e in trace_of(outcome) if e["kind"] == "user_command"]
    assert commands == ["quit"]


# ── runner ───────────────────────────────────────────────────────


def test_runner_failure_still_writes_outputs(cfg):
    service = AgentService(cfg)
    outcome = run_agent(
        "monitor",
        cfg,
        {"source": "/no/such/file.wav"},
        policy="rules",
        terminal=ScriptedTerminal([]),
        service=service,
        echo=lambda _m: None,
    )
    service.close()
    assert not outcome.ok and "Couldn't start the live session" in outcome.error
    assert outcome.run_dir.report_md.read_text().startswith("> ⚠ Run ended with an error")
    report = json.loads(outcome.run_dir.report_json.read_text())
    assert report["ok"] is False and report["outputs"]["trace"].endswith("trace.jsonl")
    assert trace_of(outcome)[-1]["kind"] == "run_end"
    assert json.loads(outcome.run_dir.config.read_text())["agent"]["poll_wait_s"] == 0.5


def test_runner_validates_inputs(cfg, monkeypatch):
    with pytest.raises(ValueError, match="policy"):
        run_agent("monitor", cfg, {"source": URL}, policy="magic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from kickvoice.research.llm import LLMError

    with pytest.raises(LLMError):
        run_agent("monitor", cfg, {"source": URL}, policy="llm")
    service = AgentService(cfg)
    outcome = run_agent(
        "dance",
        cfg,
        {"source": URL},
        policy="rules",
        service=service,
        terminal=ScriptedTerminal([]),
        echo=lambda _m: None,
    )
    service.close()
    assert not outcome.ok and "No agent for command" in outcome.error
