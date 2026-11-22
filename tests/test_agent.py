"""
The MCP agent layer: live session, chat checks, and the full agentic flow
driven through a real MCP client (in-process, and over stdio).
"""
import json
import sys
import time

import anyio
import pytest

from conftest import (
    FakeEncoder, FakeRecognizer, PacedCapture, conversation, voice_vector,
)
from kickvoice.agent import server as agent_server
from kickvoice.agent.chat import check_message, clean_message, copy_to_clipboard
from kickvoice.agent.jobs import JobRunner
from kickvoice.agent.live import LiveSession
from kickvoice.agent.server import AgentState, build_server
from kickvoice.audio.vad import EnergyVAD
from kickvoice.pipelines.transcribe import TranscriptionPipeline
from kickvoice.research.store import SELF, Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore

HOST, GUEST = 220, 550


def factory_for(capture):
    """Pipeline factory: real TranscriptionPipeline, fake models, given capture."""
    def make(source, config, on_utterance):
        return TranscriptionPipeline(
            stream_url=source, config=config, capture=capture, vad=EnergyVAD(config.audio),
            encoder=FakeEncoder(), recognizer=FakeRecognizer(), on_utterance=on_utterance,
            out=lambda _: None)
    return make


@pytest.fixture
def live_config(config):
    config.debate.min_point_words = 5
    config.debate.point_pause_s = 0.2
    config.debate.chunk_duration_s = 5.0
    return config


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ── chat checks ──────────────────────────────────────────────────

EVIDENCE = ["Rent control reduces housing supply, build more homes instead."]


def test_clean_message_flattens_markdown():
    assert clean_message("**Counter:** see [this](https://a.b/c)\n\n- point") == \
        "Counter: see this https://a.b/c point"
    assert clean_message("`quoted`") == "quoted"


@pytest.mark.parametrize("text,ok,issue", [
    ('you tweeted "rent control reduces housing supply" lol', True, None),
    ('you tweeted "Rent control reduces the housing supply" lol', True, None),   # small drift ok
    ('you said "rent control destroys every single city" in 2021', False, "not found"),
    ('that\'s just "vibes" bro', True, None),                                   # scare quotes
    ("x" * 501, False, "Too long"),
    ("   ", False, "empty"),
])
def test_check_message(text, ok, issue):
    check = check_message(text, EVIDENCE, max_chars=500)
    assert check.ok is ok
    if issue:
        assert any(issue in i for i in check.issues)


def test_check_message_quote_without_evidence_and_links():
    check = check_message('he literally said "the earth is flat and round"', [])
    assert not check.ok and "no evidence cited" in check.issues[0]
    assert check_message("see https://x.com/a", []).warnings
    assert not check_message("see https://x.com/a", [], allow_links=True).warnings


def test_copy_to_clipboard(monkeypatch):
    import kickvoice.agent.chat as chat

    calls = []
    monkeypatch.setattr(chat.sys, "platform", "linux")
    monkeypatch.setattr(chat.shutil, "which", lambda c: "/usr/bin/xclip" if c == "xclip" else None)
    monkeypatch.setattr(chat.subprocess, "run",
                        lambda cmd, **kw: calls.append((cmd, kw["input"])))
    assert copy_to_clipboard("héllo") == (True, "xclip")
    assert calls == [(["xclip", "-selection", "clipboard"], "héllo".encode())]
    monkeypatch.setattr(chat.shutil, "which", lambda c: None)
    ok, reason = copy_to_clipboard("x")
    assert not ok and "no clipboard tool" in reason


def test_job_runner():
    jobs = JobRunner()
    good = jobs.submit("t", "ok", lambda progress: (progress("step 1"), 42)[1])
    bad = jobs.submit("t", "bad", lambda progress: 1 / 0)
    assert wait_for(lambda: good.status == "done" and bad.status == "failed")
    assert good.result == 42 and good.progress == ["step 1"]
    assert "division" in bad.error
    assert {j.id for j in jobs.all()} == {good.id, bad.id}


# ── live session ─────────────────────────────────────────────────

def test_live_session_points_per_speaker(live_config):
    profiles = ProfileStore(live_config)
    profiles.add_profile("Host", voice_vector(HOST), speech_seconds=300)
    capture = PacedCapture(conversation([(HOST, 4), (GUEST, 4), (HOST, 4)]))
    live = LiveSession(live_config, "https://kick.com/x", pipeline_factory=factory_for(capture))
    live.start()
    capture.release(10)
    first = live.next_point(timeout_s=10)
    second = live.next_point(timeout_s=10)
    third = live.next_point(timeout_s=10)
    assert (first.speaker, first.label) == ("Host", "Host") and "w220" in first.text
    assert second.speaker is None and "w550" in second.text and "w220" not in second.text
    assert third.speaker == "Host"
    assert live.next_point(timeout_s=0.5) is None                      # nothing left
    lines, cursor = live.lines_since(0)
    assert cursor == len(lines) >= 3
    assert live.status()["state"] in ("running", "finished")
    live.stop()
    assert live.status()["state"] == "finished"


def test_live_session_only_tracked_speakers(live_config):
    profiles = ProfileStore(live_config)
    profiles.add_profile("Host", voice_vector(HOST))
    capture = PacedCapture(conversation([(HOST, 4), (GUEST, 4), (HOST, 4)]))
    live = LiveSession(live_config, "https://kick.com/x", speakers=["host"],
                       pipeline_factory=factory_for(capture))
    live.start()
    capture.release(10)
    points = [live.next_point(timeout_s=10) for _ in range(2)]
    assert [p.speaker for p in points] == ["Host", "Host"]
    live.stop()


def test_live_session_surfaces_pipeline_failure(live_config):
    def broken(source, config, on_utterance):
        raise RuntimeError("streamlink failed: channel is offline")

    live = LiveSession(live_config, "https://kick.com/offline", pipeline_factory=broken)
    live.start()
    assert live.next_point(timeout_s=5) is None
    status = live.status()
    assert status["state"] == "failed" and "offline" in status["error"]


# ── the full agentic flow, through an MCP client ─────────────────

@pytest.fixture
def researched_host(live_config):
    """Host: enrolled voice + research record with one verified claim."""
    ProfileStore(live_config).add_profile("Host", voice_vector(HOST), speech_seconds=600)
    with ResearchStore(live_config.research_db) as store:
        person = store.add_person("Host Person", voice_profile="Host", description="streamer")
        doc = Document(person.id, "x", "post", "1", "w220 w220 w220 is the only policy that works",
                       SELF, url="https://x.com/host/status/1", published_at="2022-05-01")
        store.upsert_document(doc, [Chunk(doc.text)])
        chunk = store.chunks_to_analyze(person.id)[0][0]
        (claim_id,) = store.add_claims([Claim(
            person.id, doc.id, chunk.id, "w220 policy", "Says w220 is the only policy that works",
            "w220 w220 w220 is the only policy that works", published_at="2022-05-01",
            url=doc.url)])
    return claim_id


def call(client, name, args=None):
    async def go():
        result = await client.call_tool(name, args or {})
        payload = json.loads(result.content[0].text) if not result.is_error else None
        return result.is_error, payload, result.content[0].text
    return go()


def test_agentic_flow_over_mcp(live_config, researched_host, monkeypatch):
    """
    What an MCP host does during a stream: listen, get a point with evidence,
    draft + check a reply, copy it; meet an unknown voice, save it mid-stream,
    and see it recognised in later lines.
    """
    copied = []
    monkeypatch.setattr(agent_server, "copy_to_clipboard", lambda t: (copied.append(t) or True, "test"))

    # Host, Guest, Host, Guest, Guest  — 4s turns, 5s blocks
    capture = PacedCapture(conversation([(HOST, 4), (GUEST, 4), (HOST, 4), (GUEST, 4),
                                         (HOST, 4), (GUEST, 4), (HOST, 4)]))
    state = AgentState(live_config, pipeline_factory=factory_for(capture))
    server = build_server(live_config, state)

    async def flow():
        from mcp import Client

        async with Client(server) as client:
            err, started, _ = await call(client, "live_start", {"source": "https://kick.com/debate"})
            assert not err
            assert {"voice": "Host", "research_subject": "Host Person"} in started["recognisable_voices"]

            capture.release(3)                          # ~15s of stream
            err, got, raw = await call(client, "live_next_point", {"timeout_s": 15})
            assert not err, raw
            point = got["point"]
            assert point["speaker"] == "Host" and point["attribution"] == "confident"
            assert point["end_offset_s"] > point["offset_s"]
            assert got["behind_live_s"] is not None and got["behind_live_s"] >= 0
            assert got["subject"]["name"] == "Host Person"
            ev = got["evidence"][0]
            assert ev["ref"] == f"C{researched_host}" and ev["first_hand"]
            assert ev["quote"] == "w220 w220 w220 is the only policy that works"
            assert ev["url"] == "https://x.com/host/status/1" and ev["date"] == "2022-05-01"

            # the agent drafts a reply quoting the evidence → verified
            draft = f'**lol** you literally posted "{ev["quote"]}" in 2022, what changed?'
            err, check, _ = await call(client, "check_chat_message", {
                "text": draft, "subject": "Host Person", "evidence_refs": [ev["ref"]]})
            assert check["ok"] and check["verified_quotes"] and not check["message"].startswith("*")
            # a misquote is caught before it reaches chat
            err, bad, _ = await call(client, "check_chat_message", {
                "text": 'you posted "w220 is a total scam and always fails"',
                "subject": "Host Person", "evidence_refs": [ev["ref"]]})
            assert not bad["ok"] and "not found" in bad["issues"][0]
            err, foreign, _ = await call(client, "check_chat_message", {
                "text": "hi", "subject": "Host Person", "evidence_refs": ["C999999"]})
            assert not foreign["ok"]

            err, clip, _ = await call(client, "copy_to_clipboard", {"text": check["message"]})
            assert clip["copied"] and copied == [check["message"]]

            # next point: the guest — not recognised
            err, got, _ = await call(client, "live_next_point", {"timeout_s": 15})
            assert got["point"]["speaker"] is None and "Unrecognised voice" in got["note"]

            err, voices, _ = await call(client, "live_voices")
            by_voice = {v["voice"]: v for v in voices["voices"]}
            assert by_voice["Host"]["research_subject"] == "Host Person"
            unknown = by_voice[None]
            assert "w550" in unknown["sample"]

            # user says "that's Guest" → save it; later lines use the name
            err, saved, raw = await call(client, "save_voice", {
                "voice_group": unknown["voice_group"], "name": "Guest"})
            assert not err, raw
            assert saved["saved"] == "Guest" and saved["warning"]      # little speech so far
            err, _, raw = await call(client, "save_voice", {
                "voice_group": by_voice["Host"]["voice_group"], "name": "Someone"})
            assert err and "already recognised as Host" in raw

            capture.release(20)                          # rest of the stream
            speakers = []
            for _ in range(6):
                err, got, _ = await call(client, "live_next_point", {"timeout_s": 10})
                if got["point"] is None:
                    break
                speakers.append(got["point"]["speaker"])
            assert "Guest" in speakers                    # recognised after saving

            err, status, _ = await call(client, "live_status")
            assert status["state"] in ("running", "finished") and status["lines"] >= 6
            assert status["lines_by_voice"].get("Guest") and "realtime_factor" in status
            err, stopped, _ = await call(client, "live_stop")
            assert stopped["state"] == "finished" and stopped["lines"] >= 6
            err, _, raw = await call(client, "live_start", {"source": "/no/such/file.wav"})
            assert err and "Not a URL" in raw

    anyio.run(flow)
    assert ProfileStore(live_config).get_profile("Guest").speech_seconds > 0


def test_research_tools_over_mcp(rconfig, fake_web, monkeypatch):
    """add → discover (links nothing) → link record → collect job → analyze needs a key."""
    from test_research_identity import WIKIDATA

    state = AgentState(rconfig)
    server = build_server(rconfig, state)

    def api(req):
        q = req["query"]
        if q.get("list") == "search":
            hit = q["srsearch"] == "haswbstatement:P2002=TheOmniLiberal"
            return 200, {}, {"query": {"search": [{"title": "Q2"}] if hit else []}}
        if q.get("action") == "wbsearchentities":
            return 200, {}, {"search": [{"id": "Q1"}, {"id": "Q2"}]}
        return 200, {}, {"entities": {i: WIKIDATA[i] for i in q["ids"].split("|")}}
    fake_web.route("GET", "/wd/api.php", api)
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    fake_web.route("GET", "/2/users/by/username/TheOmniLiberal", json={"data": {"id": "7"}})
    fake_web.route("GET", "/2/users/7/tweets",
                   json={"data": [{"id": "1", "text": "a post about policy"}], "meta": {}})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def flow():
        from mcp import Client

        async with Client(server) as client:
            err, found, _ = await call(client, "research_discover", {"name": "Destiny"})
            assert [c["wikidata_id"] for c in found["candidates"]] == ["Q2"]
            assert not found["candidates"][0]["strong_match"]
            assert found["skipped_not_people"][0]["description"] == "2014 video game"
            err, subjects, _ = await call(client, "research_subjects")
            assert subjects["subjects"] == []                        # discover linked nothing

            err, _, raw = await call(client, "research_link_record",
                                     {"name": "Destiny", "wikidata_id": "Q1"})
            assert err and "not a person" in raw
            err, prof, _ = await call(client, "research_link_record",
                                      {"name": "Destiny", "wikidata_id": "Q2", "confirm_accounts": True})
            assert prof["wikidata"] == "Q2"
            assert {(a["platform"], a["status"]) for a in prof["accounts"]} >= {("x", "confirmed")}

            err, job, _ = await call(client, "research_collect", {"name": "Destiny", "sources": ["x"]})
            for _ in range(100):
                err, status, _ = await call(client, "job_status", {"job_id": job["job"]})
                if status["status"] != "running":
                    break
                await anyio.sleep(0.05)
            assert status["status"] == "done", status
            assert status["result"]["by_source"]["x"]["new"] == 1

            err, _, raw = await call(client, "research_analyze", {"name": "Destiny"})
            assert err and "ANTHROPIC_API_KEY" in raw
            err, _, raw = await call(client, "research_profile", {"name": "Nobody"})
            assert err and "Destiny" in raw                          # lists known subjects
    anyio.run(flow)


def test_copilot_prompt_uses_config_style(live_config):
    live_config.chat.style = "lowercase, dry, no emojis"
    live_config.chat.max_chars = 300
    text = agent_server.copilot_prompt(live_config, "https://kick.com/x", "Host, Guest")
    assert "lowercase, dry, no emojis" in text and "at most 300 characters" in text
    assert "speakers=['Host', 'Guest']" in text and "https://kick.com/x" in text
    assert "never say you did" in text


def test_stdio_server_keeps_stdout_clean(tmp_path):
    """Launch the real `kickvoice mcp` command and talk to it over stdio."""
    from mcp import Client, StdioServerParameters

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"data_dir": str(tmp_path / "data")}))

    async def flow():
        params = StdioServerParameters(command=sys.executable, args=[
            "-m", "kickvoice.cli", "--config", str(cfg), "mcp"])
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"live_start", "live_next_point", "check_chat_message",
                    "copy_to_clipboard", "research_search"} <= tools
            err, added, _ = await call(client, "research_add", {
                "name": "Jane", "accounts": {"x": "JanePundit"}})
            assert not err and added["accounts"][0]["status"] == "confirmed"
    anyio.run(flow)
