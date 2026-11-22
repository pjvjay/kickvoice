"""The MCP layer: tool surface, research tools, errors the agent can read, and real stdio."""

import json
import sys
import time

import anyio
import pytest

from conftest import ScriptedLLM, conversation, fake_pipeline_factory, voice_vector
from kickvoice.agent.server import build_server
from kickvoice.agent.service import AgentService, ToolError
from kickvoice.research.net import Fetcher
from kickvoice.research.store import ResearchStore
from kickvoice.speakers.profiles import ProfileStore
from test_research_sources import x_routes

EXPECTED_TOOLS = {
    "start_live_session",
    "get_live_updates",
    "list_voices",
    "enroll_voice",
    "set_opponent",
    "stop_live_session",
    "list_live_sessions",
    "research_subjects",
    "research_profile",
    "research_search",
    "research_find_identity",
    "research_link_wikidata",
    "research_add_subject",
    "research_collect",
    "research_analyze",
    "job_status",
    "get_point_evidence",
    "suggest_counters",
    "verify_message",
    "format_for_chat",
    "copy_to_clipboard",
}


def human(label, desc, accounts, qid_type="Q5"):
    claims = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": qid_type}}}}]}
    for prop, value in accounts.items():
        claims[prop] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": value}}}]
    return {
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": desc}},
        "claims": claims,
        "sitelinks": {},
    }


@pytest.fixture
def wikidata(fake_web):
    entities = {
        "Q2": human("Jane Pundit", "political streamer", {"P2002": "JanePundit"}),
        "Q1": human("Pundit", "2014 video game", {"P2002": "PunditGame"}, "Q7889"),
    }

    def api(req):
        q = req["query"]
        if q.get("action") == "wbsearchentities":
            return 200, {}, {"search": [{"id": "Q1"}, {"id": "Q2"}]}
        if q.get("list") == "search":
            hit = q["srsearch"] == "haswbstatement:P2002=JanePundit"
            return 200, {}, {"query": {"search": [{"title": "Q2"}] if hit else []}}
        return 200, {}, {"entities": {i: entities.get(i, {"missing": ""}) for i in q["ids"].split("|")}}

    fake_web.route("GET", "/wd/api.php", api)


@pytest.fixture
def service(rconfig):
    s = AgentService(rconfig, fetcher_factory=lambda c: Fetcher(c), clipboard=lambda t: (True, "fake"))
    yield s
    s.close()


# ── research tools (service level) ───────────────────────────────


def test_identity_tools_never_link_by_themselves(service, rconfig, wikidata):
    found = service.research_find_identity("Pundit")
    assert [c["wikidata"] for c in found["candidates"]] == ["Q2"]
    assert found["skipped_not_people"][0]["description"] == "2014 video game"
    assert "NOT proof of identity" in found["note"]
    assert service.research_subjects() == {"subjects": []}  # nothing linked

    exact = service.research_find_identity("x", handles={"x": "JanePundit"})
    assert exact["candidates"][0]["strong_match"] and exact["note"] == ""
    with pytest.raises(ToolError, match="Unknown platform"):
        service.research_find_identity("x", handles={"discord": "someone"})

    with pytest.raises(ToolError, match="not a person"):
        service.research_link_wikidata("Pundit", "Q1", confirm_accounts=True)
    linked = service.research_link_wikidata("Jane Pundit", "q2")
    assert [a["status"] for a in linked["accounts"]] == ["candidate"]  # not confirmed by default
    linked = service.research_link_wikidata("Jane Pundit", "Q2", confirm_accounts=True)
    assert [a["status"] for a in linked["accounts"]] == ["confirmed"]


def test_collect_and_analyze_jobs(service, rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(
        fake_web,
        {
            None: {
                "data": [
                    {
                        "id": "1",
                        "text": "Rent control reduces housing supply and everybody knows it",
                        "created_at": "2021-03-01T00:00:00Z",
                    }
                ],
                "meta": {},
            }
        },
    )
    profile = service.research_add_subject("Jane Pundit", "streamer", accounts={"x": "JanePundit"})
    assert profile["accounts"][0]["status"] == "confirmed"

    job = service.research_collect("Jane Pundit")
    done = service.job_status(job["job_id"], wait_seconds=10)
    assert done["status"] == "done", done
    assert done["result"]["by_source"]["x"]["new"] == 1 and done["result"]["unanalysed_excerpts"] == 1

    hits = service.research_search("Jane Pundit", "rent control")["results"]
    assert hits[0]["type"] == "excerpt" and hits[0]["first_hand"]  # searchable before analysis

    llm = ScriptedLLM(
        [
            {
                "claims": [
                    {
                        "chunk": "K1",
                        "topic": "housing",
                        "claim": "RC cuts supply",
                        "quote": "Rent control reduces housing supply",
                        "kind": "position",
                    }
                ]
            }
        ]
    )
    service._llm_factory = lambda c: llm
    job = service.research_analyze("Jane Pundit")
    assert job["excerpts"] == 1 and job["estimated_tokens"]["input"] > 0
    done = service.job_status(job["job_id"], wait_seconds=10)
    assert done["status"] == "done" and done["result"]["claims"] == 1
    assert service.research_analyze("Jane Pundit") == {"status": "nothing to analyse"}
    assert service.research_profile("Jane Pundit")["top_topics"] == [{"topic": "housing", "claims": 1}]


def test_job_errors_are_reported(service):
    with pytest.raises(ToolError, match="No research subject"):
        service.research_collect("Nobody")
    with pytest.raises(ToolError, match="No job"):
        service.job_status("nope")
    job = service.jobs.submit("x", "boom", lambda progress: (_ for _ in ()).throw(RuntimeError("kaput")))
    assert service.job_status(job.id, wait_seconds=5)["error"] == "kaput"


def test_verify_message_via_service_uses_cited_and_searched_evidence(service, rconfig):
    from kickvoice.research.store import Chunk, Claim, Document

    with ResearchStore(rconfig.research_db) as store:
        p = store.add_person("Jane Pundit")
        d = Document(
            p.id,
            "x",
            "post",
            "1",
            "Honestly the minimum wage should track inflation.",
            "self",
            published_at="2023-05-05",
        )
        store.upsert_document(d, [Chunk(d.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        (cid,) = store.add_claims(
            [
                Claim(
                    p.id,
                    d.id,
                    chunk.id,
                    "wages",
                    "tie wage to CPI",
                    "the minimum wage should track inflation",
                    published_at="2023-05-05",
                )
            ]
        )
    ok = service.verify_message(
        "Jane Pundit", 'in 2023 you said "the minimum wage should track inflation"', cited_refs=[f"C{cid}"]
    )
    assert ok["ok"], ok
    uncited = service.verify_message("Jane Pundit", 'you said "the minimum wage should track inflation"')
    assert uncited["ok"]  # found by searching the record
    single = service.verify_message("Jane Pundit", "you said 'the minimum wage should track inflation'")
    assert single["ok"] and single["verified_quotes"]  # any quote marks: searched and checked the same way
    made_up = service.verify_message("Jane Pundit", "you said ‘the minimum wage should be abolished’")
    assert not made_up["ok"] and "not found" in made_up["issues"][0]
    wrong = service.verify_message("Jane Pundit", 'you said "abolish the minimum wage"')
    assert not wrong["ok"]
    assert service.copy_to_clipboard("hi") == {"copied": True, "detail": "fake", "length": 2}
    assert service.format_for_chat("**Reply:** hello\nthere")["text"] == "hello there"


def test_suggest_counters_without_api_key_explains(rconfig, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with ResearchStore(rconfig.research_db) as store:
        store.add_person("Jane Pundit")
    service = AgentService(rconfig)
    with pytest.raises(ToolError, match="get_point_evidence"):
        service.suggest_counters(text="rent control is good", opponent="Jane Pundit")
    service.close()


# ── MCP protocol (in-process) ────────────────────────────────────


def call(client, name, args=None):
    async def go():
        result = await client.call_tool(name, args or {})
        text = result.content[0].text if result.content else ""
        return result.is_error, (text if result.is_error else json.loads(text))

    return go()


def test_mcp_tool_surface_and_errors(service):
    from mcp import Client

    async def main():
        async with Client(build_server(service)) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) >= EXPECTED_TOOLS
            assert all(t.description for t in tools.values())
            schema = tools["enroll_voice"].input_schema["properties"]["utterance_id"]
            assert "sample" in schema["description"]
            assert client.instructions and "verify_message" in client.instructions

            is_error, message = await call(client, "get_live_updates", {"session_id": "nope"})
            assert is_error and "No live session 'nope'" in message  # readable, not generic
            is_error, message = await call(client, "research_profile", {"name": "Nobody"})
            assert is_error and "research_add_subject" in message

            prompt = await client.get_prompt(
                "live_chat_copilot", {"stream_url": "https://kick.com/x", "opponent": "Jane"}
            )
            assert "copy_to_clipboard" in prompt.messages[0].content.text

    anyio.run(main)


def test_mcp_live_loop(rconfig):
    """An agent-style loop over MCP: start, poll, get a point, fetch evidence, copy a reply."""
    from mcp import Client

    rconfig.debate.chunk_duration_s = 5.0
    rconfig.debate.min_point_words = 5
    ProfileStore(rconfig).add_profile("Jane", voice_vector(220))
    ProfileStore(rconfig).add_profile("Me", voice_vector(550))
    with ResearchStore(rconfig.research_db) as store:
        store.add_person("Jane Pundit", voice_profile="Jane")
    copied = []
    service = AgentService(
        rconfig,
        pipeline_factory=fake_pipeline_factory(conversation([(220, 5), (550, 4)]), delay=0.05),
        clipboard=lambda t: (copied.append(t) or True, "fake"),
    )

    async def main():
        async with Client(build_server(service)) as client:
            _, started = await call(
                client, "start_live_session", {"source": "https://kick.com/x", "opponent": "Jane Pundit"}
            )
            sid, cursor, points = started["session_id"], 0, []
            deadline = time.monotonic() + 20
            while not points and time.monotonic() < deadline:
                _, r = await call(
                    client, "get_live_updates", {"session_id": sid, "cursor": cursor, "wait_seconds": 2}
                )
                cursor = r["next_cursor"]
                points += [e for e in r["events"] if e["type"] == "point"]
            assert points, "no point events"
            _, ev = await call(
                client, "get_point_evidence", {"session_id": sid, "point_id": points[0]["point_id"]}
            )
            assert ev["point"].startswith("w220") and ev["opponent"] == "Jane Pundit"
            _, result = await call(client, "copy_to_clipboard", {"text": "ok but who pays for it?"})
            assert result["copied"]
            await call(client, "stop_live_session", {"session_id": sid})

    anyio.run(main)
    service.close()
    assert copied == ["ok but who pays for it?"]


# ── real stdio: launched the way Claude Desktop / Claude Code do ──


def test_mcp_over_real_stdio(rconfig, tmp_path):
    from mcp import Client, StdioServerParameters

    cfg_path = tmp_path / "config.json"
    rconfig.save(cfg_path)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "kickvoice.cli", "--config", str(cfg_path), "mcp"]
    )

    async def main():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names >= EXPECTED_TOOLS
            is_error, added = await call(
                client, "research_add_subject", {"name": "Jane Pundit", "accounts": {"x": "JanePundit"}}
            )
            assert not is_error and added["accounts"][0]["handle"] == "JanePundit"
            _, subjects = await call(client, "research_subjects")
            assert subjects["subjects"][0]["name"] == "Jane Pundit"

    anyio.run(main)


def test_stray_stdout_output_cannot_corrupt_the_protocol(rconfig, tmp_path):
    """
    Pipelines print progress with click.echo, and ffmpeg/yt-dlp are child
    processes. Under `kickvoice mcp`, anything written to stdout by a tool
    must end up on stderr, not in the JSON-RPC stream.
    """
    from mcp import Client, StdioServerParameters

    cfg_path = tmp_path / "config.json"
    rconfig.save(cfg_path)
    launcher = tmp_path / "noisy_server.py"
    launcher.write_text(
        "import subprocess, sys, click\n"
        "import kickvoice.agent.service as svc\n"
        "original = svc.AgentService.research_subjects\n"
        "def noisy(self):\n"
        "    click.echo('progress: collecting...')          # Python-level stdout\n"
        "    print('more stray output', flush=True)\n"
        "    subprocess.run(['echo', 'child process output'])  # inherits fd 1\n"
        "    return original(self)\n"
        "svc.AgentService.research_subjects = noisy\n"
        "from kickvoice.cli import main\n"
        "sys.argv = ['kickvoice', '--config', sys.argv[1], 'mcp']\n"
        "main()\n"
    )
    params = StdioServerParameters(command=sys.executable, args=[str(launcher), str(cfg_path)])

    async def main():
        async with Client(params) as client:
            for _ in range(3):
                is_error, subjects = await call(client, "research_subjects")
                assert not is_error and subjects == {"subjects": []}

    anyio.run(main)
