"""Agent runtime: traces, tool-calling model adapters, the MCP toolbox, episodes, the terminal."""

import json
import time

import anyio
import pytest

from conftest import ScriptedToolLLM, last_result
from kickvoice.agent.episode import EpisodeLimits, TokenBudget, run_episode
from kickvoice.agent.service import AgentService
from kickvoice.agent.terminal import ScriptedTerminal, Terminal
from kickvoice.agent.toolbox import LocalTool, LocalToolError, ToolBox, connect_mcp, schema
from kickvoice.agent.toolcalling import (
    AnthropicToolLLM,
    AssistantTurn,
    OpenAIToolLLM,
    ToolCall,
    ToolSpec,
    make_tool_llm,
)
from kickvoice.agent.trace import RunDir, Tracer, read_trace
from kickvoice.research.llm import LLMError

# ── traces ───────────────────────────────────────────────────────


def test_run_dirs_are_unique_and_named(tmp_path):
    a = RunDir.create(tmp_path, "monitor", "https://kick.com/SomeChannel")
    b = RunDir.create(tmp_path, "monitor", "https://kick.com/SomeChannel")
    assert a.path != b.path and a.path.name.endswith("_monitor_somechannel")
    assert a.trace.name == "trace.jsonl" and a.clips.name == "clips"


def test_tracer_writes_jsonl_counts_and_clips(tmp_path):
    echoed = []
    t = Tracer(tmp_path / "trace.jsonl", echo=echoed.append)
    t.event("episode_start", goal="x")
    t.event("llm_response", input_tokens=100, output_tokens=20)
    t.event("tool_result", tool="list_voices", is_error=False)
    t.event("tool_result", tool="list_voices", is_error=True)
    t.event("tool_result", tool="x", result="y" * 10_000)
    t.say("hello")
    t.close()
    t.close()  # idempotent
    events = read_trace(tmp_path / "trace.jsonl")
    assert [e["kind"] for e in events][-1] == "note" and echoed == ["hello"]
    assert events[4]["result"].endswith("[10000 chars]") and len(events[4]["result"]) < 4100
    s = t.stats.to_dict()
    assert (s["llm_calls"], s["input_tokens"], s["tool_calls"], s["tool_errors"], s["episodes"]) == (
        1,
        100,
        3,
        1,
        1,
    )
    assert list(s["by_tool"]) == ["list_voices", "x"]


# ── tool-calling model adapters (real HTTP to a fake server) ─────

TOOLS = [
    ToolSpec(
        "list_voices", "List voices", {"type": "object", "properties": {"session_id": {"type": "string"}}}
    )
]
CONVERSATION = [
    {"role": "user", "text": "Who is speaking?"},
    {
        "role": "assistant",
        "turn": AssistantTurn(
            text="Checking.",
            tool_calls=[
                ToolCall("t1", "list_voices", {"session_id": "s"}),
                ToolCall("t2", "list_voices", {"session_id": "z"}),
            ],
        ),
    },
    {"role": "tool", "call_id": "t1", "name": "list_voices", "content": '{"voices": []}', "is_error": False},
    {"role": "tool", "call_id": "t2", "name": "list_voices", "content": "Error: no", "is_error": True},
]


def test_anthropic_tool_use_roundtrip(rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    rconfig.llm.base_url = fake_web.url
    fake_web.route(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-x",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 9},
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "toolu_1", "name": "list_voices", "input": {"session_id": "s"}},
            ],
        },
    )
    llm = make_tool_llm(rconfig)
    assert isinstance(llm, AnthropicToolLLM)
    turn = llm.respond("sys", CONVERSATION, TOOLS, "claude-x", max_tokens=300)
    assert turn.text == "Let me look." and turn.tool_calls[0].arguments == {"session_id": "s"}
    assert (turn.input_tokens, turn.output_tokens, turn.stop_reason) == (50, 9, "tool_use")
    body = json.loads(fake_web.hits("/v1/messages")[0]["body"])
    assert body["tools"][0]["input_schema"]["properties"]["session_id"]["type"] == "string"
    assert body["system"] == "sys" and body["max_tokens"] == 300
    wire = body["messages"]
    assert wire[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "list_voices",
        "input": {"session_id": "s"},
    }
    # both results in ONE user message, errors flagged
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["t1", "t2"]
    assert wire[2]["content"][1]["is_error"] is True


def test_openai_function_calling_roundtrip(rconfig, fake_web, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rconfig.llm.provider = "openai"
    rconfig.llm.base_url = f"{fake_web.url}/v1"
    fake_web.route(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "usage": {"prompt_tokens": 40, "completion_tokens": 7},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "list_voices", "arguments": '{"session_id": "s"}'},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "list_voices", "arguments": "{not json"},
                            },
                            {
                                "id": "c3",
                                "type": "function",
                                "function": {"name": "list_voices", "arguments": "[1]"},
                            },
                        ],
                    },
                }
            ],
        },
    )
    llm = make_tool_llm(rconfig)
    assert isinstance(llm, OpenAIToolLLM)
    turn = llm.respond("sys", CONVERSATION, TOOLS, "qwen")
    assert turn.tool_calls[0].arguments == {"session_id": "s"} and turn.tool_calls[0].parse_error is None
    assert "valid JSON" in turn.tool_calls[1].parse_error
    assert "JSON object" in turn.tool_calls[2].parse_error
    assert (turn.input_tokens, turn.stop_reason) == (40, "tool_calls")
    body = json.loads(fake_web.hits("/v1/chat/completions")[0]["body"])
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["tools"][0]["function"]["parameters"] == TOOLS[0].input_schema
    assistant = body["messages"][2]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"session_id": "s"}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": '{"voices": []}'}


def test_openai_bad_response_and_provider_errors(rconfig, fake_web, monkeypatch):
    rconfig.llm.provider = "openai"
    rconfig.llm.base_url = f"{fake_web.url}/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    fake_web.route("POST", "/v1/chat/completions", json={"nope": True})
    with pytest.raises(LLMError, match="Unexpected response shape"):
        make_tool_llm(rconfig).respond("s", [{"role": "user", "text": "x"}], TOOLS, "m")
    assert fake_web.hits("/v1/chat/completions")[0]["headers"]["authorization"] == "Bearer k"
    rconfig.llm.provider = "mystery"
    with pytest.raises(LLMError, match="Unknown"):
        make_tool_llm(rconfig)
    rconfig.llm.provider = "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_tool_llm(rconfig)


# ── toolbox over a real (in-process) MCP connection ─────────────


def with_toolbox(config, body, tmp_path, local=()):
    service = AgentService(config, clipboard=lambda t: (True, "fake"))
    tracer = Tracer(tmp_path / "trace.jsonl")

    async def main():
        async with connect_mcp(tmp_path / "cfg.json", tmp_path / "server.log", service) as client:
            box = ToolBox(client, tracer, local)
            await box.load()
            await body(box)

    anyio.run(main)
    service.close()
    tracer.close()
    return read_trace(tmp_path / "trace.jsonl")


def test_toolbox_mcp_and_local_tools(config, tmp_path):
    seen = []

    async def echo(args):
        if args.get("fail"):
            raise LocalToolError("refused")
        return {"echo": args["x"]}

    local = [LocalTool("echo", "Echo", schema({"x": {"type": "string"}}, ["x"]), echo)]

    async def body(box):
        box.observers.append(lambda name, args, result: seen.append((name, result.ok)))
        assert "list_live_sessions" in box.mcp and "echo" in box.local
        assert [s.name for s in box.specs(["echo", "research_subjects"])] == ["echo", "research_subjects"]
        with pytest.raises(KeyError):
            box.specs(["nope"])
        ok = await box.call("research_subjects", {}, summarize=lambda d: {"n": len(d["subjects"])})
        assert ok.ok and ok.data == {"subjects": []}
        err = await box.call("research_profile", {"name": "Nobody"})
        assert not err.ok and "No research subject" in err.text
        assert (await box.call("echo", {"x": "hi"})).data == {"echo": "hi"}
        refused = await box.call("echo", {"x": "hi", "fail": True})
        assert not refused.ok and refused.text == "Error: refused"
        assert not (await box.call("not_a_tool", {})).ok

    events = with_toolbox(config, body, tmp_path, local)
    assert seen[:2] == [("research_subjects", True), ("research_profile", False)]
    results = [e for e in events if e["kind"] == "tool_result"]
    assert results[0]["result"] == {"n": 0}  # summarised in the trace
    assert events[0]["kind"] == "tools_loaded"


# ── episodes ─────────────────────────────────────────────────────


def run_with(config, tmp_path, script, tools, limits=None, budget=None, local=(), slow=0.0):
    outcome = {}
    llm = ScriptedToolLLM(script)
    if slow:
        original = llm.respond
        llm.respond = lambda *a, **k: (time.sleep(slow), original(*a, **k))[1]

    async def body(box):
        outcome["result"] = await run_episode(
            goal="test",
            llm=llm,
            model="m",
            toolbox=box,
            tracer=box.tracer,
            system="sys",
            brief="Do the thing",
            tools=tools,
            limits=limits,
            budget=budget,
        )

    events = with_toolbox(config, body, tmp_path, local)
    return outcome["result"], llm, events


async def _finish(args):
    return {"ok": True}


FINISH = [LocalTool("finish", "Finish", schema({"summary": {"type": "string"}}), _finish)]


def test_episode_tool_loop_until_model_stops(config, tmp_path):
    def script(system, messages, tools):
        if last_result(messages, "research_subjects") is None:
            return "", [("research_subjects", {})]
        return "There are no subjects yet.", []

    result, llm, events = run_with(config, tmp_path, script, ["research_subjects"], local=FINISH)
    assert (result.outcome, result.steps, result.tool_calls) == ("done", 2, ["research_subjects"])
    assert result.final_text == "There are no subjects yet."
    assert "finish" in llm.calls[0]["tools"]  # finish offered automatically
    kinds = [e["kind"] for e in events]
    assert kinds.count("llm_request") == 2 and kinds[-1] == "episode_end"


def test_episode_ends_on_finish_and_rejects_unavailable_tools(config, tmp_path):
    def script(system, messages, tools):
        if len(messages) == 1:
            return "", [("enroll_voice", {"session_id": "x"})]  # not allowed for this goal
        return "", [("finish", {"summary": "done"})]

    result, llm, events = run_with(config, tmp_path, script, ["research_subjects"], local=FINISH)
    assert result.outcome == "finished"
    error = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert error["is_error"] and "isn't available" in error["content"]


@pytest.mark.parametrize("case", ["max_steps", "budget", "timeout", "error"])
def test_episode_limits_and_failures(config, tmp_path, case):
    def loop_forever(system, messages, tools):
        if case == "error":
            raise RuntimeError("API down")
        return "", [("research_subjects", {})]

    limits = EpisodeLimits(max_steps=3, timeout_s=0.5 if case == "timeout" else 60)
    budget = TokenBudget(200) if case == "budget" else None
    result, _, events = run_with(
        config,
        tmp_path,
        loop_forever,
        ["research_subjects"],
        limits,
        budget,
        slow=1.0 if case == "timeout" else 0.0,
    )
    assert result.outcome == case
    if case == "budget":
        assert result.steps == 2 and budget.exhausted  # 150 tokens per call, cap 200
    if case == "error":
        assert result.error == "API down" and any(e["kind"] == "llm_error" for e in events)


def test_malformed_tool_arguments_go_back_to_the_model(config, tmp_path):
    class BadArgsLLM(ScriptedToolLLM):
        def respond(self, system, messages, tools, model, max_tokens=1500):
            self.calls.append({"messages": list(messages)})
            if len(messages) == 1:
                return AssistantTurn(
                    tool_calls=[
                        ToolCall("x", "research_subjects", {}, parse_error="arguments weren't valid JSON")
                    ]
                )
            return AssistantTurn(text="ok")

    llm, outcome = BadArgsLLM(None), {}

    async def body(box):
        outcome["r"] = await run_episode(
            goal="t",
            llm=llm,
            model="m",
            toolbox=box,
            tracer=box.tracer,
            system="s",
            brief="b",
            tools=["research_subjects"],
        )

    events = with_toolbox(config, body, tmp_path)
    assert outcome["r"].outcome == "done"
    sent_back = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert sent_back["is_error"] and "valid JSON" in sent_back["content"]
    assert not [e for e in events if e["kind"] == "tool_call"]  # never executed


def test_long_tool_results_are_truncated_for_the_model(config, tmp_path):
    async def big(args):
        return {"data": "x" * 20_000}

    local = [LocalTool("big", "Big", schema({}), big)]

    def script(system, messages, tools):
        return ("", [("big", {})]) if len(messages) == 1 else ("ok", [])

    result, llm, _ = run_with(config, tmp_path, script, ["big"], local=local)
    content = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert content.endswith("…[truncated]") and len(content) < 6100


def test_token_budget():
    b = TokenBudget(100)
    b.spend(60)
    assert not b.exhausted
    b.spend(40)
    assert b.exhausted
    assert not TokenBudget(0).exhausted  # 0 = no cap


# ── terminal ─────────────────────────────────────────────────────


def test_terminal_routes_answers_and_commands():
    async def main():
        t = Terminal(interactive=True, echo=lambda _m: None)
        t._reader_running = True  # as if read_stdin() were running; lines arrive via feed()
        got = {}

        async def asker():
            got["answer"] = await t.ask("Who?", ["a", "b"])

        async with anyio.create_task_group() as tg:
            tg.start_soon(asker)
            await anyio.sleep(0.05)
            t.feed("  Jane  ")  # answers the pending question
            await anyio.sleep(0.05)
            t.feed("2")  # no question pending → a command
            async for command in t.commands():
                got["command"] = command
                break
        return got

    assert anyio.run(main) == {"answer": "Jane", "command": "2"}


def test_non_interactive_terminal():
    async def main():
        t = Terminal(interactive=False)
        assert await t.ask("Who?") is None
        await t.read_stdin()  # returns immediately
        s = ScriptedTerminal(["x"], interactive=False)
        assert await s.ask("Q") is None and s.questions == ["Q"]
        s2 = ScriptedTerminal(["first"])
        assert await s2.ask("Q1") == "first" and await s2.ask("Q2") is None

    anyio.run(main)
