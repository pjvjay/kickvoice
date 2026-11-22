"""Polite HTTP (Fetcher) and the LLM clients, over real HTTP to a fake server."""
import time

import numpy as np
import pytest

from conftest import ScriptedLLM
from kickvoice.research.llm import (
    AnthropicClient, EmbeddingClient, InvalidOutput, LLMError, OpenAICompatibleClient,
    complete_json, make_llm, parse_json,
)
from kickvoice.research.net import Blocked, Fetcher, RateLimited


def test_fetcher_sends_user_agent_and_caches(rconfig, fake_web):
    fake_web.route("GET", "/data", json={"ok": 1})
    f = Fetcher(rconfig)
    assert f.get_json(f"{fake_web.url}/data", {"q": "x"}) == {"ok": 1}
    assert f.get(f"{fake_web.url}/data", {"q": "x"}).from_cache
    assert len(fake_web.hits("/data")) == 1
    assert fake_web.hits("/data")[0]["headers"]["user-agent"].startswith("kickvoice-research")
    f.get(f"{fake_web.url}/data", {"q": "x"}, cache=False)
    assert len(fake_web.hits("/data")) == 2


def test_fetcher_retries_then_succeeds(rconfig, fake_web):
    calls = []

    def flaky(req):
        calls.append(1)
        return (503, {"Retry-After": "0"}, {"e": 1}) if len(calls) < 3 else (200, {}, {"ok": True})

    fake_web.route("GET", "/flaky", flaky)
    assert Fetcher(rconfig).get_json(f"{fake_web.url}/flaky", cache=False) == {"ok": True}
    assert len(calls) == 3


def test_fetcher_gives_up_on_long_rate_limit(rconfig, fake_web):
    fake_web.route("GET", "/limited", status=429,
                   headers={"x-rate-limit-reset": str(int(time.time()) + 900)}, json={})
    with pytest.raises(RateLimited, match="15 min"):
        Fetcher(rconfig).get(f"{fake_web.url}/limited", cache=False)


def test_fetcher_throttles_per_host(rconfig, fake_web):
    rconfig.research.min_request_interval_s = 0.2
    fake_web.route("GET", "/t", json={})
    f = Fetcher(rconfig)
    started = time.monotonic()
    for _ in range(3):
        f.get(f"{fake_web.url}/t", cache=False)
    assert time.monotonic() - started >= 0.4


def test_robots_txt_respected(rconfig, fake_web):
    fake_web.route("GET", "/robots.txt", text="User-agent: *\nDisallow: /private\n")
    fake_web.route("GET", "/private/page", text="<p>secret</p>")
    fake_web.route("GET", "/public/page", text="<p>hello</p>")
    f = Fetcher(rconfig)
    with pytest.raises(Blocked):
        f.get(f"{fake_web.url}/private/page", respect_robots=True)
    assert f.get(f"{fake_web.url}/public/page", respect_robots=True).status == 200
    assert fake_web.hits("/private/page") == []


# ── LLM clients ──────────────────────────────────────────────────

def test_anthropic_client_request_and_parse(rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    rconfig.llm.base_url = fake_web.url
    fake_web.route("POST", "/v1/messages", json={
        "model": "claude-x", "content": [{"type": "text", "text": '{"a": 1}'}],
        "usage": {"input_tokens": 11, "output_tokens": 3}})
    reply = AnthropicClient(rconfig).complete("sys", "hi", "claude-x", max_tokens=50)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"a": 1}', 11, 3)
    req = fake_web.hits("/v1/messages")[0]
    assert req["headers"]["x-api-key"] == "sk-test"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    import json
    body = json.loads(req["body"])
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 50


def test_anthropic_requires_key(rconfig, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(rconfig).complete("s", "p", "m")


def test_openai_compatible_client_without_key(rconfig, fake_web, monkeypatch):
    """Local servers (Ollama) don't need a key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rconfig.llm.provider = "openai"
    rconfig.llm.base_url = f"{fake_web.url}/v1"
    fake_web.route("POST", "/v1/chat/completions", json={
        "model": "llama3", "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    client = make_llm(rconfig)
    assert isinstance(client, OpenAICompatibleClient)
    reply = client.complete("sys", "hi", "llama3")
    assert reply.text == "hello" and reply.input_tokens == 5
    req = fake_web.hits("/v1/chat/completions")[0]
    assert "authorization" not in req["headers"]
    import json
    assert json.loads(req["body"])["messages"][0] == {"role": "system", "content": "sys"}


def test_llm_http_error_is_reported(rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    rconfig.llm.base_url = fake_web.url
    fake_web.route("POST", "/v1/messages", status=400, json={"error": {"message": "bad model"}})
    with pytest.raises(LLMError, match="bad model"):
        AnthropicClient(rconfig).complete("s", "p", "nope")


def test_unknown_provider(rconfig):
    rconfig.llm.provider = "mystery"
    with pytest.raises(LLMError):
        make_llm(rconfig)


def test_embeddings(rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    rconfig.llm.embedding_model = "emb"
    rconfig.llm.embedding_base_url = f"{fake_web.url}/v1"
    fake_web.route("POST", "/v1/embeddings", lambda req: (200, {}, {"data": [
        {"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]}))
    vectors = EmbeddingClient(rconfig).embed(["a", "b"])
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])        # ordered by index, normalised


# ── JSON helpers ─────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go:\n{"a": 1}\nHope that helps.',
])
def test_parse_json_tolerates_chatter(text):
    assert parse_json(text) == {"a": 1}


def test_parse_json_rejects_garbage():
    with pytest.raises(InvalidOutput):
        parse_json("no json here")


def test_complete_json_retries_with_feedback():
    llm = ScriptedLLM(["not json", '{"n": 2}'])

    def validate(data):
        if data.get("n") != 2:
            raise ValueError("n must be 2")
        return data["n"]

    value, _ = complete_json(llm, "sys", "prompt", "m", validate)
    assert value == 2
    assert "previous reply was invalid" in llm.calls[1]["prompt"]


def test_complete_json_gives_up():
    llm = ScriptedLLM(['{"n": 1}', '{"n": 1}'])

    def validate(data):
        raise ValueError("never good")

    with pytest.raises(InvalidOutput, match="never good"):
        complete_json(llm, "s", "p", "m", validate)
