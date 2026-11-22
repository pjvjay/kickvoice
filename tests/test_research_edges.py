"""Research layer edge cases: chunking, HTTP politeness, model retries, retrieval, collection branches."""

import json
import os
import time

import numpy as np
import pytest
import requests

from conftest import ScriptedLLM
from kickvoice.pipelines.research import CollectOptions, collect
from kickvoice.research import llm as llm_mod
from kickvoice.research.chunking import chunk_text, chunk_timed
from kickvoice.research.net import Blocked, Fetcher, NetError, RateLimited, Response
from kickvoice.research.retrieve import Retriever, snippet
from kickvoice.research.sources import social, web, youtube
from kickvoice.research.sources.base import SourceContext
from kickvoice.research.store import CONFIRMED, Chunk, Claim, Document, ResearchStore
from test_research_sources import FakeYtDlp

# ── chunking ─────────────────────────────────────────────────────


def test_chunk_text_paragraphs_sentences_and_runons():
    assert chunk_text("   ") == []
    short = chunk_text("One para.\n\nTwo para.", max_chars=100)
    assert [c.text for c in short] == ["One para.\n\nTwo para."]
    long_para = " ".join(f"Sentence number {i} is here." for i in range(40))
    pieces = chunk_text(long_para, max_chars=200)
    assert all(len(c.text) <= 200 for c in pieces) and len(pieces) > 4
    assert pieces[0].text.startswith("Sentence number 0") and pieces[0].text.endswith(".")
    runon = chunk_text("x" * 450, max_chars=200)  # no punctuation at all
    assert [len(c.text) for c in runon] == [200, 200, 50]


def test_chunk_timed_splits_on_size_and_silence():
    lines = [(0, 1, "a" * 50), (1, 2, "b" * 50), (2, 3, " "), (60, 61, "after a long pause")]
    chunks = chunk_timed(lines, max_chars=80, max_gap_s=20)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 1), (1, 2), (60, 61)]
    assert chunk_timed([]) == []


# ── polite HTTP ──────────────────────────────────────────────────


def test_fetcher_network_errors_retry_then_fail(rconfig, monkeypatch):
    calls = []

    def broken(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("refused")

    f = Fetcher(rconfig)
    monkeypatch.setattr(f.session, "request", broken)
    monkeypatch.setattr("kickvoice.research.net.time.sleep", lambda s: None)
    with pytest.raises(NetError, match="refused"):
        f.get("http://example.invalid/x", cache=False)
    assert len(calls) == 4


def test_fetcher_gives_up_after_repeated_5xx(rconfig, fake_web, monkeypatch):
    monkeypatch.setattr("kickvoice.research.net.time.sleep", lambda s: None)
    fake_web.route("GET", "/down", status=503, json={"e": 1}, headers={"Retry-After": "nonsense"})
    r = Fetcher(rconfig).get(f"{fake_web.url}/down", cache=False)
    assert r.status == 503 and len(fake_web.hits("/down")) == 4
    with pytest.raises(NetError, match="503"):
        Fetcher(rconfig).get_json(f"{fake_web.url}/down", cache=False)


def test_rate_limit_short_wait_is_retried(rconfig, fake_web, monkeypatch):
    slept = []
    monkeypatch.setattr("kickvoice.research.net.time.sleep", slept.append)
    calls = []

    def limited(req):
        calls.append(1)
        if len(calls) == 1:
            return 429, {"x-rate-limit-reset": str(int(time.time()) + 2)}, {}
        return 200, {}, {"ok": True}

    fake_web.route("GET", "/limited", limited)
    assert Fetcher(rconfig).get_json(f"{fake_web.url}/limited", cache=False) == {"ok": True}
    assert 0 < slept[0] <= 2


def test_cache_expiry_and_corruption(rconfig, fake_web):
    fake_web.route("GET", "/c", json={"n": 1})
    f = Fetcher(rconfig)
    f.get(f"{fake_web.url}/c")
    path = f._cache_path(f"{fake_web.url}/c")
    path.write_text("{corrupt")
    assert not f.get(f"{fake_web.url}/c").from_cache  # corrupt → refetched
    os.utime(path, (0, 0))  # very old
    rconfig.research.cache_ttl_hours = 1
    assert not f.get(f"{fake_web.url}/c").from_cache
    assert len(fake_web.hits("/c")) == 3


def test_response_json_error_and_robots_edge_cases(rconfig, fake_web):
    with pytest.raises(NetError, match="Expected JSON"):
        Response("u", 200, "<html>", {}).json()
    fake_web.route("GET", "/robots.txt", status=500, text="oops")
    fake_web.route("GET", "/page", text="<p>hi</p>")
    assert Fetcher(rconfig).allowed_by_robots(f"{fake_web.url}/page")  # robots error → allowed
    f = Fetcher(rconfig)
    f._robots[fake_web.url] = None
    f._robots.clear()
    fake_web.route("GET", "/robots.txt", text="User-agent: *\nDisallow: /page\n")
    with pytest.raises(Blocked):
        f.get(f"{fake_web.url}/page", respect_robots=True)


def test_post_helper(rconfig, fake_web):
    fake_web.route("POST", "/p", json={"ok": 1})
    r = Fetcher(rconfig).post(f"{fake_web.url}/p", json_body={"a": 1})
    assert r.json() == {"ok": 1} and json.loads(fake_web.hits("/p")[0]["body"]) == {"a": 1}
    with pytest.raises(RateLimited):
        raise RateLimited("x", 429)


# ── model API retries ────────────────────────────────────────────


def test_llm_post_retries_on_429_and_network_errors(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    responses = [
        requests.ConnectionError("reset"),
        type("R", (), {"status_code": 429, "headers": {"retry-after": "1"}, "text": ""})(),
        type("R", (), {"status_code": 200, "headers": {}, "json": lambda self: {"ok": 1}})(),
    ]

    def fake_post(*a, **k):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    assert llm_mod._post("http://x", {}, {}, 5) == {"ok": 1}

    monkeypatch.setattr(
        llm_mod.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow"))
    )
    with pytest.raises(llm_mod.LLMError, match="slow"):
        llm_mod._post("http://x", {}, {}, 5)


def test_openai_unexpected_shape_and_embedder_factory(rconfig, fake_web):
    rconfig.llm.provider = "openai"
    rconfig.llm.base_url = f"{fake_web.url}/v1"
    fake_web.route("POST", "/v1/chat/completions", json={"choices": []})
    with pytest.raises(llm_mod.LLMError, match="Unexpected response shape"):
        llm_mod.make_llm(rconfig).complete("s", "p", "m")
    assert llm_mod.make_embedder(rconfig) is None
    rconfig.llm.embedding_model = "emb"
    assert isinstance(llm_mod.make_embedder(rconfig), llm_mod.EmbeddingClient)


# ── retrieval with embeddings ────────────────────────────────────


class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = np.array([1.0 if "housing" in t else 0.0, 1.0 if "tax" in t else 0.0, 0.1])
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def test_retriever_fuses_embedding_hits(config):
    with ResearchStore(config.research_db) as store:
        p = store.add_person("Jane")
        d = Document(p.id, "x", "post", "1", "Shelter costs keep rising everywhere.", "self")
        store.upsert_document(d, [Chunk(d.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        (cid,) = store.add_claims(
            [
                Claim(
                    p.id,
                    d.id,
                    chunk.id,
                    "housing",
                    "housing costs rise",
                    "Shelter costs keep rising everywhere",
                )
            ]
        )
        store.set_claim_embeddings([(cid, FakeEmbedder().embed(["housing"])[0])])
        # The query shares no words with the claim: only the embedding can find it.
        hits = Retriever(store, p, FakeEmbedder()).search("housing crisis", k=5)
        assert hits and hits[0].ref == f"C{cid}"
        assert Retriever(store, p).search("zzz", k=5) == []
        assert snippet("short text", "anything") == "short text"
        assert snippet("word " * 200, "").endswith(" …")


# ── collection branches ──────────────────────────────────────────


@pytest.fixture
def person(rconfig):
    store = ResearchStore(rconfig.research_db)
    p = store.add_person("Jane Pundit", voice_profile="Jane")
    yield store, p
    store.close()


def test_collect_voice_modes_twitch_website_and_search(person, rconfig, fake_web, monkeypatch):
    store, p = person
    store.add_account(p.id, "youtube", "UC" + "a" * 22, status=CONFIRMED)
    store.add_account(p.id, "twitch", "jane", status=CONFIRMED)
    store.add_account(p.id, "website", "https://jane.example", url=f"{fake_web.url}/blog", status=CONFIRMED)
    fake_web.route("GET", "/robots.txt", status=404, text="")
    fake_web.route(
        "GET", "/blog", text="<article><p>" + "My own essay on housing policy. " * 10 + "</p></article>"
    )
    fake_web.route(
        "GET",
        "/elsewhere",
        text="<article><p>" + "An article about Jane and housing. " * 10 + "</p></article>",
    )
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    fake_web.route(
        "GET",
        "/brave",
        json={
            "web": {
                "results": [
                    {"title": "About Jane", "url": f"{fake_web.url}/elsewhere"},
                    {"title": "Her blog", "url": f"{fake_web.url}/blog"},
                ]
            }
        },
    )
    voice = lambda url, workdir: [(1.0, 4.0, f"my words from {url[-3:]}")]  # noqa: E731
    results = collect(
        store,
        p,
        rconfig,
        CollectOptions(voice=True, appearances=2, web_search=2, max_videos=1),
        ytdlp=FakeYtDlp(videos=2),
        voice_transcriber=voice,
    )
    assert results["youtube"].new == 1 and results["twitch"].new == 1
    assert results["appearances"].new >= 1 and results["web"].new == 2
    kinds = {(d.source, d.authorship) for d in store.documents(p.id)}
    assert ("youtube", "self_voice") in kinds and ("web", "about") in kinds and ("web", "self") in kinds
    vod = store.documents(p.id, source="twitch")[0]  # a Twitch VOD is labelled as Twitch
    assert vod.url == "https://www.twitch.tv/videos/0"


def test_deep_links():
    from kickvoice.research.store import deep_link

    assert deep_link("https://www.twitch.tv/videos/9", 3725) == "https://www.twitch.tv/videos/9?t=1h2m5s"
    assert deep_link("https://youtu.be/x", 5) == "https://youtu.be/x?t=5s"


def test_collect_only_filter_and_search_errors(person, rconfig, monkeypatch):
    store, p = person
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    store.add_account(p.id, "x", "jane", status=CONFIRMED)
    results = collect(store, p, rconfig, CollectOptions(only=("web",), web_search=3))
    assert "x" not in results and "BRAVE_API_KEY" in results["web"].errors[0]


def test_collectors_error_branches(person, rconfig, fake_web, monkeypatch):
    store, p = person
    ctx = SourceContext(rconfig, Fetcher(rconfig), store, p)
    monkeypatch.setenv("X_BEARER_TOKEN", "t")
    fake_web.route("GET", "/2/users/by/username/ghost", json={"errors": [{"detail": "not found"}]})
    assert "not found" in social.collect_x(ctx, store.add_account(p.id, "x", "ghost")).errors[0]
    fake_web.route("GET", "/2/users/by/username/jane", json={"data": {"id": "9"}})
    fake_web.route("GET", "/2/users/9/tweets", status=401, json={"title": "Unauthorized"})
    assert "401" in social.collect_x(ctx, store.add_account(p.id, "x", "jane")).errors[0]

    monkeypatch.setenv("REDDIT_CLIENT_ID", "i")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "s")
    fake_web.route("POST", "/reddit/api/v1/access_token", status=401, json={"error": "invalid"})
    assert (
        "Reddit auth failed" in social.collect_reddit(ctx, store.add_account(p.id, "reddit", "j")).errors[0]
    )
    fake_web.route("POST", "/reddit/api/v1/access_token", json={"access_token": "T"})
    fake_web.route("GET", "/reddit/user/j/comments", status=500, json={})
    fake_web.route("GET", "/reddit/user/j/submitted", json={"data": {"children": []}})
    monkeypatch.setattr("kickvoice.research.net.time.sleep", lambda s: None)
    assert "reddit comments" in social.collect_reddit(ctx, store.add_account(p.id, "reddit", "j")).errors[0]
    assert social._utc("not a time") is None

    fake_web.route("GET", "/robots.txt", status=404, text="")
    fake_web.route("GET", "/missing", status=404, text="gone")
    fake_web.route("GET", "/data.json", json={"not": "html"})
    fake_web.route("GET", "/thin", text="<p>too short</p>")
    assert "HTTP 404" in web.collect_url(ctx, f"{fake_web.url}/missing", False).errors[0]
    assert "not an HTML page" in web.collect_url(ctx, f"{fake_web.url}/data.json", False).errors[0]
    assert "no readable" in web.collect_url(ctx, f"{fake_web.url}/thin", False).errors[0]
    assert "failed" in web.collect_url(ctx, "http://127.0.0.1:1/closed", False).errors[0]


def test_youtube_error_branches(person, rconfig, monkeypatch):
    store, p = person
    ctx = SourceContext(rconfig, Fetcher(rconfig), store, p)

    class Broken:
        def run(self, args, timeout=600):
            raise RuntimeError("yt-dlp failed: HTTP 429")

    acct = store.add_account(p.id, "youtube", "@jane")
    assert "429" in youtube.collect_channel(ctx, acct, ytdlp=Broken()).errors[0]
    assert "429" in youtube.collect_appearances(ctx, "q", lambda u, w: [], set(), ytdlp=Broken()).errors[0]
    # voice mode that finds nothing of theirs stores nothing
    stats = youtube.collect_channel(ctx, acct, ytdlp=FakeYtDlp(videos=1), voice=lambda u, w: [])
    assert stats.new == 0
    assert youtube._published({"upload_date": "2024"}) is None

    import subprocess

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "--timeout" in cmd:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: private video")

    monkeypatch.setattr(youtube.subprocess, "run", fake_run)
    monkeypatch.setattr(youtube, "require_tool", lambda name, hint="": name)
    with pytest.raises(RuntimeError, match="private video"):
        youtube.YtDlp().run(["x"])
    monkeypatch.setattr(
        youtube.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd, 1))
    )
    with pytest.raises(RuntimeError, match="timed out"):
        youtube.YtDlp().run(["x"])


def test_reddit_and_x_missing_credentials(person, rconfig, monkeypatch):
    store, p = person
    ctx = SourceContext(rconfig, Fetcher(rconfig), store, p)
    for name in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    assert "REDDIT_CLIENT_ID" in social.collect_reddit(ctx, store.add_account(p.id, "reddit", "j")).errors[0]


def test_extraction_handles_bad_model_items(config):
    from kickvoice.research.extract import _validate_shape, extract_claims, find_quote

    assert _validate_shape(
        {
            "claims": [
                "junk",
                {"chunk": "K1"},
                {"chunk": "K1", "topic": "t", "claim": "c", "quote": "q", "kind": "weird", "sincere": False},
            ]
        }
    ) == [{"chunk": "K1", "topic": "t", "claim": "c", "quote": "q", "kind": "position", "sincere": False}]
    with pytest.raises(ValueError):
        _validate_shape(["not", "a", "dict"])
    assert find_quote("one two three four five six seven eight nine", "short source") is None
    with ResearchStore(config.research_db) as store:
        p = store.add_person("X")
        assert extract_claims(store, p, ScriptedLLM([]), config).chunks == 0
