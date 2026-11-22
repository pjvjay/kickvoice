"""Collectors against a fake web server / fake yt-dlp."""

import json
from pathlib import Path

import pytest

from kickvoice.pipelines.research import CollectOptions, collect, discover
from kickvoice.research.net import Fetcher
from kickvoice.research.sources import local, social, web, wiki, youtube
from kickvoice.research.sources.base import SourceContext
from kickvoice.research.store import (
    ABOUT,
    CANDIDATE,
    CONFIRMED,
    SELF,
    SELF_CHANNEL,
    SELF_VOICE,
    ResearchStore,
)

# Real YouTube auto-caption layout: a whitespace-only line INSIDE each cue,
# an empty line between cues, and each line repeated in the next cue.
AUTO_VTT = (
    "WEBVTT\nKind: captions\nLanguage: en\n\n"
    "00:00:00.160 --> 00:00:02.270 align:start position:0%\n"
    " \n"
    "so<00:00:00.400><c> today</c><00:00:00.640><c> we</c><00:00:00.880><c> talk</c>\n\n"
    "00:00:02.270 --> 00:00:02.280 align:start position:0%\n"
    "so today we talk\n"
    " \n\n"
    "00:00:02.280 --> 00:00:05.110 align:start position:0%\n"
    "so today we talk\n"
    "about<00:00:02.520><c> the</c><00:00:02.760><c> minimum</c><00:00:03.000><c> wage</c>\n\n"
    "00:00:05.110 --> 00:00:05.120 align:start position:0%\n"
    "about the minimum wage\n"
    " \n\n"
    "00:01:05.120 --> 00:01:08.000 align:start position:0%\n"
    "about the minimum wage\n"
    "it<00:01:05.300><c> should</c><00:01:05.500><c> track</c><00:01:05.700><c> inflation</c> &amp; rents\n"
)

# Standard (non-YouTube) WebVTT with numeric cue identifiers.
MANUAL_VTT = (
    "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nFirst line.\n\n"
    "2\n00:00:03.500 --> 00:00:05.000\nSecond line,\nstill second cue.\n"
)


@pytest.fixture
def store(rconfig):
    with ResearchStore(rconfig.research_db) as s:
        yield s


@pytest.fixture
def person(store):
    return store.add_person("Jane Pundit", voice_profile="Jane")


@pytest.fixture
def ctx(rconfig, store, person):
    return SourceContext(rconfig, Fetcher(rconfig), store, person)


# ── Wikidata / Wikipedia ─────────────────────────────────────────

WIKIDATA_ENTITY = {
    "entities": {
        "Q42": {
            "labels": {"en": {"value": "Jane Pundit"}},
            "descriptions": {"en": {"value": "American political streamer"}},
            "sitelinks": {"enwiki": {"title": "Jane Pundit"}},
            "claims": {
                "P31": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
                "P2002": [
                    {"rank": "normal", "mainsnak": {"datavalue": {"value": "JanePundit"}}},
                    {"rank": "deprecated", "mainsnak": {"datavalue": {"value": "OldJane"}}},
                ],
                "P2397": [{"rank": "preferred", "mainsnak": {"datavalue": {"value": "UC" + "a" * 22}}}],
                "P856": [{"rank": "normal", "mainsnak": {"datavalue": {"value": "https://jane.example"}}}],
            },
        }
    }
}


def wikidata_routes(fake_web):
    def api(req):
        q = req["query"]
        if q.get("action") == "wbsearchentities":
            return (
                200,
                {},
                {
                    "search": [
                        {"id": "Q42", "label": "Jane Pundit", "description": "American political streamer"}
                    ]
                },
            )
        if q.get("action") == "query" and q.get("list") == "search":
            hit = q["srsearch"] == "haswbstatement:P2002=JanePundit"
            return 200, {}, {"query": {"search": [{"title": "Q42"}] if hit else []}}
        if q.get("action") == "wbgetentities":
            return 200, {}, WIKIDATA_ENTITY
        return 400, {}, {"error": "?"}

    fake_web.route("GET", "/wd/api.php", api)


def test_wikidata_person_accounts(rconfig, fake_web):
    wikidata_routes(fake_web)
    p = wiki.get_person(Fetcher(rconfig), rconfig.research.wikidata_api, "Q42")
    assert p.label == "Jane Pundit" and p.wikipedia_title == "Jane Pundit"
    assert ("x", "JanePundit", "https://x.com/JanePundit") in p.accounts
    assert not any(h == "OldJane" for _, h, _ in p.accounts)  # deprecated skipped
    assert ("youtube", "UC" + "a" * 22, "https://www.youtube.com/channel/UC" + "a" * 22) in p.accounts
    assert ("website", "https://jane.example", "https://jane.example") in p.accounts


def test_discover_by_handle_then_by_name(rconfig, fake_web):
    wikidata_routes(fake_web)
    by_handle = discover(Fetcher(rconfig), rconfig, "whoever", {"x": "JanePundit"})
    assert [(c.record.qid, c.matched_by, c.strong) for c in by_handle.candidates] == [
        ("Q42", "account", True)
    ]
    assert not by_handle.searched_by_name
    by_name = discover(Fetcher(rconfig), rconfig, "Jane Pundit")
    assert [(c.record.qid, c.matched_by, c.strong) for c in by_name.candidates] == [("Q42", "name", False)]
    assert wiki.get_person(Fetcher(rconfig), rconfig.research.wikidata_api, "Q42").is_human


def test_collect_wikipedia_sections(ctx, store, person, rconfig, fake_web):
    fake_web.route(
        "GET",
        "/w/api.php",
        json={
            "query": {
                "pages": [
                    {
                        "pageid": 1,
                        "title": "Jane Pundit",
                        "pageprops": {"wikibase_item": "Q42"},
                        "revisions": [{"timestamp": "2025-01-02T03:04:05Z"}],
                        "extract": "Jane Pundit is a streamer.\n\n== Political views ==\nShe supports a carbon tax "
                        "and opposes rent control.\n\n== Personal life ==\nLives somewhere.",
                    }
                ]
            }
        },
    )
    acct = store.add_account(person.id, "wikipedia", "Jane Pundit", status=CONFIRMED)
    stats = wiki.collect_wikipedia(ctx, acct)
    assert stats.new == 1
    (doc,) = store.documents(person.id)
    assert doc.authorship == ABOUT and doc.url.endswith("/wiki/Jane_Pundit")
    assert doc.published_at.startswith("2025-01-02") and doc.meta["wikidata"] == "Q42"
    chunks = [c.text for c, _ in store.chunks_to_analyze(person.id)]
    assert any(t.startswith("[Political views] She supports a carbon tax") for t in chunks)
    # re-collect: unchanged, no duplicates
    assert wiki.collect_wikipedia(ctx, acct).unchanged == 1


def test_collect_wikipedia_missing(ctx, store, person, fake_web):
    fake_web.route("GET", "/w/api.php", json={"query": {"pages": [{"title": "Nope", "missing": True}]}})
    acct = store.add_account(person.id, "wikipedia", "Nope", status=CONFIRMED)
    assert "not found" in wiki.collect_wikipedia(ctx, acct).errors[0]


# ── X ────────────────────────────────────────────────────────────


def x_routes(fake_web, pages):
    fake_web.route(
        "GET", "/2/users/by/username/JanePundit", json={"data": {"id": "99", "username": "JanePundit"}}
    )

    def timeline(req):
        token = req["query"].get("pagination_token")
        return 200, {}, pages[token]

    fake_web.route("GET", "/2/users/99/tweets", timeline)


def test_collect_x_paginates(ctx, store, person, fake_web, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(
        fake_web,
        {
            None: {
                "data": [
                    {
                        "id": "1",
                        "text": "Rent control fails every time.",
                        "created_at": "2024-05-01T10:00:00Z",
                    },
                    {
                        "id": "2",
                        "text": "@bob no, read the study",
                        "created_at": "2024-05-02T10:00:00Z",
                        "referenced_tweets": [{"type": "replied_to", "id": "0"}],
                    },
                ],
                "meta": {"next_token": "p2"},
            },
            "p2": {
                "data": [{"id": "3", "text": "Carbon tax now.", "created_at": "2024-04-01T10:00:00Z"}],
                "meta": {},
            },
        },
    )
    acct = store.add_account(person.id, "x", "@JanePundit", status=CONFIRMED)
    stats = social.collect_x(ctx, acct)
    assert stats.new == 3 and not stats.errors
    docs = {d.external_id: d for d in store.documents(person.id)}
    assert docs["1"].authorship == SELF and docs["1"].url == "https://x.com/JanePundit/status/1"
    assert docs["2"].kind == "reply" and docs["1"].kind == "post"
    req = fake_web.hits("/2/users/99/tweets")[0]
    assert req["headers"]["authorization"] == "Bearer tok"
    assert req["query"]["exclude"] == "retweets"
    assert fake_web.hits("/2/users/99/tweets")[1]["query"]["pagination_token"] == "p2"


def test_collect_x_without_token(ctx, store, person, monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    acct = store.add_account(person.id, "x", "JanePundit", status=CONFIRMED)
    assert "X_BEARER_TOKEN" in social.collect_x(ctx, acct).errors[0]


def test_collect_x_respects_limit(ctx, store, person, fake_web, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(
        fake_web,
        {
            None: {
                "data": [{"id": str(i), "text": f"post {i}"} for i in range(10)],
                "meta": {"next_token": "more"},
            },
            "more": {"data": [{"id": "x", "text": "never"}], "meta": {}},
        },
    )
    acct = store.add_account(person.id, "x", "JanePundit", status=CONFIRMED)
    social.collect_x(ctx, acct, limit=10)
    assert len(fake_web.hits("/2/users/99/tweets")) == 1


# ── Reddit ───────────────────────────────────────────────────────


def test_collect_reddit(ctx, store, person, fake_web, monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    fake_web.route("POST", "/reddit/api/v1/access_token", json={"access_token": "T", "expires_in": 3600})
    fake_web.route(
        "GET",
        "/reddit/user/janep/comments",
        json={
            "data": {
                "after": None,
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "name": "t1_a",
                            "body": "Zoning is the real problem.",
                            "permalink": "/r/x/comments/1/a",
                            "created_utc": 1700000000,
                            "subreddit": "economics",
                            "link_title": "Housing",
                        },
                    },
                    {"kind": "t1", "data": {"name": "t1_b", "body": "[deleted]", "permalink": "/r/x/b"}},
                ],
            }
        },
    )
    fake_web.route(
        "GET",
        "/reddit/user/janep/submitted",
        json={
            "data": {
                "after": None,
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "name": "t3_c",
                            "title": "My take on UBI",
                            "selftext": "It works.",
                            "permalink": "/r/x/comments/2/c",
                            "created_utc": 1700000100,
                        },
                    }
                ],
            }
        },
    )
    acct = store.add_account(person.id, "reddit", "u/janep", status=CONFIRMED)
    stats = social.collect_reddit(ctx, acct)
    assert stats.new == 2
    docs = {d.external_id: d for d in store.documents(person.id)}
    assert docs["t1_a"].url == "https://www.reddit.com/r/x/comments/1/a"
    assert docs["t3_c"].text.startswith("My take on UBI") and docs["t3_c"].kind == "post"
    assert docs["t1_a"].published_at.startswith("2023-11-14")
    auth = fake_web.hits("/reddit/api/v1/access_token")[0]
    assert auth["headers"]["authorization"].startswith("Basic ")
    assert fake_web.hits("/reddit/user/janep/comments")[0]["headers"]["authorization"] == "bearer T"


# ── web ──────────────────────────────────────────────────────────

ARTICLE = """<html><head><title>Ignored</title>
<meta property="og:title" content="Why I changed my mind on rent control">
<meta property="article:published_time" content="2023-03-01T00:00:00Z">
<script>var tracking = "nope";</script></head>
<body><nav>Home | About | Subscribe</nav>
<article><h1>Why I changed my mind</h1>
<p>For years I argued that rent control protects tenants. The evidence convinced me otherwise.</p>
<p>Supply is the only durable fix, and zoning reform is how we get it in most cities.</p></article>
<footer>© 2023 all rights reserved</footer></body></html>"""


def test_html_to_text_keeps_prose_drops_chrome():
    title, text, meta = web.html_to_text(ARTICLE)
    assert title == "Why I changed my mind on rent control"
    assert "rent control protects tenants" in text and "zoning reform" in text
    assert "tracking" not in text and "Subscribe" not in text and "rights reserved" not in text
    assert meta["article:published_time"].startswith("2023-03-01")


def test_collect_url(ctx, store, person, fake_web):
    fake_web.route("GET", "/robots.txt", status=404, text="")
    fake_web.route("GET", "/blog/post", text=ARTICLE)
    stats = web.collect_url(ctx, f"{fake_web.url}/blog/post", first_hand=True)
    assert stats.new == 1
    (doc,) = store.documents(person.id)
    assert doc.authorship == SELF and doc.title.startswith("Why I changed")


def test_collect_url_respects_robots(ctx, store, person, fake_web):
    fake_web.route("GET", "/robots.txt", text="User-agent: *\nDisallow: /\n")
    fake_web.route("GET", "/blog/post", text=ARTICLE)
    assert "robots.txt" in web.collect_url(ctx, f"{fake_web.url}/blog/post", True).errors[0]
    assert fake_web.hits("/blog/post") == []


def test_search_web_needs_key(ctx, monkeypatch, fake_web):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BRAVE_API_KEY"):
        web.search_web(ctx, "x")
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    fake_web.route("GET", "/brave", json={"web": {"results": [{"title": "t", "url": "https://a.example/x"}]}})
    assert web.search_web(ctx, "Jane")[0]["url"] == "https://a.example/x"
    assert fake_web.hits("/brave")[0]["headers"]["x-subscription-token"] == "b"


# ── YouTube ──────────────────────────────────────────────────────


def test_parse_vtt_standard_file_with_cue_identifiers():
    lines = youtube.parse_vtt(MANUAL_VTT)
    assert [t for _, _, t in lines] == ["First line.", "Second line,", "still second cue."]
    assert lines[1][0] == pytest.approx(3.5)


def test_parse_vtt_dedupes_rolling_auto_captions():
    lines = youtube.parse_vtt(AUTO_VTT)
    assert [t for _, _, t in lines] == [
        "so today we talk",
        "about the minimum wage",
        "it should track inflation & rents",
    ]
    assert lines[0][0] == pytest.approx(0.16) and lines[2][0] == pytest.approx(65.12)


@pytest.mark.parametrize(
    "handle,url",
    [
        ("UC" + "b" * 22, "https://www.youtube.com/channel/UC" + "b" * 22 + "/videos"),
        ("@jane", "https://www.youtube.com/@jane/videos"),
        ("jane", "https://www.youtube.com/@jane/videos"),
        ("https://www.youtube.com/@jane", "https://www.youtube.com/@jane/videos"),
        (
            "https://www.twitch.tv/jane/videos?filter=archives",
            "https://www.twitch.tv/jane/videos?filter=archives",
        ),
    ],
)
def test_channel_videos_url(handle, url):
    assert youtube.channel_videos_url(handle) == url


class FakeYtDlp:
    def __init__(self, vtt=AUTO_VTT, videos=2):
        self.vtt = vtt
        self.videos = videos
        self.calls = []

    def run(self, args, timeout=600):
        self.calls.append(args)
        url = args[-1]
        if "--flat-playlist" in args:
            prefix = "s" if url.startswith("ytsearch") else "v"
            count = self.videos
            if "-I" in args:  # honour "-I 1:N" like yt-dlp
                count = min(count, int(args[args.index("-I") + 1].split(":")[1]))
            twitch = "twitch.tv" in url
            return json.dumps(
                {
                    "channel_id": "UC" + "a" * 22,
                    "channel": "Jane",
                    "entries": [
                        {
                            "id": f"{'t' if twitch else prefix}{i}",
                            "title": f"Video {i}",
                            **({"url": f"https://www.twitch.tv/videos/{i}"} if twitch else {}),
                            "channel_id": "UC" + ("a" if prefix == "v" or i == 0 else "z") * 22,
                        }
                        for i in range(count)
                    ],
                }
            )
        vid = url.rsplit("=", 1)[-1]
        meta = {
            "id": vid,
            "title": f"Title {vid}",
            "upload_date": "20240102",
            "channel": "Jane",
            "channel_id": "UC" + "a" * 22,
            "duration": 300,
        }
        if "--no-simulate" in args and self.vtt is not None:
            out = Path(args[args.index("-o") + 1]).parent
            (out / f"{vid}.en.vtt").write_text(self.vtt)
            (out / f"{vid}.en-orig.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nworse\n")
        return json.dumps(meta) + "\n"


def test_collect_channel_captions(ctx, store, person):
    acct = store.add_account(person.id, "youtube", "@jane", status=CONFIRMED)
    yt = FakeYtDlp()
    stats = youtube.collect_channel(ctx, acct, ytdlp=yt, limit=2)
    assert stats.new == 2
    doc = store.documents(person.id)[0]
    assert doc.authorship == SELF_CHANNEL and doc.published_at == "2024-01-02"
    assert doc.url.startswith("https://www.youtube.com/watch?v=v")
    assert "worse" not in doc.text  # preferred plain 'en' track
    assert any("-I" in c and "1:2" in c for c in yt.calls)
    chunks = store.chunks_to_analyze(person.id)
    assert chunks[0][0].start_s == pytest.approx(0.16)


def test_collect_channel_without_captions(ctx, store, person):
    acct = store.add_account(person.id, "youtube", "@jane", status=CONFIRMED)
    stats = youtube.collect_channel(ctx, acct, ytdlp=FakeYtDlp(vtt=None), limit=1)
    assert stats.new == 0 and "no captions" in stats.errors[0]


def test_collect_channel_voice_mode(ctx, store, person):
    acct = store.add_account(person.id, "youtube", "@jane", status=CONFIRMED)
    voice = lambda url, workdir: [(10.0, 14.0, "my own words on housing")]  # noqa: E731
    stats = youtube.collect_channel(ctx, acct, ytdlp=FakeYtDlp(), voice=voice, limit=1)
    assert stats.new == 1
    doc = store.documents(person.id)[0]
    assert doc.authorship == SELF_VOICE and doc.external_id.endswith(":voice")
    assert doc.text == "my own words on housing" and doc.title == "Title v0"


def test_appearances_exclude_own_channel(ctx, store, person):
    voice = lambda url, workdir: [(1.0, 3.0, f"said on {url[-2:]}")]  # noqa: E731
    stats = youtube.collect_appearances(
        ctx, '"Jane"', voice, {"UC" + "a" * 22}, ytdlp=FakeYtDlp(videos=3), limit=3
    )
    # s0 is on her own channel → skipped; s1, s2 are other channels
    assert stats.new == 2


# ── local ────────────────────────────────────────────────────────


def test_collect_transcripts_only_confident_lines_of_that_voice(ctx, store, person, rconfig):
    from kickvoice.transcripts.store import TranscriptStore, Utterance

    with TranscriptStore(rconfig.transcripts_db) as ts:
        s = ts.create_session("s1", "https://kick.com/debates")
        ts.add_utterances(
            [
                Utterance("s1", 0, 4, "carbon tax is the answer", speaker="Jane", attribution="confident"),
                Utterance("s1", 5, 6, "maybe", speaker="Jane", attribution="probable"),
                Utterance("s1", 7, 9, "no it is not", speaker="Bob", attribution="confident"),
            ]
        )
    stats = local.collect_transcripts(ctx, "Jane")
    assert stats.new == 1
    (doc,) = store.documents(person.id)
    assert doc.text == "carbon tax is the answer" and doc.authorship == SELF_VOICE
    assert doc.url == "https://kick.com/debates" and s.id in doc.external_id


def test_import_files(ctx, store, person, tmp_path):
    jsonl = tmp_path / "posts.jsonl"
    jsonl.write_text(
        '{"id": "a", "text": "first post", "date": "2024-01-01"}\n{"text": ""}\n'
        '{"text": "second post", "url": "https://e.example/2"}\n'
    )
    assert local.import_file(ctx, jsonl).new == 2
    archive = tmp_path / "tweets.js"
    archive.write_text(
        'window.YTD.tweets.part0 = [{"tweet": {"id_str": "77", '
        '"full_text": "archived tweet", "created_at": "Wed Oct 10 20:19:24 +0000 2018"}}]'
    )
    assert local.import_file(ctx, archive).new == 1
    assert any(d.source == "x" and d.external_id == "77" for d in store.documents(person.id))
    notes = tmp_path / "profile.md"
    notes.write_text("A long magazine profile about her.")
    assert local.import_file(ctx, notes, first_hand=False).new == 1
    assert store.documents(person.id, source="file")[0].authorship in (SELF, ABOUT)


# ── orchestration ────────────────────────────────────────────────


def test_collect_uses_only_confirmed_accounts(store, person, rconfig, fake_web, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(fake_web, {None: {"data": [{"id": "1", "text": "hello"}], "meta": {}}})
    store.add_account(person.id, "x", "JanePundit", status=CANDIDATE)
    assert collect(store, person, rconfig) == {}  # nothing confirmed yet
    store.add_account(person.id, "x", "JanePundit", status=CONFIRMED)
    results = collect(store, person, rconfig)
    assert results["x"].new == 1


def test_collect_reports_unsupported_and_voice_requirements(store, person, rconfig):
    store.add_account(person.id, "instagram", "jane", status=CONFIRMED)
    store.add_account(person.id, "twitch", "jane", status=CONFIRMED)
    results = collect(store, person, rconfig, CollectOptions(appearances=3), ytdlp=FakeYtDlp())
    assert "no supported collector" in results["instagram"].errors[0]
    assert "--voice" in results["twitch"].errors[0]
    assert "--voice" in results["appearances"].errors[0]


def test_collect_voice_requires_profile(store, rconfig):
    nobody = store.add_person("No Voice")
    with pytest.raises(ValueError, match="voice profile"):
        collect(store, nobody, rconfig, CollectOptions(voice=True))


def test_voice_transcriber_keeps_confident_lines_and_cleans_up(rconfig, monkeypatch, tmp_path):
    """Voice mode must not leave a transcript session behind (it would be re-imported)."""
    import kickvoice.audio.download as dl
    import kickvoice.pipelines.transcribe as tmod
    from kickvoice.transcripts.store import TranscriptStore, Utterance

    monkeypatch.setattr(dl, "download_audio", lambda url, workdir: tmp_path / "a.wav")

    class FakePipeline:
        def __init__(self, file_path=None, config=None, on_utterance=None, out=None):
            self.transcripts = TranscriptStore(config.transcripts_db)

        def run(self, max_duration_s=None):
            s = self.transcripts.create_session("vid", "a.wav")
            self.transcripts.add_utterances(
                [
                    Utterance(s.id, 1, 4, "mine and sure", speaker="Jane", attribution="confident"),
                    Utterance(s.id, 5, 6, "mine but short", speaker="Jane", attribution="probable"),
                    Utterance(s.id, 7, 9, "the host talking", speaker="Host", attribution="confident"),
                ]
            )
            return s

    monkeypatch.setattr(tmod, "TranscriptionPipeline", FakePipeline)
    lines = youtube.voice_transcriber(rconfig, "jane")("https://youtu.be/x", tmp_path)
    assert lines == [(1, 4, "mine and sure")]
    with TranscriptStore(rconfig.transcripts_db) as ts:
        assert ts.list_sessions() == []
