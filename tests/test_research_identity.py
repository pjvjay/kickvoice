"""
Identity linking must never attach someone else's accounts to a subject.

Regression: `discover "Destiny" --x <handle> -y` where the handle wasn't on
Wikidata fell back to a name search, took the first hit — the 2014 video
game — and confirmed its accounts (@DestinyTheGame) for the debater.
"""

import json
import sqlite3

import pytest

from kickvoice.pipelines.research import discover
from kickvoice.research.net import Fetcher
from kickvoice.research.store import (
    CONFIRMED,
    REJECTED,
    Chunk,
    Claim,
    Document,
    ResearchStore,
)


def entity(label, description, instance_of, accounts=None, enwiki=None):
    claims = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": q}}}} for q in instance_of]}
    for prop, value in (accounts or {}).items():
        claims[prop] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": value}}}]
    e = {
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": description}},
        "claims": claims,
        "sitelinks": {},
    }
    if enwiki:
        e["sitelinks"]["enwiki"] = {"title": enwiki}
    return e


GAME = entity(
    "Destiny", "2014 video game", ["Q7889"], {"P2002": "DestinyTheGame"}, enwiki="Destiny (video game)"
)
STREAMER = entity(
    "Steven Bonnell",
    "American streamer and political commentator",
    ["Q5"],
    {"P2002": "TheOmniLiberal", "P2397": "UC" + "d" * 22},
    enwiki="Destiny (streamer)",
)
COMPANY = entity("Bungie", "American video game developer", ["Q210167"], {"P2002": "Bungie"})

WIKIDATA = {"Q1": GAME, "Q2": STREAMER, "Q3": COMPANY}


@pytest.fixture
def wikidata(fake_web):
    """Fake Wikidata: which records list which handles, and a name search."""
    state = {
        "name_hits": ["Q1", "Q2"],
        "handles": {"P2002=DestinyTheGame": ["Q1"], "P2002=TheOmniLiberal": ["Q2"]},
    }

    def api(req):
        q = req["query"]
        if q.get("action") == "wbsearchentities":
            return (
                200,
                {},
                {
                    "search": [
                        {"id": i, "label": WIKIDATA[i]["labels"]["en"]["value"]} for i in state["name_hits"]
                    ]
                },
            )
        if q.get("list") == "search":
            key = q["srsearch"].removeprefix("haswbstatement:")
            return 200, {}, {"query": {"search": [{"title": t} for t in state["handles"].get(key, [])]}}
        if q.get("action") == "wbgetentities":
            return 200, {}, {"entities": {i: WIKIDATA.get(i, {"missing": ""}) for i in q["ids"].split("|")}}
        return 400, {}, {}

    fake_web.route("GET", "/wd/api.php", api)
    return state


def person_state(rconfig, name="Destiny"):
    with ResearchStore(rconfig.research_db) as store:
        p = store.get_person(name)
        if p is None:
            return None, []
        return p, [(a.platform, a.handle, a.status) for a in store.accounts(p.id)]


# ── the reported bug ─────────────────────────────────────────────


def test_unmatched_handle_never_falls_back_to_first_name_hit(run, rconfig, wikidata):
    """Exactly the report: handle not on Wikidata, -y, name search → the game."""
    wikidata["name_hits"] = ["Q1"]  # only the video game matches the name
    out = run("research", "discover", "Destiny", "--x", "not_on_wikidata", "-y").output
    assert "No Wikidata record lists x:not_on_wikidata" in out
    assert 'Skipped Q1 "Destiny" — 2014 video game (not a person)' in out
    assert "No person matching" in out
    person, accounts = person_state(rconfig)
    assert accounts == [] and (person is None or person.description == "")


def test_yes_never_links_on_a_name_match(run, rconfig, wikidata):
    """Even when the name search finds a real person, -y must not pick for you."""
    result = run("research", "discover", "Destiny", "--x", "not_on_wikidata", "-y", expect_ok=False)
    assert result.exit_code == 1
    assert "name search alone isn't enough" in result.output
    assert "--wikidata Q2" in result.output  # tells you how to be exact
    assert "Skipped Q1" in result.output  # the game isn't even offered
    assert "[1] Steven Bonnell (Q2)" in result.output and "name match only" in result.output
    assert person_state(rconfig)[1] == []


def test_exact_handle_of_a_non_person_is_rejected(run, rconfig, wikidata):
    """--x DestinyTheGame matches exactly — but it's a game, not a person."""
    result = run("research", "discover", "Destiny", "--x", "DestinyTheGame", "-y", expect_ok=False)
    assert "(not a person)" in result.output
    assert "DestinyTheGame" not in json.dumps(person_state(rconfig)[1])


def test_exact_handle_of_a_person_links_with_yes(run, rconfig, wikidata):
    out = run("research", "discover", "Destiny", "--x", "TheOmniLiberal", "-y").output
    assert "Matched Steven Bonnell (Q2)" in out and "via x:TheOmniLiberal" in out
    person, accounts = person_state(rconfig)
    assert person.wikidata_id == "Q2" and "political commentator" in person.description
    assert ("x", "TheOmniLiberal", CONFIRMED) in accounts
    assert ("wikipedia", "Destiny (streamer)", CONFIRMED) in accounts


def test_explicit_wikidata_id(run, rconfig, wikidata):
    run("research", "discover", "Destiny", "--wikidata", "q2", "-y")
    assert person_state(rconfig)[0].wikidata_id == "Q2"
    result = run("research", "discover", "Destiny", "--wikidata", "Q3", "-y")  # a company
    assert "(not a person)" in result.output and person_state(rconfig)[0].wikidata_id == "Q2"


def test_interactive_choice_from_name_search(run, rconfig, wikidata, monkeypatch):
    import kickvoice.cli_research as cr

    monkeypatch.setattr(cr, "_interactive", lambda: True)
    # pick [1] (the only person), accept x, reject youtube, accept wikipedia
    out = run("research", "discover", "Destiny", input="1\ny\ny\nn\n").output
    assert "Which one is them?" in out and "Destiny (video game)" not in out
    accounts = dict(((p, h), st) for p, h, st in person_state(rconfig)[1])
    assert accounts[("x", "TheOmniLiberal")] == CONFIRMED
    assert accounts[("youtube", "UC" + "d" * 22)] == REJECTED


def test_interactive_can_decline_everything(run, rconfig, wikidata, monkeypatch):
    import kickvoice.cli_research as cr

    monkeypatch.setattr(cr, "_interactive", lambda: True)
    assert "Nothing linked" in run("research", "discover", "Destiny", input="0\n").output
    assert (
        "Nothing linked"
        in run("research", "discover", "Destiny", "--x", "TheOmniLiberal", input="n\n").output
    )
    assert person_state(rconfig)[1] == []


def test_discover_result_fields(rconfig, wikidata):
    r = discover(Fetcher(rconfig), rconfig, "Destiny", {"x": "nope", "reddit": "nope2"})
    assert r.unmatched_handles == ["x:nope", "reddit:nope2"] and r.searched_by_name
    assert [s.qid for s in r.skipped] == ["Q1"]
    assert [(c.record.qid, c.strong) for c in r.candidates] == [("Q2", False)]


# ── cleaning up after a wrong link ───────────────────────────────


@pytest.fixture
def polluted(rconfig):
    """The state the bug left behind: linked to the game, its accounts confirmed,
    and content already collected (before provenance existed → account_id NULL)."""
    with ResearchStore(rconfig.research_db) as store:
        p = store.add_person("Destiny", description="2014 video game")
        p.wikidata_id = "Q1"
        store.update_person(p)
        store.add_account(
            p.id,
            "x",
            "DestinyTheGame",
            "https://x.com/DestinyTheGame",
            status=CONFIRMED,
            found_via="wikidata Q1",
        )
        store.add_account(
            p.id, "wikipedia", "Destiny (video game)", status=CONFIRMED, found_via="wikidata Q1"
        )
        mine = store.add_account(p.id, "x", "TheOmniLiberal", status=CONFIRMED, found_via="added by you")
        docs = [
            Document(
                p.id,
                "x",
                "post",
                "g1",
                "New raid drops Tuesday, Guardians!",
                "self",
                url="https://x.com/DestinyTheGame/status/g1",
            ),
            Document(
                p.id,
                "wikipedia",
                "article",
                "Destiny (video game)",
                "Destiny is a 2014 online first-person shooter.",
                "about",
            ),
            Document(
                p.id,
                "x",
                "post",
                "s1",
                "Rent control reduces housing supply.",
                "self",
                url="https://x.com/TheOmniLiberal/status/s1",
                account_id=mine.id,
            ),
        ]
        for d in docs:
            store.upsert_document(d, [Chunk(d.text)])
        for chunk, doc in store.chunks_to_analyze(p.id):
            store.add_claims([Claim(p.id, doc.id, chunk.id, "topic " + doc.external_id, "claim", doc.text)])
    return rconfig


def test_relink_rejects_old_records_accounts_and_fixes_description(run, polluted, wikidata):
    out = run("research", "discover", "Destiny", "--wikidata", "Q2", "-y").output
    assert "Re-linked Destiny from Q1 to Q2" in out
    assert "x:DestinyTheGame" in out and "--purge" in out
    person, accounts = person_state(polluted)
    assert person.description == "American streamer and political commentator"
    status = {(p, h): st for p, h, st in accounts}
    assert status[("x", "DestinyTheGame")] == REJECTED
    assert status[("wikipedia", "Destiny (video game)")] == REJECTED
    assert status[("x", "TheOmniLiberal")] == CONFIRMED  # yours — untouched


def test_purge_removes_everything_from_rejected_accounts(run, polluted, wikidata):
    run("research", "discover", "Destiny", "--wikidata", "Q2", "-y")
    shown = run("research", "show", "Destiny").output
    assert "2 document(s) came from rejected accounts" in shown
    out = run("research", "accounts", "Destiny", "--purge").output
    assert "Deleted 2 document(s)" in out
    with ResearchStore(polluted.research_db) as store:
        p = store.get_person("Destiny")
        assert [d.external_id for d in store.documents(p.id)] == ["s1"]
        assert [c.quote for c in store.claims(p.id)] == ["Rent control reduces housing supply."]
        assert store.search_chunks(p.id, "raid Guardians") == []
    assert "rejected accounts" not in run("research", "show", "Destiny").output


def test_unlink(run, polluted):
    out = run("research", "unlink", "Destiny", "--purge").output
    assert "Unlinked Destiny from Q1; rejected 2 account(s), deleted 2 document(s)" in out
    person, accounts = person_state(polluted)
    assert person.wikidata_id is None and person.description == ""
    assert ("x", "TheOmniLiberal", CONFIRMED) in accounts
    assert "isn't linked" in run("research", "unlink", "Destiny").output


def test_rejecting_an_account_takes_effect_before_purge(rconfig):
    """Documents with provenance are ignored as soon as their account is rejected."""
    with ResearchStore(rconfig.research_db) as store:
        p = store.add_person("Jane")
        acct = store.add_account(p.id, "x", "wrong", status=CONFIRMED)
        d = Document(p.id, "x", "post", "1", "raid night tonight guardians", "self", account_id=acct.id)
        store.upsert_document(d, [Chunk(d.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        store.add_claims([Claim(p.id, d.id, chunk.id, "raids", "likes raids", d.text)])
        assert store.search_claims(p.id, "raid") and store.topics(p.id)
        store.set_account_status(acct.id, REJECTED)
        assert store.search_claims(p.id, "raid") == []
        assert store.search_chunks(p.id, "raid") == []
        assert store.claims(p.id) == [] and store.topics(p.id) == []
        assert store.documents_from_account(store.get_account(acct.id)) == [d.id]


def test_collect_records_which_account_each_document_came_from(rconfig, fake_web, monkeypatch):
    from kickvoice.pipelines.research import collect
    from test_research_sources import x_routes

    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(fake_web, {None: {"data": [{"id": "1", "text": "hello"}], "meta": {}}})
    with ResearchStore(rconfig.research_db) as store:
        p = store.add_person("Jane")
        acct = store.add_account(p.id, "x", "JanePundit", status=CONFIRMED)
        collect(store, p, rconfig)
        assert [d.account_id for d in store.documents(p.id)] == [acct.id]


def test_old_database_gets_account_column(rconfig):
    """research.db files created before provenance existed are migrated in place."""
    path = rconfig.research_db
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE persons (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE
            COLLATE NOCASE, voice_profile TEXT, wikidata_id TEXT, description TEXT NOT NULL
            DEFAULT '', created_at TEXT NOT NULL);
        CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL,
            source TEXT NOT NULL, kind TEXT NOT NULL, external_id TEXT NOT NULL,
            url TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '', published_at TEXT,
            authorship TEXT NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL, meta TEXT,
            fetched_at TEXT NOT NULL, UNIQUE (person_id, source, external_id));
        INSERT INTO persons (name, created_at) VALUES ('Old', '2025-01-01');
        INSERT INTO documents (person_id, source, kind, external_id, authorship, text,
            content_hash, fetched_at) VALUES (1, 'x', 'post', '1', 'self', 'old text', 'h', 'now');
    """)
    db.close()
    with ResearchStore(path) as store:
        (doc,) = store.documents(store.get_person("Old").id)
        assert doc.text == "old text" and doc.account_id is None


def test_legacy_matching_rules():
    from kickvoice.research.store import Account, _legacy_match

    def row(source, external_id="", url="", meta=None):
        return {"source": source, "external_id": external_id, "url": url, "meta": json.dumps(meta or {})}

    x = Account(1, 1, "x", "@DestinyTheGame")
    assert _legacy_match(x, row("x", url="https://x.com/destinythegame/status/9"))
    assert not _legacy_match(x, row("x", url="https://x.com/DestinyTheGameFan/status/9"))
    wp = Account(2, 1, "wikipedia", "Destiny (video game)")
    assert _legacy_match(wp, row("wikipedia", "Destiny_(video_game)"))
    yt = Account(3, 1, "youtube", "UCabc")
    assert _legacy_match(yt, row("youtube", meta={"channel_id": "UCabc", "own_channel": True}))
    assert not _legacy_match(yt, row("youtube", meta={"channel_id": "UCabc", "own_channel": False}))
    site = Account(4, 1, "website", "https://game.example", url="https://game.example")
    assert _legacy_match(site, row("web", "https://game.example/news/1"))
