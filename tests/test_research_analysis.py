"""Store, claim extraction (quote verification), retrieval, live counters, briefs."""
import re

import pytest

from conftest import ScriptedLLM
from kickvoice.research.brief import build_brief
from kickvoice.research.counter import CounterEngine, load_notes, relevant_notes
from kickvoice.research.extract import extract_claims, find_quote, make_batches
from kickvoice.research.retrieve import Retriever, snippet
from kickvoice.research.store import (
    ABOUT, CANDIDATE, CONFIRMED, SELF, SELF_CHANNEL, Chunk, Claim, Document, ResearchStore,
    deep_link, fts_or_query,
)


@pytest.fixture
def store(config):
    with ResearchStore(config.research_db) as s:
        yield s


@pytest.fixture
def person(store):
    return store.add_person("Jane Pundit", description="political streamer")


def claim_starting(claims: dict, prefix: str):
    (match,) = [c for q, c in claims.items() if q.startswith(prefix)]
    return match


def add_doc(store, person, ext, text, authorship=SELF, source="x", url="", published="2024-01-01",
            chunks=None, kind="post"):
    doc = Document(person_id=person.id, source=source, kind=kind, external_id=ext, text=text,
                   authorship=authorship, url=url, published_at=published)
    store.upsert_document(doc, chunks if chunks is not None else [Chunk(text)])
    return doc


# ── store ────────────────────────────────────────────────────────

def test_person_names_are_case_insensitive(store, person):
    assert store.get_person("jane pundit").id == person.id
    with pytest.raises(Exception):
        store.add_person("JANE PUNDIT")


def test_confirmed_account_is_never_downgraded(store, person):
    a = store.add_account(person.id, "x", "jane", status=CONFIRMED)
    b = store.add_account(person.id, "x", "jane", status=CANDIDATE)   # rediscovered later
    assert a.id == b.id and b.status == CONFIRMED


def test_upsert_document_new_unchanged_updated(store, person):
    doc = Document(person_id=person.id, source="x", kind="post", external_id="1",
                   text="old text", authorship=SELF)
    _, outcome = store.upsert_document(doc, [Chunk("old text")])
    assert outcome == "new"
    chunk_id = store.chunks_to_analyze(person.id)[0][0].id
    store.add_claims([Claim(person.id, doc.id, chunk_id, "t", "c", "old text quote")])
    assert store.upsert_document(doc, [Chunk("old text")])[1] == "unchanged"

    doc.text = "edited text"
    assert store.upsert_document(doc, [Chunk("edited text")])[1] == "updated"
    assert [c.text for c, _ in store.chunks_to_analyze(person.id)] == ["edited text"]
    assert store.claims(person.id) == []                   # stale claims removed with old chunks


def test_chunks_to_analyze_skips_analyzed(store, person):
    add_doc(store, person, "1", "one")
    add_doc(store, person, "2", "two")
    first = store.chunks_to_analyze(person.id)
    store.mark_analyzed([first[0][0].id], "m")
    assert len(store.chunks_to_analyze(person.id)) == 1


def test_fts_or_query_and_deep_link():
    assert fts_or_query("I think the minimum wage is too low!") == '"minimum" OR "wage" OR "low"'
    assert fts_or_query("uh yeah ok") == ""
    assert deep_link("https://www.youtube.com/watch?v=abc", 65.7) == \
        "https://www.youtube.com/watch?v=abc&t=65s"
    assert deep_link("https://x.com/a/status/1", 5) == "https://x.com/a/status/1"


def test_remove_person_cascades(store, person):
    add_doc(store, person, "1", "text")
    store.remove_person(person.id)
    assert store.list_persons() == [] and store.documents(person.id) == []


# ── quote verification ───────────────────────────────────────────

SRC = ("Look, I've said this a hundred times. The minimum wage should be tied to inflation, "
       "period. Anyone who disagrees hasn't read the data.")


@pytest.mark.parametrize("quote,expected", [
    ("The minimum wage should be tied to inflation, period.",
     "The minimum wage should be tied to inflation, period"),
    ("the minimum wage should be tied to inflation period",
     "The minimum wage should be tied to inflation, period"),
    ("The minimum wage must be tied to inflation, period.",          # one word "tidied"
     "The minimum wage should be tied to inflation, period"),          # → source's words
    ("He thinks wages should follow inflation levels", None),         # paraphrase
    ("Raising the minimum wage destroys small businesses", None),     # fabricated
    ("period", None),                                                 # too short
])
def test_find_quote(quote, expected):
    assert find_quote(quote, SRC) == expected


# ── extraction ───────────────────────────────────────────────────

def extraction_llm(extra_items=None):
    """Replies with one good claim per chunk, plus a fabricated and a bad-ref claim."""
    def fn(system, prompt):
        refs = re.findall(r"\[(K\d+)\]", prompt)
        items = []
        for ref in refs:
            body = prompt.split(f"[{ref}]")[1].split("\n", 2)[1]
            words = body.split()
            items.append({"chunk": ref, "topic": "Housing", "claim": "paraphrase",
                          "quote": " ".join(words[:8]), "kind": "position",
                          "sincere": "lol" not in body})
        items.append({"chunk": refs[0], "topic": "made up", "claim": "x",
                      "quote": "this sentence does not appear anywhere at all"})
        items.append({"chunk": "K99999", "topic": "ghost", "claim": "x", "quote": "whatever it is"})
        return {"claims": items + (extra_items or [])}
    return ScriptedLLM(fn=fn)


def test_extract_claims_keeps_only_verified_quotes(store, person, config):
    yt = add_doc(store, person, "vid:captions", "…", authorship=SELF_CHANNEL, source="youtube",
                 url="https://www.youtube.com/watch?v=abc",
                 chunks=[Chunk("rent control always reduces housing supply over time and everyone knows it", 95.0, 110.0)])
    add_doc(store, person, "t1", "lol rent control is my favourite policy ever said no economist")
    llm = extraction_llm()
    stats = extract_claims(store, person, llm, config)

    assert stats.claims == 2 and stats.rejected_quotes == 2 and stats.failed_batches == 0
    claims = {c.quote: c for c in store.claims(person.id)}
    yt_claim = claim_starting(claims, "rent control always reduces")
    assert yt_claim.topic == "housing" and yt_claim.authorship == SELF_CHANNEL
    assert yt_claim.url == "https://www.youtube.com/watch?v=abc&t=95s"   # deep link to the moment
    assert yt_claim.document_id == yt.id
    joke = [c for c in claims.values() if c.quote.startswith("lol")][0]
    assert joke.sincere is False
    assert store.chunks_to_analyze(person.id) == []
    assert "Subject: Jane Pundit (political streamer)" in llm.calls[0]["prompt"]
    assert "(their channel (speaker unverified)" in llm.calls[0]["prompt"]


def test_failed_batch_is_retried_next_run(store, person, config):
    add_doc(store, person, "1", "some text worth analysing about taxes and spending")
    stats = extract_claims(store, person, ScriptedLLM(["nope", "still nope"]), config)
    assert stats.failed_batches == 1 and stats.claims == 0
    assert len(store.chunks_to_analyze(person.id)) == 1        # not marked analysed


def test_batches_respect_size(store, person):
    for i in range(5):
        add_doc(store, person, str(i), "x" * 500)
    batches = make_batches(store.chunks_to_analyze(person.id), max_chars=1200)
    assert [len(b) for b in batches] == [2, 2, 1]


# ── retrieval ────────────────────────────────────────────────────

@pytest.fixture
def record(store, person):
    """A small record: tweets, a video with a timed chunk, and a Wikipedia bit."""
    docs = {
        "t1": add_doc(store, person, "t1", "Rent control reduces housing supply. Build more homes.",
                      published="2021-03-01", url="https://x.com/j/status/t1"),
        "t2": add_doc(store, person, "t2", "Honestly rent control is fine in the short term for tenants.",
                      published="2024-06-01", url="https://x.com/j/status/t2"),
        "t3": add_doc(store, person, "t3", "lol rent control is the best policy ever, trust me bro",
                      published="2024-07-01"),
        "w": add_doc(store, person, "wiki", "[Views] She has opposed rent control.", authorship=ABOUT,
                     source="wikipedia", url="https://en.wikipedia.org/wiki/Jane"),
        "tax": add_doc(store, person, "t4", "A carbon tax with a dividend is the cleanest climate policy.",
                       published="2023-01-01"),
    }
    by_doc = {doc.id: c for c, doc in store.chunks_to_analyze(person.id)}
    claims = [
        Claim(person.id, docs["t1"].id, by_doc[docs["t1"].id].id, "rent control",
              "Rent control reduces housing supply", "Rent control reduces housing supply",
              authorship=SELF, published_at="2021-03-01", url="https://x.com/j/status/t1"),
        Claim(person.id, docs["t2"].id, by_doc[docs["t2"].id].id, "rent control",
              "Rent control is acceptable short term", "rent control is fine in the short term",
              authorship=SELF, published_at="2024-06-01", url="https://x.com/j/status/t2"),
        Claim(person.id, docs["t3"].id, by_doc[docs["t3"].id].id, "rent control",
              "Rent control is the best policy (joking)", "rent control is the best policy ever",
              authorship=SELF, sincere=False, published_at="2024-07-01"),
        Claim(person.id, docs["w"].id, by_doc[docs["w"].id].id, "rent control",
              "Reportedly opposes rent control", "She has opposed rent control",
              authorship=ABOUT, url="https://en.wikipedia.org/wiki/Jane"),
        Claim(person.id, docs["tax"].id, by_doc[docs["tax"].id].id, "carbon tax",
              "Supports a carbon tax and dividend", "A carbon tax with a dividend is the cleanest",
              authorship=SELF, published_at="2023-01-01"),
    ]
    store.add_claims(claims)
    store.mark_analyzed([c.chunk_id for c in claims], "m")
    return {c.quote: c for c in claims}


def test_retriever_finds_claims_first(store, person, record):
    hits = Retriever(store, person).search("rent control hurts housing supply", k=6)
    assert hits[0].kind == "claim"
    assert hits[0].quote == "Rent control reduces housing supply"
    assert all("carbon" not in h.text for h in hits[:4])
    assert {h.kind for h in hits} == {"claim", "chunk"}


def test_snippet_centres_on_query_terms():
    text = ("filler " * 100) + "the carbon tax dividend is key " + ("filler " * 100)
    s = snippet(text, "carbon dividend", width=80)
    assert "carbon tax dividend" in s and s.startswith("… ") and len(s) < 120


# ── live counters ────────────────────────────────────────────────

def counter_engine(store, person, config, reply, notes=None):
    return CounterEngine(store, person, ScriptedLLM([reply]), config, notes=notes)


def test_counter_validation(store, person, config, record, tmp_path):
    own_old = claim_starting(record, "Rent control reduces")
    wiki = claim_starting(record, "She has opposed")
    joke = claim_starting(record, "rent control is the best")
    notes_file = tmp_path / "prep.md"
    notes_file.write_text("Rent control studies: San Francisco 2019 study found supply fell 15%.\n\n"
                          "Unrelated note about football.")
    notes = load_notes(notes_file)
    reply = {
        "is_argument": True, "their_point": "Rent control protects tenants",
        "counters": [
            {"text": "You said it reduces supply.", "evidence": [f"C{own_old.id}"]},
            {"text": "The SF study says supply fell.", "evidence": ["N1"]},
            {"text": "Invented citation.", "evidence": ["C99999"]},          # dropped
            {"text": "Pure reasoning.", "evidence": []},
        ],
        "contradictions": [
            {"evidence": f"C{own_old.id}", "explanation": "In 2021 you said the opposite."},
            {"evidence": f"C{wiki.id}", "explanation": "Wikipedia says..."},  # not first-hand
            {"evidence": f"C{joke.id}", "explanation": "You said best policy"},  # insincere
        ],
    }
    s = counter_engine(store, person, config, reply, notes).suggest(
        "rent control protects tenants from greedy landlords and keeps housing affordable",
        ["Me: rent control is bad"])

    assert s.is_argument and s.summary == "Rent control protects tenants"
    assert [c.text for c in s.counters] == ["You said it reduces supply.",
                                           "The SF study says supply fell.", "Pure reasoning."]
    assert s.counters[0].evidence[0].quote == "Rent control reduces housing supply"  # from the DB
    assert s.counters[1].notes[0].id == 1 and not s.counters[2].sourced
    assert len(s.contradictions) == 1
    assert s.contradictions[0].evidence.published_at == "2021-03-01"
    assert s.dropped == 3


def test_counter_prompt_marks_provenance(store, person, config, record):
    engine = counter_engine(store, person, config, {"is_argument": False, "counters": []})
    engine.suggest("rent control is good for tenants")
    prompt = engine.llm.calls[0]["prompt"]
    assert "reported about them" in prompt and "insincere" in prompt
    assert "their own words" in prompt
    assert engine.llm.calls[0]["model"] == config.llm.live_model


def test_counter_survives_model_failure(store, person, config, record):
    engine = CounterEngine(store, person, ScriptedLLM(["garbage", "garbage"]), config)
    s = engine.suggest("rent control is good")
    assert s.summary.startswith("(no suggestion") and not s.counters


def test_relevant_notes():
    notes = load_notes(None)
    assert notes == []
    from kickvoice.research.counter import Note
    notes = [Note(1, "Carbon tax dividend evidence from Canada"), Note(2, "Football scores")]
    assert [n.id for n in relevant_notes(notes, "the carbon tax is a scam")] == [1]


# ── brief ────────────────────────────────────────────────────────

def test_brief_validates_contradictions_and_cites_db_quotes(store, person, config, record):
    old = claim_starting(record, "Rent control reduces")
    new = claim_starting(record, "rent control is fine")
    wiki = claim_starting(record, "She has opposed")

    def fn(system, prompt):
        if "Topic: rent control" in prompt:
            return {"summary": "Mixed views on rent control.",
                    "contradictions": [{"a": f"C{new.id}", "b": f"C{old.id}", "explanation": "Changed view"},
                                       {"a": f"C{old.id}", "b": f"C{wiki.id}", "explanation": "vs wiki"}],
                    "expected_arguments": ["Tenants need protection"],
                    "counters": [{"text": "Supply matters", "evidence": [f"C{old.id}"]},
                                 {"text": "Bogus", "evidence": ["C424242"]}]}
        return {"summary": "Pro carbon tax.", "contradictions": [], "expected_arguments": [],
                "counters": []}

    md = build_brief(store, person, ScriptedLLM(fn=fn), config)
    assert "# Debate brief: Jane Pundit" in md and "## Rent Control" in md and "## Carbon Tax" in md
    assert "⚠ Changed view" in md and "vs wiki" not in md          # wiki isn't first-hand
    # in the contradiction: oldest statement first, verbatim DB quotes with dates
    contradiction = md.split("⚠ Changed view")[1].split("**")[0]
    assert contradiction.index('"Rent control reduces housing supply" (2021-03-01') < \
        contradiction.index('"rent control is fine in the short term" (2024-06-01')
    assert "Supply matters" in md and "Bogus" not in md
    assert "best policy ever" not in md                             # insincere claims excluded
    assert "(model summary)" in md
