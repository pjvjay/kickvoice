"""SQLite transcript store."""

import numpy as np
import pytest

from kickvoice.transcripts.attribution import CONFIDENT, PROBABLE, UNKNOWN
from kickvoice.transcripts.store import TranscriptStore, Utterance


@pytest.fixture
def store(tmp_path):
    with TranscriptStore(tmp_path / "t.db") as s:
        yield s


def utt(session, start, speaker, level, text, emb=None):
    return Utterance(
        session_id=session,
        start_s=start,
        end_s=start + 3,
        text=text,
        speaker=speaker,
        similarity=0.8,
        attribution=level,
        asr_confidence=0.9,
        embedding=emb,
    )


@pytest.fixture
def populated(store):
    s1 = store.create_session(
        "20250101-120000_chan", "https://kick.com/chan", asr_model="asr", embedding_model="ecapa"
    )
    s2 = store.create_session("20250102-120000_chan", "https://kick.com/chan")
    store.add_utterances(
        [
            utt(
                s1.id,
                0,
                "Host",
                CONFIDENT,
                "I don't think gambling sponsors are fine",
                emb=np.arange(4, dtype=float),
            ),
            utt(s1.id, 5, "Guest", PROBABLE, "gambling is just entertainment"),
            utt(s1.id, 9, None, UNKNOWN, "chat is going crazy"),
            utt(s2.id, 0, "Host", CONFIDENT, "today we play chess"),
        ]
    )
    return store, s1, s2


def test_roundtrip_and_order(populated):
    store, s1, _ = populated
    rows = store.utterances(s1.id)
    assert [u.start_s for u in rows] == [0, 5, 9]
    assert rows[0].text.startswith("I don't") and rows[0].session_id == s1.id
    assert np.allclose(rows[0].embedding, np.arange(4))
    assert rows[1].embedding is None


def test_filters(populated):
    store, s1, _ = populated
    assert len(store.utterances(speaker="host")) == 2  # case-insensitive
    assert len(store.utterances(s1.id, min_attribution=CONFIDENT)) == 1
    assert len(store.utterances(s1.id, min_attribution=PROBABLE)) == 2
    assert len(store.utterances(s1.id, min_attribution=UNKNOWN)) == 3
    with pytest.raises(ValueError):
        store.utterances(min_attribution="maybe")


def test_search(populated):
    store, _, _ = populated
    hits = store.search("gambling")
    assert {u.speaker for u in hits} == {"Host", "Guest"}
    assert [u.speaker for u in store.search("gambling", speaker="Host")] == ["Host"]
    assert store.search("gambling", min_attribution=CONFIDENT)[0].speaker == "Host"
    # punctuation / FTS operators in user input must not break the query
    assert len(store.search("don't")) == 1
    assert [u.text for u in store.search('chess"')] == ["today we play chess"]
    assert store.search("chess AND OR (") == []  # operators are literal terms
    assert store.search("nonexistentword") == []


def test_search_without_fts(tmp_path, monkeypatch):
    with TranscriptStore(tmp_path / "t.db") as store:
        store.has_fts = False
        s = store.create_session("x", "src")
        store.add_utterances([utt(s.id, 0, "A", CONFIDENT, "hello world")])
        assert len(store.search("world")) == 1


def test_sessions_listing_and_lookup(populated):
    store, s1, s2 = populated
    listed = store.list_sessions()
    assert [s.id for s in listed] == [s2.id, s1.id]  # newest first
    assert listed[1].n_utterances == 3 and listed[1].speakers == ["Guest", "Host"]
    assert store.get_session(s1.id).asr_model == "asr"
    assert store.get_session("20250101").id == s1.id  # unique prefix
    assert store.get_session("2025") is None  # ambiguous
    assert store.get_session("nope") is None


def test_duplicate_session_ids_get_suffix(store):
    a = store.create_session("same", "src")
    b = store.create_session("same", "src")
    assert (a.id, b.id) == ("same", "same-2")


def test_finish_session(populated):
    store, s1, _ = populated
    store.finish_session(s1.id, 123.4)
    got = store.get_session(s1.id)
    assert got.captured_seconds == 123.4 and got.ended_at


def test_delete_cascades_to_utterances_and_search(populated):
    store, s1, _ = populated
    assert store.delete_session(s1.id)
    assert store.utterances(s1.id) == []
    assert store.search("gambling") == []
    assert not store.delete_session(s1.id)


def test_speaker_summary(populated):
    store, _, _ = populated
    rows = {r["speaker"]: r for r in store.speaker_summary()}
    assert rows["Host"]["utterances"] == 2 and rows["Host"]["sessions"] == 2
    assert rows["Host"]["confident_seconds"] == 6
    assert rows["Guest"]["confident_seconds"] == 0
    assert None in rows


def test_update_and_delete_utterances(populated):
    store, s1, _ = populated
    first, second, third = store.utterances(s1.id)
    store.update_attributions([(second.id, "Renamed", 0.91, CONFIDENT, 2)])
    after = store.utterances(s1.id)[1]
    assert (after.speaker, after.attribution, after.voice_group) == ("Renamed", CONFIDENT, 2)
    assert after.similarity == pytest.approx(0.91)
    store.delete_utterances([third.id])
    assert [u.id for u in store.utterances(s1.id)] == [first.id, second.id]
    assert store.search("crazy") == []  # search index kept in sync


def test_persists_across_connections(tmp_path):
    path = tmp_path / "t.db"
    with TranscriptStore(path) as store:
        s = store.create_session("x", "src")
        store.add_utterances([utt(s.id, 0, "A", CONFIDENT, "hello")])
    with TranscriptStore(path) as store:
        assert store.utterances()[0].text == "hello"
        assert len(store.search("hello")) == 1


def test_to_dict_includes_wall_clock(store):
    from datetime import datetime

    s = store.create_session("x", "https://kick.com/chan", started_at=datetime(2025, 1, 1, 20, 0, 0))
    store.add_utterances([utt(s.id, 65, "Guest", PROBABLE, "hi")])
    d = store.utterances(s.id)[0].to_dict(store.get_session(s.id))
    assert d["speaker"] == "Guest" and d["source"] == "https://kick.com/chan"
    assert d["said_at"] == "2025-01-01T20:01:05"
