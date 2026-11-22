"""Smaller paths of the core modules: identification, packs, briefs, review, imports, the MCP surface."""

import json

import anyio
import numpy as np
import pytest

from conftest import ScriptedLLM, voice_vector
from kickvoice.research.brief import _validate, brief_topics, build_brief, build_topic, render_markdown
from kickvoice.research.store import Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.identify import SpeakerIdentifier
from kickvoice.speakers.packs import PackError, export_pack, import_pack
from kickvoice.speakers.profiles import ProfileStore

# ── classic identification ───────────────────────────────────────


def test_identifier_levels_batch_and_reload(config):
    ident = SpeakerIdentifier(config)
    none = ident.identify(voice_vector(1))
    assert (none.confidence, none.matched, none.best) == ("none", False, None)
    ProfileStore(config).add_profile("A", voice_vector(1))
    ident.reload_profiles()
    config.identify.high_confidence_threshold = 0.99
    medium = ident.identify(voice_vector(1) * 0.97 + voice_vector(2) * 0.2)
    assert medium.confidence == "medium" and medium.matched and medium.best[0] == "A"
    assert "MEDIUM" in str(medium)
    low = ident.identify(voice_vector(3))
    assert low.confidence == "low" and "UNKNOWN" in str(low)
    assert len(ident.identify_batch([voice_vector(1), voice_vector(3)])) == 2
    assert SpeakerIdentifier.cosine_similarity(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(1)


# ── profile packs ────────────────────────────────────────────────


def test_pack_error_paths(config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("A", voice_vector(1), embedding_model="other/model")
    with pytest.raises(PackError, match="was made with other/model"):
        export_pack(store, tmp_path / "p.json")
    with pytest.raises(PackError, match="on_conflict"):
        import_pack(store, tmp_path / "p.json", on_conflict="explode")
    with pytest.raises(PackError, match="Can't read"):
        import_pack(store, tmp_path / "missing.json")
    newer = tmp_path / "new.json"
    newer.write_text(json.dumps({"format": "kickvoice-profile-pack", "version": 99}))
    with pytest.raises(PackError, match="newer"):
        import_pack(store, newer)
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "format": "kickvoice-profile-pack",
                "version": 1,
                "embedding_model": config.embedding.model_source,
                "profiles": [{"name": "X", "embedding": {"not": "a list"}}],
            }
        )
    )
    with pytest.raises(PackError, match="Invalid profile entry"):
        import_pack(store, bad)


def test_pack_export_skips_profiles_without_embeddings(config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("A", voice_vector(1))
    store._embeddings.pop("A")  # e.g. the .npy went missing
    assert export_pack(store, tmp_path / "p.json") == 0


# ── briefs ───────────────────────────────────────────────────────


@pytest.fixture
def record(config):
    store = ResearchStore(config.research_db)
    p = store.add_person("Jane")
    d = Document(p.id, "x", "post", "1", "Carbon tax now.", "self", published_at="2023-01-01")
    store.upsert_document(d, [Chunk(d.text)])
    ((chunk, _),) = store.chunks_to_analyze(p.id)
    store.add_claims([Claim(p.id, d.id, chunk.id, "carbon tax", "supports it", "Carbon tax now")])
    yield store, p
    store.close()


def test_brief_topic_filter_and_failures(record, config):
    store, p = record
    assert brief_topics(store, p, "CARBON", 5) == ["carbon tax"]
    assert brief_topics(store, p, "zoning", 5) == ["zoning"]  # unknown topic kept as asked
    empty = build_topic(store, p, ScriptedLLM([]), config, "zoning")
    assert empty.error == "no claims on this topic yet"
    failed = build_topic(store, p, ScriptedLLM(["not json", "still not"]), config, "carbon tax")
    assert failed.error and failed.claims
    md = render_markdown(store, p, [empty, failed])
    assert "_no claims on this topic yet_" in md and "Analysis failed for this topic" in md
    assert "No topics yet" in render_markdown(store, p, [])
    with pytest.raises(ValueError):
        _validate(["not a dict"])
    assert build_brief(store, p, ScriptedLLM([{"summary": "s"}]), config).startswith("# Debate brief")


# ── review (extract-profiles without an agent) ───────────────────


def test_ffplay_player(tmp_path, monkeypatch):
    from kickvoice.pipelines import review

    monkeypatch.setattr(review.shutil, "which", lambda name: None)
    assert review.ffplay_player(tmp_path / "x.wav") is False
    calls = []
    monkeypatch.setattr(review.shutil, "which", lambda name: "/usr/bin/ffplay")
    monkeypatch.setattr(review.subprocess, "run", lambda cmd, check: calls.append(cmd))
    assert review.ffplay_player(tmp_path / "x.wav") is True and calls[0][0] == "ffplay"

    def interrupted(cmd, check):
        raise KeyboardInterrupt

    monkeypatch.setattr(review.subprocess, "run", interrupted)
    assert review.ffplay_player(tmp_path / "x.wav") is True  # Ctrl+C stops playback only


def test_review_merge_edge_cases(config):
    from conftest import ScriptedInput
    from kickvoice.pipelines.review import Reviewer
    from kickvoice.pipelines.session import ExtractionSession

    session = ExtractionSession.create(
        config.extractions_dir, "https://kick.com/x", config.embedding.model_source
    )
    session.add_candidate(
        embedding=voice_vector(5),
        sample_pcm=b"\x00\x00" * 100,
        sample_rate=16000,
        speech_seconds=30,
        n_windows=10,
        first_heard_s=0,
        cohesion=0.9,
    )
    session.save()
    out = []
    # m with no profiles → message; p with no player → path shown; then skip
    Reviewer(session, config, input_fn=ScriptedInput(["m", "p", "s"]), out=out.append, player=None).run()
    text = "\n".join(out)
    assert "No existing profiles to merge into" in text and "open this file to listen" in text

    store = ProfileStore(config)
    store.add_profile("A", voice_vector(6))
    store.add_profile("B", voice_vector(7))
    store.add_profile("C", voice_vector(8))
    session2 = ExtractionSession.create(
        config.extractions_dir, "https://kick.com/y", config.embedding.model_source
    )
    session2.add_candidate(
        embedding=voice_vector(6),
        sample_pcm=b"\x00\x00" * 100,
        sample_rate=16000,
        speech_seconds=30,
        n_windows=10,
        first_heard_s=0,
        cohesion=0.9,
    )
    session2.save()
    out2 = []
    Reviewer(
        session2,
        config,
        store=store,
        input_fn=ScriptedInput(["n", "", "m", "", "s"]),
        out=out2.append,
        player=None,
    ).run()
    assert "Next closest" in "\n".join(out2)


# ── imports ──────────────────────────────────────────────────────


def test_import_json_and_csv(config, tmp_path):
    from kickvoice.research.net import Fetcher
    from kickvoice.research.sources.base import SourceContext
    from kickvoice.research.sources.local import collect_transcripts, import_file

    with ResearchStore(config.research_db) as store:
        p = store.add_person("Jane")
        ctx = SourceContext(config, Fetcher(config), store, p)
        assert collect_transcripts(ctx, "Jane").total == 0  # no transcripts database yet
        (tmp_path / "one.json").write_text(json.dumps({"text": "a single post", "id": "j1"}))
        (tmp_path / "many.json").write_text(json.dumps([{"text": "first"}, {"text": "second"}]))
        (tmp_path / "posts.csv").write_text("id,text,date\nc1,from a csv,2024-01-01\n")
        assert import_file(ctx, tmp_path / "one.json").new == 1
        assert import_file(ctx, tmp_path / "many.json").new == 2
        assert import_file(ctx, tmp_path / "posts.csv").new == 1
        assert {d.external_id for d in store.documents(p.id)} >= {"j1", "c1"}


# ── every MCP tool is callable over the protocol ─────────────────


def test_every_mcp_tool_answers_over_the_protocol(rconfig, monkeypatch):
    """Call each tool through a real MCP client; none may crash (errors must be readable)."""
    from mcp import Client

    from kickvoice.agent.server import build_server
    from kickvoice.agent.service import AgentService

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with ResearchStore(rconfig.research_db) as store:
        store.add_person("Jane")
    service = AgentService(rconfig, clipboard=lambda t: (True, "fake"))
    calls = {
        "list_live_sessions": {},
        "research_subjects": {},
        "research_profile": {"name": "Jane"},
        "research_search": {"name": "Jane", "query": "tax"},
        "research_add_subject": {"name": "Jane", "accounts": {"x": "jane"}},
        "research_find_identity": {"name": "Jane", "handles": {"discord": "x"}},
        "research_link_wikidata": {"name": "Jane", "wikidata": "Q0"},
        "research_collect": {"name": "Nobody"},
        "research_analyze": {"name": "Jane"},
        "job_status": {"job_id": "nope"},
        "get_point_evidence": {"text": "carbon tax is bad", "opponent": "Jane"},
        "suggest_counters": {"text": "carbon tax is bad", "opponent": "Jane"},
        "verify_message": {"opponent": "Jane", "message": "who pays?"},
        "format_for_chat": {"text": "**hi**"},
        "copy_to_clipboard": {"text": "hi"},
        "get_live_updates": {"session_id": "nope"},
        "list_voices": {"session_id": "nope"},
        "enroll_voice": {"session_id": "nope", "utterance_id": 1, "name": "x"},
        "export_voice_clip": {"session_id": "nope", "utterance_id": 1},
        "set_opponent": {"session_id": "nope"},
        "stop_live_session": {"session_id": "nope"},
        "start_live_session": {"source": "/no/such/file.wav"},
    }

    async def main():
        results = {}
        async with Client(build_server(service), cache=None) as client:
            listed = {t.name for t in (await client.list_tools()).tools}
            assert listed == set(calls), listed ^ set(calls)  # the table covers every tool
            for name, args in calls.items():
                r = await client.call_tool(name, args)
                results[name] = (r.is_error, r.content[0].text if r.content else "")
        return results

    results = anyio.run(main)
    service.close()
    for name, (is_error, text) in results.items():
        assert text, name
        if is_error:
            # Anticipated failures carry a readable reason. A crash would surface as the
            # SDK's bare "Error executing tool <name>" with nothing after it.
            prefix = f"Error executing tool {name}"
            reason = text[len(prefix) :].lstrip(": ") if text.startswith(prefix) else text
            assert len(reason) > 10, (name, text)
    assert results["format_for_chat"] == (False, '{\n  "text": "hi",\n  "length": 2\n}')
    assert "No live session 'nope'" in results["get_live_updates"][1]
    assert "Unknown platform" in results["research_find_identity"][1]
    assert not results["research_add_subject"][0] and not results["copy_to_clipboard"][0]
