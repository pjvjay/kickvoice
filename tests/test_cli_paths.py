"""User-facing CLI paths that aren't the happy path: usage errors, empty states, confirmations."""

import json

import numpy as np

from conftest import SR, ScriptedLLM, conversation, voice_vector
from kickvoice.audio.ffmpeg import write_wav
from kickvoice.research.store import Chunk, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore


def test_name_argument_rules(run):
    assert "not both" in run("profiles", "info", "A", "--name", "B", expect_ok=False).output
    assert "Missing NAME" in run("profiles", "info", expect_ok=False).output
    assert "Profile not found" in run("profiles", "remove", "nobody", "-y", expect_ok=False).output


def test_profiles_listing_details(run, rconfig):
    store = ProfileStore(rconfig)
    store.add_profile("A", voice_vector(1), source_urls=["u1", "u2", "u3"], speech_seconds=60)
    store.merge_into_profile("A", voice_vector(2), speech_seconds=10, note="more audio")
    assert "(+1 more)" in run("profiles", "list").output
    assert "Updated:" in run("profiles", "info", "A").output


def test_pack_commands_errors(run, rconfig, tmp_path):
    assert (
        "not found"
        in run(
            "profiles", "export", "--out", str(tmp_path / "p.json"), "--name", "Zed", expect_ok=False
        ).output
    )
    assert "Give a pack file" in run("profiles", "import", expect_ok=False).output
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "format": "kickvoice-profile-pack",
                "version": 1,
                "embedding_model": rconfig.embedding.model_source,
                "profiles": [],
            }
        )
    )
    assert "Pack contained no profiles" in run("profiles", "import", str(empty)).output


def test_review_states(run, rconfig, monkeypatch):
    assert "No extraction sessions" in run("review", "--list").output
    from kickvoice.pipelines.session import ExtractionSession

    s = ExtractionSession.create(
        rconfig.extractions_dir, "https://kick.com/x", rconfig.embedding.model_source
    )
    s.add_candidate(
        embedding=voice_vector(1),
        sample_pcm=b"\x00\x00",
        sample_rate=SR,
        speech_seconds=30,
        n_windows=5,
        first_heard_s=0,
        cohesion=0.9,
    )
    s.save()
    assert "interactive terminal" in run("review", expect_ok=False).output
    s.candidates[0].status = "skipped"
    s.save()
    assert "already been reviewed" in run("review", "--session", s.id).output


def test_classic_commands_options_and_errors(run, rconfig, tmp_path, monkeypatch):
    import kickvoice.pipelines.extract as extract_mod
    import kickvoice.pipelines.monitor as monitor_mod

    audio = tmp_path / "a.wav"
    write_wav(audio, conversation([(220, 3)]), SR)
    seen = {}

    class FakeExtractor:
        def __init__(self, stream_url=None, file_path=None, config=None):
            seen.update(
                threshold=config.identify.match_threshold, cluster=config.extract.cluster_distance_threshold
            )

        def run(self, max_duration_s=None):
            return None  # nobody found

    monkeypatch.setattr(extract_mod, "ProfileExtractor", FakeExtractor)
    run(
        "extract-profiles",
        "--file",
        str(audio),
        "--no-agent",
        "--threshold",
        "0.7",
        "--cluster-threshold",
        "0.5",
    )
    assert seen == {"threshold": 0.7, "cluster": 0.5}

    class Broken:
        def __init__(self, **kw):
            pass

        def run(self):
            raise RuntimeError("streamlink failed: offline")

    monkeypatch.setattr(monitor_mod, "Monitor", Broken)
    out = run("monitor", "--file", str(audio), "--no-agent", "--threshold", "0.7", expect_ok=False).output
    assert "offline" in out


def test_extract_without_terminal_defers_review(run, rconfig, tmp_path, monkeypatch):
    import kickvoice.pipelines.extract as extract_mod

    class Session:
        id = "20250101-000000_x"

    class FoundOne:
        def __init__(self, **kw):
            pass

        def run(self, max_duration_s=None):
            return Session()

    monkeypatch.setattr(extract_mod, "ProfileExtractor", FoundOne)
    audio = tmp_path / "a.wav"
    write_wav(audio, conversation([(220, 3)]), SR)
    out = run("extract-profiles", "--file", str(audio), "--no-agent").output
    assert "Not an interactive terminal" in out and "review --session 20250101-000000_x" in out


def test_transcripts_empty_states(run):
    assert "No transcripts yet" in run("transcripts", "list").output
    assert "exactly one" in run("transcribe", expect_ok=False).output


# ── research CLI ─────────────────────────────────────────────────


def test_discover_lookup_failure_and_ambiguous_strong_matches(run, fake_web, monkeypatch):
    monkeypatch.setattr("kickvoice.research.net.time.sleep", lambda s: None)
    fake_web.route("GET", "/wd/api.php", status=500, json={})
    assert "Wikidata lookup failed" in run("research", "discover", "Jane", "--x", "j", expect_ok=False).output

    person = {
        "labels": {"en": {"value": "Jane"}},
        "descriptions": {"en": {"value": "pundit"}},
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]},
        "sitelinks": {},
    }

    def api(req):
        q = req["query"]
        if q.get("list") == "search":
            return 200, {}, {"query": {"search": [{"title": "Q1"}, {"title": "Q2"}]}}
        return 200, {}, {"entities": {i: person for i in q["ids"].split("|")}}

    fake_web.route("GET", "/wd/api.php", api)
    out = run("research", "discover", "Jane", "--x", "shared", "-y", expect_ok=False).output
    assert "Several records list that account" in out


def test_research_commands_empty_and_error_states(run, rconfig, tmp_path):
    run("research", "add", "Jane")
    assert "No accounts yet" in run("research", "accounts", "Jane").output
    assert "No confirmed accounts" in run("research", "collect", "Jane").output
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert "Couldn't import" in run("research", "import", "Jane", str(bad), expect_ok=False).output
    assert "No matches" in run("research", "search", "Jane", "anything").output
    assert "Nothing new to analyse" in run("research", "analyze", "Jane", "--yes").output


def test_collect_reports_many_errors(run, rconfig, monkeypatch):
    import kickvoice.pipelines.research as research_mod
    from kickvoice.research.sources.base import CollectStats

    run("research", "add", "Jane", "--x", "jane")
    monkeypatch.setattr(
        research_mod, "collect", lambda *a, **k: {"x": CollectStats(errors=[f"error {i}" for i in range(8)])}
    )
    out = run("research", "collect", "Jane").output
    assert "! error 4" in out and "… 3 more" in out


def test_analyze_confirmation_embeddings_and_problems(run, rconfig, fake_llm, monkeypatch, tmp_path):
    run("research", "add", "Jane")
    with ResearchStore(rconfig.research_db) as store:
        p = store.get_person("Jane")
        for i in range(2):
            d = Document(p.id, "x", "post", str(i), f"text number {i} about carbon taxes and policy", "self")
            store.upsert_document(d, [Chunk(d.text)])
    llm = fake_llm(
        ScriptedLLM(
            [
                {
                    "claims": [
                        {
                            "chunk": "K1",
                            "topic": "tax",
                            "claim": "c",
                            "quote": "text number 0 about carbon taxes and policy",
                        },
                        {
                            "chunk": "K2",
                            "topic": "tax",
                            "claim": "c",
                            "quote": "a quote that is not there at all",
                        },
                    ]
                }
            ]
        )
    )
    import kickvoice.cli_research as cr

    monkeypatch.setattr(cr, "_interactive", lambda: True)
    assert "Proceed?" in run("research", "analyze", "Jane", input="n\n").output
    assert not llm.calls  # declined → nothing spent

    class Embedder:
        def embed(self, texts):
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("kickvoice.research.llm.make_embedder", lambda config: Embedder())
    out = run("research", "analyze", "Jane", input="y\n").output
    assert "1 claim(s) extracted" in out and "1 embedded" in out and "1 claim(s) dropped" in out


def test_analyze_reports_failed_batches(run, rconfig, fake_llm):
    run("research", "add", "Jane")
    with ResearchStore(rconfig.research_db) as store:
        p = store.get_person("Jane")
        d = Document(p.id, "x", "post", "1", "something worth analysing here", "self")
        store.upsert_document(d, [Chunk(d.text)])
    fake_llm(ScriptedLLM(["garbage", "garbage"]))
    assert "failed and will be retried" in run("research", "analyze", "Jane", "--yes").output


def test_brief_prints_markdown(run, rconfig, fake_llm):
    from kickvoice.research.store import Claim

    run("research", "add", "Jane")
    with ResearchStore(rconfig.research_db) as store:
        p = store.get_person("Jane")
        d = Document(p.id, "x", "post", "1", "Carbon tax now.", "self")
        store.upsert_document(d, [Chunk(d.text)])
        ((chunk, _),) = store.chunks_to_analyze(p.id)
        store.add_claims([Claim(p.id, d.id, chunk.id, "carbon tax", "supports", "Carbon tax now")])
    fake_llm(ScriptedLLM([{"summary": "Supports a carbon tax."}]))
    assert "# Debate brief: Jane" in run("research", "brief", "Jane").output


def test_debate_command_guards(run, rconfig, monkeypatch):
    import kickvoice.pipelines.transcribe as tmod

    assert "exactly one" in run("debate", "--opponent", "Jane", expect_ok=False).output
    run("research", "add", "Jane", "--voice", "Jane")

    class Failing:
        def __init__(self, **kw):
            pass

        def run(self, max_duration_s=None):
            raise RuntimeError("stream went away")

    monkeypatch.setattr(tmod, "TranscriptionPipeline", Failing)
    out = run("debate", "--stream", "https://kick.com/x", "--opponent", "Jane", expect_ok=False).output
    assert "No analysed claims" in out and "stream went away" in out
