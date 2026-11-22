"""`kickvoice research …` and `kickvoice debate` through Click."""
import json
import re

import pytest
from click.testing import CliRunner

from conftest import ScriptedLLM
from kickvoice.cli import cli
from kickvoice.research.store import ResearchStore
from test_research_sources import wikidata_routes, x_routes


@pytest.fixture
def run(rconfig, tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    rconfig.save(cfg_path)
    runner = CliRunner()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def _run(*args, input=None, expect_ok=True):
        result = runner.invoke(cli, ["--config", str(cfg_path), *args], input=input, obj={})
        if expect_ok and result.exit_code != 0:
            raise AssertionError(f"exit {result.exit_code}: {result.output}\n{result.exception!r}")
        return result
    return _run


@pytest.fixture
def fake_llm(monkeypatch):
    """Make every command use a scripted model."""
    holder = {}

    def install(llm):
        import kickvoice.research.llm as llm_mod
        monkeypatch.setattr(llm_mod, "make_llm", lambda config: llm)
        holder["llm"] = llm
        return llm
    return install


def db(rconfig):
    return ResearchStore(rconfig.research_db)


def test_add_list_show_remove(run, rconfig):
    out = run("research", "add", "Jane Pundit", "--x", "JanePundit", "--youtube", "@jane",
              "--voice", "Jane", "--description", "streamer").output
    assert "Jane Pundit" in out and "x:JanePundit" in out and "voice profile: Jane" in out
    with db(rconfig) as store:
        p = store.get_person("jane pundit")
        assert p.voice_profile == "Jane"
        accounts = {a.platform: a for a in store.accounts(p.id)}
        assert accounts["x"].status == "confirmed" and accounts["x"].url == "https://x.com/JanePundit"
        assert accounts["youtube"].url == "https://www.youtube.com/@jane"
    assert "Jane Pundit" in run("research", "list").output
    shown = run("research", "show", "Jane Pundit").output
    assert "Voice profile: Jane" in shown and "JanePundit" in shown
    assert "Cancelled" in run("research", "remove", "Jane Pundit", input="n\n").output
    run("research", "remove", "Jane Pundit", "-y")
    assert "No research subjects" in run("research", "list").output


def test_unknown_subject(run):
    result = run("research", "show", "Nobody", expect_ok=False)
    assert result.exit_code == 1 and "research add" in result.output


def test_discover_yes_confirms_wikidata_accounts(run, rconfig, fake_web):
    wikidata_routes(fake_web)
    out = run("research", "discover", "Jane Pundit", "--x", "JanePundit", "--yes").output
    assert "wikidata Q42" in out
    with db(rconfig) as store:
        p = store.get_person("Jane Pundit")
        assert p.wikidata_id == "Q42" and p.description == "American political streamer"
        statuses = {(a.platform, a.handle): a.status for a in store.accounts(p.id)}
    assert statuses[("x", "JanePundit")] == "confirmed"
    assert statuses[("wikipedia", "Jane Pundit")] == "confirmed"
    assert ("x", "OldJane") not in statuses                      # deprecated handle ignored


def test_discover_without_tty_leaves_candidates(run, rconfig, fake_web):
    wikidata_routes(fake_web)
    out = run("research", "discover", "Jane Pundit", "--x", "JanePundit").output
    assert "candidate" in out and "--confirm" in out
    with db(rconfig) as store:
        p = store.get_person("Jane Pundit")
        accounts = store.accounts(p.id)
    assert {a.status for a in accounts} == {"candidate"}
    x_id = [a.id for a in accounts if a.platform == "x"][0]
    site_id = [a.id for a in accounts if a.platform == "website"][0]
    out = run("research", "accounts", "Jane Pundit", "--confirm", str(x_id),
              "--reject", str(site_id)).output
    assert re.search(rf"{x_id}\s+x\s+JanePundit\s+confirmed", out)
    assert re.search(rf"{site_id}\s+website\s+\S+\s+rejected", out)
    bad = run("research", "accounts", "Jane Pundit", "--confirm", "9999", expect_ok=False)
    assert bad.exit_code == 1


def test_discover_nothing_found(run, fake_web):
    fake_web.route("GET", "/wd/api.php", json={"search": []})
    assert "No Wikidata record" in run("research", "discover", "Mystery Person").output


def test_collect_import_analyze_search_brief(run, rconfig, fake_web, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    x_routes(fake_web, {None: {"data": [
        {"id": "1", "text": "Rent control reduces housing supply, build more homes instead.",
         "created_at": "2021-03-01T00:00:00Z"},
        {"id": "2", "text": "A carbon tax with a dividend is the cleanest climate policy we have.",
         "created_at": "2023-01-01T00:00:00Z"}], "meta": {}}})
    run("research", "add", "Jane Pundit", "--x", "JanePundit")
    out = run("research", "collect", "Jane Pundit").output
    assert re.search(r"x\s+new\s+2", out) and "research analyze" in out

    posts = tmp_path / "extra.jsonl"
    posts.write_text(json.dumps({"id": "e1", "text": "Zoning reform beats rent control every time."}) + "\n")
    assert "Imported 1" in run("research", "import", "Jane Pundit", str(posts)).output

    def model(system, prompt):
        if "Reply with ONLY JSON: {\"claims\"" in system:
            items = []
            for ref, text in re.findall(r"\[(K\d+)\][^\n]*\n([^\n]+)", prompt):
                topic = "carbon tax" if "carbon" in text else "rent control"
                items.append({"chunk": ref, "topic": topic, "claim": f"says: {text[:30]}",
                              "quote": " ".join(text.split()[:7]), "kind": "position", "sincere": True})
            return {"claims": items}
        return {"summary": "Consistent opponent of rent control.", "contradictions": [],
                "expected_arguments": ["Supply matters"], "counters": []}
    fake_llm(ScriptedLLM(fn=model))

    refused = run("research", "analyze", "Jane Pundit", expect_ok=False)     # not a TTY
    assert refused.exit_code == 1 and "--yes" in refused.output
    out = run("research", "analyze", "Jane Pundit", "--yes").output
    assert "3 claim(s) extracted" in out and "tokens" in out
    assert "Nothing new" in run("research", "analyze", "Jane Pundit", "--yes").output

    shown = run("research", "show", "Jane Pundit").output
    assert re.search(r"2\s+rent control", shown) and "carbon tax" in shown
    topic = run("research", "show", "Jane Pundit", "--topic", "carbon").output
    assert '"A carbon tax with a dividend is"' in topic and "2023-01-01" in topic

    hits = run("research", "search", "Jane Pundit", "rent control housing").output
    first_hit = hits.split("\n[")[0]
    assert "Rent control reduces housing" in first_hit and "carbon" not in first_hit
    assert "https://x.com/JanePundit/status/1" in hits

    brief = tmp_path / "brief.md"
    run("research", "brief", "Jane Pundit", "--out", str(brief))
    md = brief.read_text()
    assert "# Debate brief: Jane Pundit" in md and "## Rent Control" in md
    assert '"Rent control reduces housing supply, build more' in md


def test_brief_needs_claims(run):
    run("research", "add", "Jane Pundit")
    result = run("research", "brief", "Jane Pundit", expect_ok=False)
    assert result.exit_code == 1 and "research analyze" in result.output


def test_analyze_needs_api_key(run, monkeypatch, rconfig):
    run("research", "add", "Jane Pundit")
    with db(rconfig) as store:
        from kickvoice.research.store import Chunk, Document
        p = store.get_person("Jane Pundit")
        store.upsert_document(Document(p.id, "x", "post", "1", "text here", "self"), [Chunk("text here")])
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    result = run("research", "analyze", "Jane Pundit", "--yes", expect_ok=False)
    assert result.exit_code == 1 and "ANTHROPIC_API_KEY" in result.output


# ── debate ───────────────────────────────────────────────────────

def test_debate_requires_voice_or_any_speaker(run):
    run("research", "add", "Jane Pundit")
    result = run("debate", "--stream", "https://kick.com/x", "--opponent", "Jane Pundit",
                 expect_ok=False)
    assert result.exit_code == 1 and "--any-speaker" in result.output


def test_debate_cli_end_to_end(run, rconfig, fake_llm, monkeypatch):
    """Transcription is faked (lines fed directly); everything else is real."""
    import kickvoice.pipelines.transcribe as tmod
    from kickvoice.transcripts.store import Utterance

    run("research", "add", "Jane Pundit", "--voice", "Jane")
    with db(rconfig) as store:
        from kickvoice.research.store import Chunk, Claim, Document
        p = store.get_person("Jane Pundit")
        doc = Document(p.id, "x", "post", "1", "Rent control reduces housing supply everywhere.",
                       "self", url="https://x.com/j/status/1", published_at="2021-03-01")
        store.upsert_document(doc, [Chunk(doc.text)])
        k = store.chunks_to_analyze(p.id)[0][0]
        (cid,) = store.add_claims([Claim(p.id, doc.id, k.id, "rent control",
                                         "Rent control reduces supply",
                                         "Rent control reduces housing supply everywhere",
                                         published_at="2021-03-01", url=doc.url)])

    class FakePipeline:
        def __init__(self, stream_url=None, file_path=None, config=None, on_utterance=None):
            assert config.transcribe.chunk_duration_s == config.debate.chunk_duration_s
            self.on_utterance = on_utterance

        def run(self, max_duration_s=None):
            lines = [("Jane", "rent control protects tenants and keeps housing supply stable for everyone"
                              " who needs an affordable home in this city"),
                     ("Me", "that's not what the evidence shows")]
            for i, (who, text) in enumerate(lines):
                self.on_utterance(Utterance("s", i * 5.0, i * 5.0 + 4, text, speaker=who,
                                            attribution="confident"))

    monkeypatch.setattr(tmod, "TranscriptionPipeline", FakePipeline)
    fake_llm(ScriptedLLM([{
        "is_argument": True, "their_point": "Rent control keeps housing stable",
        "counters": [{"text": "In 2021 you said it reduces supply.", "evidence": [f"C{cid}"]}],
        "contradictions": [{"evidence": f"C{cid}", "explanation": "Opposite of your 2021 view"}]}]))

    out = run("debate", "--stream", "https://kick.com/x", "--opponent", "Jane Pundit").output
    assert "Debate assist vs Jane Pundit" in out
    assert "Jane: rent control protects tenants" in out
    assert "CONTRADICTION" in out and "Opposite of your 2021 view" in out
    assert '"Rent control reduces housing supply everywhere" — 2021-03-01' in out
    assert "↩ 1." in out and "https://x.com/j/status/1" in out
    assert "1 suggestion(s)" in out
    logs = list((rconfig.data_dir / "debates").glob("*.jsonl"))
    assert len(logs) == 1 and "Opposite of your 2021 view" in logs[0].read_text()
