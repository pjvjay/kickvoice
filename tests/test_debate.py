"""Live debate assistant: point detection, queueing, threading, end to end."""

import json
import time

from conftest import (
    FakeCapture,
    FakeEncoder,
    FakeRecognizer,
    ScriptedLLM,
    conversation,
    voice_vector,
)
from kickvoice.audio.vad import EnergyVAD
from kickvoice.pipelines.debate import (
    DebateAssistant,
    PointAggregator,
    opponent_matcher,
)
from kickvoice.pipelines.transcribe import TranscriptionPipeline, speaker_label
from kickvoice.research.counter import CounterEngine, Suggestion
from kickvoice.research.store import SELF, Chunk, Claim, Document, ResearchStore
from kickvoice.speakers.profiles import ProfileStore
from kickvoice.transcripts.store import TranscriptStore, Utterance


def line(speaker, text, start=0.0, attribution="confident"):
    return Utterance(
        "s", start, start + 2, text, speaker=speaker, attribution=attribution if speaker else "unknown"
    )


def words(n, word="word"):
    return " ".join([word] * n)


# ── which lines are the opponent's ───────────────────────────────


def test_opponent_matcher():
    is_opp = opponent_matcher("Jane")
    assert is_opp(line("jane", "x"))  # case-insensitive
    assert not is_opp(line("Bob", "x"))
    assert not is_opp(line(None, "x"))
    anyone = opponent_matcher(None, me="Me", any_speaker=True)
    assert anyone(line("Bob", "x")) and anyone(line(None, "x")) and not anyone(line("me", "x"))
    assert not opponent_matcher(None)(line("Jane", "x"))  # no profile → nobody


# ── points ───────────────────────────────────────────────────────


def aggregator(**kw):
    return PointAggregator(opponent_matcher("Jane"), speaker_label, **kw)


def test_point_ready_when_someone_else_speaks():
    agg = aggregator(min_words=10)
    assert agg.add(line("Jane", words(6))) is None
    assert agg.add(line("Jane", words(6))) is None
    point = agg.add(line("Me", "let me respond"))
    assert point is not None and len(point.text.split()) == 12
    assert point.context[-1] == "Jane: " + words(6)  # context before my reply


def test_short_aside_is_ignored():
    agg = aggregator(min_words=20)
    agg.add(line("Jane", "no way"))
    assert agg.add(line("Me", "yes way")) is None
    assert agg.buffered_words == 0


def test_long_monologue_is_split():
    agg = aggregator(min_words=10, max_words=30)
    assert agg.add(line("Jane", words(20))) is None
    point = agg.add(line("Jane", words(15)))
    assert point is not None and len(point.text.split()) == 35


def test_idle_flush():
    agg = aggregator(min_words=10)
    agg.add(line("Jane", words(12)), now=100.0)
    assert agg.flush_if_idle(4.0, now=102.0) is None
    assert agg.flush_if_idle(4.0, now=104.5).text == words(12)
    assert agg.flush_if_idle(4.0, now=200.0) is None  # already flushed


# ── assistant (threads + queue) ──────────────────────────────────


class FakeEngine:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.points = []

    def suggest(self, point, context=None):
        self.points.append(point)
        time.sleep(self.delay)
        return Suggestion(point=point, summary=f"summary of {point[:10]}", is_argument=True)


def test_assistant_processes_points_and_logs(config, tmp_path):
    engine, rendered = FakeEngine(), []
    config.debate.min_point_words = 5
    log = tmp_path / "debate.jsonl"
    assistant = DebateAssistant(
        engine, aggregator(min_words=5), config, log, render=lambda p, s: rendered.append(s)
    )
    assistant.start()
    assistant.on_utterance(line("Jane", "first point " + words(5)))
    assistant.on_utterance(line("Me", "reply"))
    assistant.on_utterance(line("Jane", "second point " + words(5)))
    assistant.stop()  # drains the unfinished 2nd point
    assert [p.split()[0] for p in engine.points] == ["first", "second"]
    assert len(rendered) == 2
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert entries[0]["summary"].startswith("summary of first")


def test_assistant_drops_stale_points_when_model_is_slow(config):
    engine = FakeEngine(delay=0.3)
    config.debate.max_queued_points = 1
    assistant = DebateAssistant(engine, aggregator(min_words=3), config, None, render=lambda p, s: None)
    assistant.start()
    for i in range(5):
        assistant.on_utterance(line("Jane", f"point{i} " + words(4)))
        assistant.on_utterance(line("Me", "reply"))
    assistant.stop()
    assert assistant.skipped >= 2
    assert engine.points[-1].startswith("point4")  # newest is always answered


# ── end to end: real transcription + real engine + scripted model ─


def test_live_debate_end_to_end(config, tmp_path):
    JANE, ME = 220, 550
    profiles = ProfileStore(config)
    profiles.add_profile("Jane", voice_vector(JANE), speech_seconds=300)
    profiles.add_profile("Me", voice_vector(ME), speech_seconds=300)

    research = ResearchStore(config.research_db)
    person = research.add_person("Jane Pundit", voice_profile="Jane")
    doc = Document(
        person.id,
        "x",
        "post",
        "1",
        "w220 w220 w220 is what I believe",
        SELF,
        url="https://x.com/j/status/1",
        published_at="2022-02-02",
    )
    research.upsert_document(doc, [Chunk(doc.text)])
    chunk = research.chunks_to_analyze(person.id)[0][0]
    (claim_id,) = research.add_claims(
        [
            Claim(
                person.id,
                doc.id,
                chunk.id,
                "w220",
                "Believes in w220",
                "w220 w220 w220 is what I believe",
                published_at="2022-02-02",
                url=doc.url,
            )
        ]
    )

    def model(system, prompt):
        assert "Jane Pundit's latest point" in prompt and f"[C{claim_id}]" in prompt
        return {
            "is_argument": True,
            "their_point": "Jane repeats w220",
            "counters": [{"text": "You said this in 2022 too.", "evidence": [f"C{claim_id}"]}],
            "contradictions": [],
        }

    config.debate.min_point_words = 5
    config.transcribe.chunk_duration_s = 5.0
    engine = CounterEngine(research, person, ScriptedLLM(fn=model), config)
    rendered = []
    assistant = DebateAssistant(
        engine,
        PointAggregator(opponent_matcher("Jane"), speaker_label, min_words=5),
        config,
        tmp_path / "log.jsonl",
        render=lambda p, s: rendered.append(s),
    )
    with TranscriptStore(config.transcripts_db) as transcripts:
        pipeline = TranscriptionPipeline(
            stream_url="https://kick.com/debate",
            config=config,
            capture=FakeCapture(conversation([(JANE, 5), (ME, 4), (JANE, 5), (ME, 3)])),
            vad=EnergyVAD(config.audio),
            encoder=FakeEncoder(),
            recognizer=FakeRecognizer(),
            profiles=profiles,
            transcripts=transcripts,
            on_utterance=assistant.on_utterance,
            out=lambda _: None,
        )
        assistant.start()
        pipeline.run(max_duration_s=0)
        assistant.stop()
    research.close()

    assert len(rendered) == 2  # one per Jane turn
    s = rendered[0]
    assert s.summary == "Jane repeats w220"
    assert s.counters[0].evidence[0].quote == "w220 w220 w220 is what I believe"
    assert s.counters[0].evidence[0].url == "https://x.com/j/status/1"
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[0]["counters"][0]["evidence"][0]["date"] == "2022-02-02"
