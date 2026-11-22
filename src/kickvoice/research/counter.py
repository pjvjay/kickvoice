"""
Live counter-arguments: given what the opponent just said, suggest replies
grounded in their own public record (and your prep notes).

Same principle as claim extraction — the model proposes, the code checks:
  - counters may only cite evidence ids that were actually provided
  - a "contradiction" must point at something THEY said first-hand and
    sincerely (not a Wikipedia summary, not a joke)
  - quotes shown to you come from the database, never from the model's text
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from kickvoice.config import Config
from kickvoice.research.llm import LLMClient, complete_json
from kickvoice.research.retrieve import Retriever
from kickvoice.research.store import (
    AUTHORSHIP_LABELS,
    FIRST_HAND,
    Evidence,
    Person,
    ResearchStore,
    fts_or_query,
)

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """A paragraph of your own prep notes (facts, talking points)."""

    id: int
    text: str

    @property
    def ref(self) -> str:
        return f"N{self.id}"


@dataclass
class Counter:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    @property
    def sourced(self) -> bool:
        return bool(self.evidence or self.notes)


@dataclass
class Contradiction:
    evidence: Evidence
    explanation: str


@dataclass
class Suggestion:
    point: str  # what they said (transcript)
    summary: str = ""  # the model's one-line read of their point
    is_argument: bool = False
    counters: list[Counter] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    dropped: int = 0  # model items discarded by validation
    latency_s: float = 0.0
    model: str = ""


def load_notes(path: str | Path | None) -> list[Note]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Note(i + 1, p[:1200]) for i, p in enumerate(paragraphs)]


def relevant_notes(notes: list[Note], text: str, k: int = 4) -> list[Note]:
    terms = {t.strip('"') for t in fts_or_query(text).split(" OR ") if t}
    if not terms:
        return []
    scored = []
    for note in notes:
        words = set(re.findall(r"[a-z0-9']+", note.text.lower()))
        hits = len(terms & words)
        if hits:
            scored.append((hits, note))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:k]]


SYSTEM = """You are a sharp, honest debate coach whispering to your debater during a LIVE debate
against {name}. You get {name}'s latest point (from a live transcript, so expect
transcription errors), recent context, evidence from {name}'s own public record, and your
debater's prep notes. Help your debater respond in the next few seconds.

Rules:
- If the latest point isn't an argument or claim (banter, greetings, filler), set
  "is_argument": false and return no counters.
- Up to 3 counters, each ONE short sentence (max 35 words) your debater could say out loud.
- Ground counters in the evidence/notes where possible and cite their ids (e.g. "C12",
  "K7", "N2"). Only cite ids that appear below. Never invent quotes, dates or statistics;
  if a counter needs a fact you weren't given, phrase it as a question or as reasoning.
- A "contradiction" is ONLY when evidence marked first-hand shows {name} previously said
  something that genuinely conflicts with the latest point. Don't stretch. Evidence
  marked "reported about them" or "insincere" can't be a contradiction.
- Be fair: attack the argument, not the person.

{style_rules}
Reply with ONLY JSON:
{{"is_argument": true, "their_point": "one-line summary",
  "counters": [{{"text": "...", "evidence": ["C12", "N1"]}}],
  "contradictions": [{{"evidence": "C40", "explanation": "..."}}]}}"""


STYLE_RULES = {
    "spoken": "",
    "chat": (
        "STYLE: each counter is a message a viewer will type into the stream's live chat.\n"
        "- At most {max_chars} characters, one line, plain text: no markdown, hashtags,\n"
        "  emojis, @mentions or links.\n"
        '- Casual, direct, conversational, addressed to {name} ("you").\n'
        "- Quote marks ONLY around words copied exactly from {name}'s first-hand evidence.\n"
        "- Mention a year only if the cited evidence is from that year.\n"
    ),
}


class CounterEngine:
    def __init__(
        self,
        store: ResearchStore,
        person: Person,
        llm: LLMClient,
        config: Config,
        retriever: Retriever | None = None,
        notes: list[Note] | None = None,
    ):
        self.store = store
        self.person = person
        self.llm = llm
        self.config = config
        self.retriever = retriever or Retriever(store, person)
        self.notes = notes or []

    def suggest(
        self, point: str, context: list[str] | None = None, style: str = "spoken", max_chars: int = 300
    ) -> Suggestion:
        """
        `style="chat"` writes each counter as a ready-to-paste chat message
        of at most `max_chars` characters.
        """
        if style not in STYLE_RULES:
            raise ValueError(f"style must be one of {tuple(STYLE_RULES)}")
        started = time.monotonic()
        k = self.config.debate.evidence_items
        evidence = self.retriever.search(point, k=k)
        notes = relevant_notes(self.notes, point)
        insincere = self.store.insincere_refs(evidence)

        prompt = self._prompt(point, context or [], evidence, notes, insincere)
        by_ref = {e.ref: e for e in evidence}
        notes_by_ref = {n.ref: n for n in notes}
        suggestion = Suggestion(point=point)
        try:
            rules = STYLE_RULES[style].format(name=self.person.name, max_chars=max_chars)
            data, reply = complete_json(
                self.llm,
                SYSTEM.format(name=self.person.name, style_rules=rules),
                prompt,
                self.config.llm.live_model,
                validate=_validate,
                max_tokens=700,
            )
        except Exception as e:
            logger.warning("Live suggestion failed: %s", e)
            suggestion.summary = f"(no suggestion: {e})"
            suggestion.latency_s = time.monotonic() - started
            return suggestion

        suggestion.model = reply.model
        suggestion.is_argument = data["is_argument"]
        suggestion.summary = data["their_point"]
        for item in data["counters"]:
            refs = [str(r).strip("[]") for r in item["evidence"]]
            unknown = [r for r in refs if r not in by_ref and r not in notes_by_ref]
            if unknown:
                suggestion.dropped += 1  # cited something it wasn't given
                continue
            if len(suggestion.counters) == 3:
                break  # validate first, then cap — bad items don't use slots
            text = item["text"].strip()
            if style == "chat":
                from kickvoice.agent.chat import fit_to_chat

                text = fit_to_chat(text, max_chars)
            suggestion.counters.append(
                Counter(
                    text=text,
                    evidence=[by_ref[r] for r in refs if r in by_ref],
                    notes=[notes_by_ref[r] for r in refs if r in notes_by_ref],
                )
            )
        for item in data["contradictions"]:
            ev = by_ref.get(str(item["evidence"]).strip("[]"))
            if ev is None or ev.authorship not in FIRST_HAND or ev.ref in insincere:
                suggestion.dropped += 1
                continue
            if len(suggestion.contradictions) < 2:
                suggestion.contradictions.append(Contradiction(ev, item["explanation"].strip()))
        suggestion.latency_s = time.monotonic() - started
        return suggestion

    def _prompt(self, point, context, evidence, notes, insincere) -> str:
        lines = ["Recent conversation (oldest first):"]
        lines += [f"  {c}" for c in context[-6:]] or ["  (none)"]
        lines += [
            "",
            f"{self.person.name}'s latest point:",
            f"  {point}",
            "",
            f"Evidence from {self.person.name}'s public record:",
        ]
        if not evidence:
            lines.append("  (nothing relevant found)")
        for ev in evidence:
            tags = [AUTHORSHIP_LABELS.get(ev.authorship, ev.authorship)]
            if ev.ref in insincere:
                tags.append("insincere")
            when = f", {ev.published_at[:10]}" if ev.published_at else ""
            body = f'{ev.text} — quote: "{ev.quote}"' if ev.kind == "claim" else f'"{ev.quote}"'
            lines.append(f"  [{ev.ref}] ({'; '.join(tags)}; {ev.source}{when}) {body}")
        lines += ["", "Your debater's prep notes:"]
        lines += [f"  [{n.ref}] {n.text}" for n in notes] or ["  (none relevant)"]
        return "\n".join(lines)


def _validate(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    counters = []
    for item in data.get("counters") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            ev = item.get("evidence") or []
            counters.append({"text": item["text"], "evidence": ev if isinstance(ev, list) else [ev]})
    contradictions = [
        {"evidence": item["evidence"], "explanation": str(item.get("explanation") or "")}
        for item in (data.get("contradictions") or [])
        if isinstance(item, dict) and isinstance(item.get("evidence"), (str, int))
    ]
    return {
        "is_argument": data.get("is_argument") is not False,
        "their_point": str(data.get("their_point") or "").strip(),
        "counters": counters,
        "contradictions": contradictions,
    }
