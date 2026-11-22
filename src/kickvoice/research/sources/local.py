"""
Local sources: your own stream transcripts, and files you import.

Transcripts are the most trustworthy first-hand source there is: lines
kickvoice itself attributed to their voice, with "confident" attribution.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from kickvoice.research.chunking import chunk_timed
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import ABOUT, SELF, SELF_VOICE, Chunk, Document
from kickvoice.transcripts.attribution import CONFIDENT
from kickvoice.transcripts.store import TranscriptStore


def collect_transcripts(ctx: SourceContext, voice_profile: str) -> CollectStats:
    """Confident lines attributed to `voice_profile`, one document per session."""
    stats = CollectStats()
    db = ctx.config.transcripts_db
    if not Path(db).exists():
        return stats
    with TranscriptStore(db) as transcripts:
        for session, utterances in transcripts.iter_sessions_with_utterances(
            speaker=voice_profile, min_attribution=CONFIDENT
        ):
            lines = [(u.start_s, u.end_s, u.text) for u in utterances]
            url = session.source if "://" in session.source else ""
            ctx.save(
                Document(
                    person_id=ctx.person.id,
                    source="stream",
                    kind="stream",
                    external_id=session.id,
                    text="\n".join(t for _, _, t in lines),
                    authorship=SELF_VOICE,
                    url=url,
                    title=f"Stream transcript {session.id}",
                    published_at=session.started_at,
                    meta={"transcript_session": session.id, "lines": len(lines)},
                ),
                chunk_timed(lines),
                stats,
            )
    ctx.progress(f"transcripts: {stats.total} session(s) with confident lines by {voice_profile}")
    return stats


def import_file(
    ctx: SourceContext, path: str | Path, first_hand: bool = True, source: str = "file"
) -> CollectStats:
    """
    Import text you legitimately have: a JSONL/CSV of posts, an X data
    export (tweets.js), or a plain text/markdown file.

    JSONL/CSV rows use the fields: text (required), id, url, date, title.
    """
    path = Path(path)
    stats = CollectStats()
    authorship = SELF if first_hand else ABOUT
    rows: list[dict]
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.name.startswith("tweets") and suffix == ".js":
        rows = _x_archive_rows(raw)
        source = "x"
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(raw)
        rows = data if isinstance(data, list) else [data]
    elif suffix == ".csv":
        rows = list(csv.DictReader(raw.splitlines()))
    else:
        ctx.save(
            Document(
                person_id=ctx.person.id,
                source=source,
                kind="file",
                external_id=str(path.resolve()),
                text=raw,
                authorship=authorship,
                title=path.name,
            ),
            None,
            stats,
        )
        return stats

    for i, row in enumerate(rows):
        text = str(row.get("text") or row.get("full_text") or row.get("body") or "").strip()
        if not text:
            continue
        ctx.save(
            Document(
                person_id=ctx.person.id,
                source=source,
                kind="post",
                external_id=str(row.get("id") or row.get("id_str") or f"{path.name}:{i}"),
                text=text,
                authorship=authorship,
                url=str(row.get("url") or ""),
                title=str(row.get("title") or ""),
                published_at=str(row.get("date") or row.get("created_at") or "") or None,
            ),
            [Chunk(text)] if len(text) < 1500 else None,
            stats,
        )
    ctx.progress(f"import: {path.name} — {stats.total} item(s)")
    return stats


def _x_archive_rows(raw: str) -> list[dict]:
    """X data export `tweets.js`: `window.YTD.tweets.part0 = [ {tweet: {...}} ]`."""
    body = re.sub(r"^\s*window\.YTD\.[\w.]+\s*=\s*", "", raw)
    rows = []
    for item in json.loads(body):
        tweet = item.get("tweet", item)
        rows.append(
            {
                "id": tweet.get("id_str") or tweet.get("id"),
                "text": tweet.get("full_text") or tweet.get("text"),
                "created_at": tweet.get("created_at"),
            }
        )
    return rows
