"""
Research orchestration: who is this person, and what have they said?

    discover  → Wikidata/Wikipedia: candidate identity + official accounts
                (you confirm each one — nothing is collected from unconfirmed accounts)
    collect   → every confirmed account's public content → documents
    analyze   → documents → verified claims (see research/extract.py)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kickvoice.config import Config
from kickvoice.research.net import Fetcher, NetError
from kickvoice.research.sources import local, social, web, wiki, youtube
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import CONFIRMED, Person, ResearchStore

logger = logging.getLogger(__name__)

# Platforms we can collect from, and which need voice mode.
COLLECTABLE = ("wikipedia", "youtube", "x", "reddit", "website", "twitch", "transcripts")
VOICE_ONLY = ("twitch",)        # no captions → only via voice transcription


@dataclass
class CollectOptions:
    only: tuple[str, ...] = ()          # empty = everything available
    voice: bool = False                 # use voice-verified transcription for videos
    appearances: int = 0                # search N videos on other channels (voice only)
    web_search: int = 0                 # fetch N web search results about them
    max_videos: int | None = None
    max_posts: int | None = None


def discover(fetcher: Fetcher, config: Config, name: str,
             handles: dict[str, str] | None = None) -> list[wiki.WikidataPerson]:
    """
    Candidate identities from Wikidata: by the accounts you know (exact
    match), otherwise by name. Returns full entities with their accounts.
    """
    api = config.research.wikidata_api
    qids: list[str] = []
    for platform, handle in (handles or {}).items():
        for qid in wiki.find_by_account(fetcher, api, platform, handle):
            if qid not in qids:
                qids.append(qid)
    if not qids:
        qids = [hit["id"] for hit in wiki.search_people(fetcher, api, name)]
    people = []
    for qid in qids[:5]:
        try:
            people.append(wiki.get_person(fetcher, api, qid))
        except (ValueError, NetError) as e:
            logger.info("Skipping %s: %s", qid, e)
    return people


def collect(
    store: ResearchStore,
    person: Person,
    config: Config,
    options: CollectOptions | None = None,
    fetcher: Fetcher | None = None,
    ytdlp: youtube.YtDlp | None = None,
    voice_transcriber: youtube.VoiceTranscriber | None = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> dict[str, CollectStats]:
    """Collect from every confirmed account. Returns stats per source."""
    options = options or CollectOptions()
    fetcher = fetcher or Fetcher(config)
    ctx = SourceContext(config, fetcher, store, person, progress)
    results: dict[str, CollectStats] = {}
    wanted = lambda platform: not options.only or platform in options.only  # noqa: E731

    if options.voice and voice_transcriber is None:
        if not person.voice_profile:
            raise ValueError(f"--voice needs a voice profile linked to {person.name} "
                             "(research add NAME --voice PROFILE)")
        voice_transcriber = youtube.voice_transcriber(config, person.voice_profile)
    voice = voice_transcriber if options.voice else None

    def run(key: str, fn, *args, **kwargs):
        try:
            stats = fn(*args, **kwargs)
        except (NetError, RuntimeError, ValueError) as e:
            stats = CollectStats(errors=[f"{key}: {e}"])
        results.setdefault(key, CollectStats()).merge(stats)

    own_channels: set[str] = set()
    for account in store.accounts(person.id, status=CONFIRMED):
        platform = account.platform
        if not wanted(platform):
            continue
        if platform == "wikipedia":
            run("wikipedia", wiki.collect_wikipedia, ctx, account)
        elif platform == "youtube":
            if account.handle.startswith("UC"):
                own_channels.add(account.handle)
            run("youtube", youtube.collect_channel, ctx, account, ytdlp=ytdlp, voice=voice,
                limit=options.max_videos)
        elif platform == "twitch":
            if voice is None:
                results.setdefault("twitch", CollectStats()).errors.append(
                    "twitch VODs have no captions — use --voice")
                continue
            run("twitch", youtube.collect_channel, ctx,
                _as_url_account(account, f"https://www.twitch.tv/{account.handle}/videos?filter=archives"),
                ytdlp=ytdlp, voice=voice, limit=options.max_videos)
        elif platform == "x":
            run("x", social.collect_x, ctx, account, limit=options.max_posts)
        elif platform == "reddit":
            run("reddit", social.collect_reddit, ctx, account, limit=options.max_posts)
        elif platform == "website":
            run("web", web.collect_url, ctx, account.url or account.handle, first_hand=True)
        else:
            results.setdefault(platform, CollectStats()).errors.append(
                f"{platform}: no supported collector (public API unavailable)")

    if person.voice_profile and wanted("transcripts") and Path(config.transcripts_db).exists():
        run("transcripts", local.collect_transcripts, ctx, person.voice_profile)

    if options.appearances and wanted("youtube"):
        if voice is None:
            results.setdefault("appearances", CollectStats()).errors.append(
                "appearances need --voice (captions can't tell who is speaking)")
        else:
            run("appearances", youtube.collect_appearances, ctx, f'"{person.name}"', voice,
                own_channels, ytdlp=ytdlp, limit=options.appearances)

    if options.web_search and wanted("web"):
        try:
            hits = web.search_web(ctx, f'"{person.name}"', count=options.web_search)
        except (RuntimeError, NetError) as e:
            results.setdefault("web", CollectStats()).errors.append(str(e))
            hits = []
        own = {a.url for a in store.accounts(person.id, status=CONFIRMED, platform="website")}
        for hit in hits:
            if hit["url"] not in own:
                run("web", web.collect_url, ctx, hit["url"], first_hand=False)
    return results


def _as_url_account(account, url):
    from dataclasses import replace

    return replace(account, handle=url)
