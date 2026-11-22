"""
Research orchestration: who is this person, and what have they said?

    discover  → Wikidata/Wikipedia: candidate identity + official accounts
                (you confirm each one — nothing is collected from unconfirmed accounts)
    collect   → every confirmed account's public content → documents
    analyze   → documents → verified claims (see research/extract.py)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from kickvoice.config import Config
from kickvoice.research.net import Fetcher, NetError
from kickvoice.research.sources import local, social, web, wiki, youtube
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import (
    CANDIDATE,
    CONFIRMED,
    REJECTED,
    Account,
    Person,
    ResearchStore,
)

logger = logging.getLogger(__name__)

# Platforms we can collect from, and which need voice mode.
COLLECTABLE = ("wikipedia", "youtube", "x", "reddit", "website", "twitch", "transcripts")
VOICE_ONLY = ("twitch",)  # no captions → only via voice transcription


@dataclass
class CollectOptions:
    only: tuple[str, ...] = ()  # empty = everything available
    voice: bool = False  # use voice-verified transcription for videos
    appearances: int = 0  # search N videos on other channels (voice only)
    web_search: int = 0  # fetch N web search results about them
    max_videos: int | None = None
    max_posts: int | None = None


ACCOUNT_MATCH = "account"  # a Wikidata record lists a handle you gave — strong evidence
QID_MATCH = "qid"  # you named the record explicitly — strong evidence
NAME_MATCH = "name"  # only the name matched — never enough on its own


@dataclass
class Candidate:
    record: wiki.WikidataPerson
    matched_by: str
    detail: str = ""  # e.g. "x:SomeHandle"

    @property
    def strong(self) -> bool:
        return self.matched_by in (ACCOUNT_MATCH, QID_MATCH)


@dataclass
class DiscoverResult:
    candidates: list[Candidate] = field(default_factory=list)  # people only
    skipped: list[wiki.WikidataPerson] = field(default_factory=list)  # matched, but not a person
    unmatched_handles: list[str] = field(default_factory=list)  # no record lists these
    searched_by_name: bool = False

    @property
    def strong(self) -> list[Candidate]:
        return [c for c in self.candidates if c.strong]


def discover(
    fetcher: Fetcher, config: Config, name: str, handles: dict[str, str] | None = None, qid: str | None = None
) -> DiscoverResult:
    """
    Candidate identities from Wikidata.

    Evidence, strongest first: a record you name (`qid`), a record that lists
    a handle you gave (exact statement match), and only then a name search.
    Records that aren't people (games, companies, shows…) are never
    candidates, however they matched.
    """
    api = config.research.wikidata_api
    result = DiscoverResult()
    matched: dict[str, tuple[str, str]] = {}  # qid → (how, detail), in order

    if qid:
        matched[qid.strip().upper()] = (QID_MATCH, qid)
    else:
        for platform, handle in (handles or {}).items():
            hits = wiki.find_by_account(fetcher, api, platform, handle)
            if not hits:
                result.unmatched_handles.append(f"{platform}:{handle}")
            for hit in hits:
                matched.setdefault(hit, (ACCOUNT_MATCH, f"{platform}:{handle}"))

    records = wiki.get_people(fetcher, api, list(matched)) if matched else []
    for record in records:
        how, detail = matched[record.qid]
        if record.is_human:
            result.candidates.append(Candidate(record, how, detail))
        else:
            result.skipped.append(record)

    # A name search is only a fallback, and its results are weak evidence.
    if not result.candidates and not qid:
        result.searched_by_name = True
        hits = [h["id"] for h in wiki.search_people(fetcher, api, name) if h["id"] not in matched]
        for record in wiki.get_people(fetcher, api, hits) if hits else []:
            if record.is_human:
                result.candidates.append(Candidate(record, NAME_MATCH, name))
            else:
                result.skipped.append(record)
    return result


def account_url(platform: str, handle: str) -> str:
    """Canonical profile URL for an account."""
    if platform == "wikipedia":
        return wiki.wikipedia_url(handle)
    if platform == "youtube":
        return youtube.channel_videos_url(handle).removesuffix("/videos")
    for plat, template in wiki.ACCOUNT_PROPERTIES.values():
        if plat == platform:
            return template.format(handle)
    return handle


@dataclass
class LinkOutcome:
    person: Person
    relinked_from: str | None = None
    dropped: list[Account] = field(default_factory=list)  # old record's accounts, now rejected


def link_record(
    store: ResearchStore,
    name: str,
    record: wiki.WikidataPerson,
    decide: Callable[[str, str], str | None],
) -> LinkOutcome:
    """
    Link a Wikidata *person* record to a research subject.

    `decide(platform, handle)` returns CONFIRMED / REJECTED, or None to leave
    an account as a candidate. Re-linking to a different record rejects the
    accounts the old record brought in and replaces its description.
    """
    if not record.is_human:
        raise ValueError(f"{record.qid} ({record.description or record.label}) is not a person")
    person = store.get_person(name) or store.add_person(name)
    found = list(record.accounts)
    if record.wikipedia_title:
        found.insert(0, ("wikipedia", record.wikipedia_title, wiki.wikipedia_url(record.wikipedia_title)))

    outcome = LinkOutcome(person)
    previous = person.wikidata_id
    if previous and previous != record.qid:
        keep = {(p, h) for p, h, _ in found}
        outcome.relinked_from = previous
        outcome.dropped = reject_wikidata_accounts(store, person, previous, keep)
        person.description = record.description  # the old one came from the wrong record
    else:
        person.description = person.description or record.description
    person.wikidata_id = record.qid
    store.update_person(person)

    for platform, handle, url in found:
        account = store.add_account(
            person.id, platform, handle, url, status=CANDIDATE, found_via=f"wikidata {record.qid}"
        )
        if account.status == CONFIRMED:
            continue
        status = decide(platform, handle)
        if status:
            store.set_account_status(account.id, status)
    return outcome


def reject_wikidata_accounts(
    store: ResearchStore, person: Person, qid: str, keep=frozenset()
) -> list[Account]:
    """Reject the accounts a Wikidata record brought in (except `keep`)."""
    dropped = []
    for account in store.accounts(person.id):
        if (
            account.found_via == f"wikidata {qid}"
            and account.status != REJECTED
            and (account.platform, account.handle) not in keep
        ):
            store.set_account_status(account.id, REJECTED)
            dropped.append(account)
    return dropped


def purge_rejected(store: ResearchStore, person: Person) -> int:
    """Delete every document collected from a rejected account."""
    return store.delete_documents(_rejected_document_ids(store, person))


def rejected_document_count(store: ResearchStore, person: Person) -> int:
    return len(_rejected_document_ids(store, person))


def _rejected_document_ids(store: ResearchStore, person: Person) -> set[int]:
    ids: set[int] = set()
    for account in store.accounts(person.id, status=REJECTED):
        ids.update(store.documents_from_account(account))
    return ids


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

    def wanted(platform: str) -> bool:
        return not options.only or platform in options.only

    if options.voice and voice_transcriber is None:
        if not person.voice_profile:
            raise ValueError(
                f"--voice needs a voice profile linked to {person.name} (research add NAME --voice PROFILE)"
            )
        voice_transcriber = youtube.voice_transcriber(config, person.voice_profile)
    voice = voice_transcriber if options.voice else None

    def run(key: str, fn, *args, **kwargs):
        try:
            stats = fn(*args, **kwargs)
        except (NetError, RuntimeError, ValueError) as e:
            stats = CollectStats(errors=[f"{key}: {e}"])
        results.setdefault(key, CollectStats()).merge(stats)

    base_ctx = ctx
    own_channels: set[str] = set()
    for account in store.accounts(person.id, status=CONFIRMED):
        platform = account.platform
        if not wanted(platform):
            continue
        ctx = base_ctx.for_account(account.id)
        if platform == "wikipedia":
            run("wikipedia", wiki.collect_wikipedia, ctx, account)
        elif platform == "youtube":
            if account.handle.startswith("UC"):
                own_channels.add(account.handle)
            run(
                "youtube",
                youtube.collect_channel,
                ctx,
                account,
                ytdlp=ytdlp,
                voice=voice,
                limit=options.max_videos,
            )
        elif platform == "twitch":
            if voice is None:
                results.setdefault("twitch", CollectStats()).errors.append(
                    "twitch VODs have no captions — use --voice"
                )
                continue
            run(
                "twitch",
                youtube.collect_channel,
                ctx,
                _as_url_account(account, f"https://www.twitch.tv/{account.handle}/videos?filter=archives"),
                ytdlp=ytdlp,
                voice=voice,
                limit=options.max_videos,
            )
        elif platform == "x":
            run("x", social.collect_x, ctx, account, limit=options.max_posts)
        elif platform == "reddit":
            run("reddit", social.collect_reddit, ctx, account, limit=options.max_posts)
        elif platform == "website":
            run("web", web.collect_url, ctx, account.url or account.handle, first_hand=True)
        else:
            results.setdefault(platform, CollectStats()).errors.append(
                f"{platform}: no supported collector (public API unavailable)"
            )

    ctx = base_ctx  # the rest isn't tied to one account
    if person.voice_profile and wanted("transcripts") and Path(config.transcripts_db).exists():
        run("transcripts", local.collect_transcripts, ctx, person.voice_profile)

    if options.appearances and wanted("youtube"):
        if voice is None:
            results.setdefault("appearances", CollectStats()).errors.append(
                "appearances need --voice (captions can't tell who is speaking)"
            )
        else:
            run(
                "appearances",
                youtube.collect_appearances,
                ctx,
                f'"{person.name}"',
                voice,
                own_channels,
                ytdlp=ytdlp,
                limit=options.appearances,
            )

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
    return replace(account, handle=url)
