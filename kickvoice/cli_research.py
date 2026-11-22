"""
`kickvoice research …` and `kickvoice debate` commands.

    research add NAME --x handle --youtube @handle --voice PROFILE
    research discover NAME             find official accounts via Wikidata (you confirm)
    research collect NAME              fetch their public record
    research analyze NAME              extract verified claims (uses the LLM)
    research brief NAME                pre-debate brief (markdown)
    debate --stream URL --opponent NAME
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from kickvoice import console

PLATFORM_OPTIONS = {
    "x": "X/Twitter username",
    "youtube": "YouTube channel id (UC…), @handle or URL",
    "reddit": "Reddit username",
    "twitch": "Twitch channel",
    "wikipedia": "English Wikipedia article title",
    "website": "Their own website/blog URL",
}


def _config(ctx):
    return ctx.obj["config"]


def _store(ctx):
    from kickvoice.research.store import ResearchStore

    return ResearchStore(_config(ctx).research_db)


def _person(store, name: str):
    person = store.get_person(name)
    if person is None:
        raise click.ClickException(
            f"No research subject '{name}'. Add one: kickvoice research add \"{name}\" --x HANDLE")
    return person


def _account_url(platform: str, handle: str) -> str:
    from kickvoice.research.sources.wiki import ACCOUNT_PROPERTIES, wikipedia_url

    if platform == "wikipedia":
        return wikipedia_url(handle)
    if platform == "youtube":
        from kickvoice.research.sources.youtube import channel_videos_url
        return channel_videos_url(handle).removesuffix("/videos")
    for plat, template in ACCOUNT_PROPERTIES.values():
        if plat == platform:
            return template.format(handle)
    return handle


def _llm(config):
    from kickvoice.research.llm import LLMError, make_llm

    try:
        if config.llm.provider == "anthropic":
            from kickvoice.research.llm import _api_key
            _api_key(config.llm.api_key_env or "ANTHROPIC_API_KEY")   # fail fast, before any work
        return make_llm(config)
    except LLMError as e:
        raise click.ClickException(str(e)) from e


@click.group()
def research():
    """Research a public figure's public record for debate prep."""


def _platform_options(fn):
    for platform, help_text in reversed(PLATFORM_OPTIONS.items()):
        fn = click.option(f"--{platform}", f"{platform}_handle", help=help_text)(fn)
    return fn


@research.command("add")
@click.argument("name")
@click.option("--voice", "voice_profile", help="Link their kickvoice voice profile.")
@click.option("--description", default="", help="Who they are (helps the model).")
@_platform_options
@click.pass_context
def research_add(ctx, name, voice_profile, description, **handles):
    """
    Add (or update) a research subject and the accounts you know are theirs.

    Accounts given here are marked confirmed. To find official accounts
    automatically, use `research discover`.

    \b
    Example:
      kickvoice research add "Destiny" --x TheOmniLiberal --youtube @destiny --voice Destiny
    """
    from kickvoice.research.store import CONFIRMED

    with _store(ctx) as store:
        person = store.get_person(name) or store.add_person(name, description=description)
        if voice_profile:
            person.voice_profile = voice_profile
        if description:
            person.description = description
        store.update_person(person)
        added = []
        for key, handle in handles.items():
            if handle:
                platform = key.removesuffix("_handle")
                store.add_account(person.id, platform, handle, _account_url(platform, handle),
                                  status=CONFIRMED, found_via="added by you")
                added.append(f"{platform}:{handle}")
    console.success(f"{person.name}" + (f" — accounts: {', '.join(added)}" if added else ""))
    if voice_profile:
        console.info(f"   voice profile: {voice_profile}")
    console.hint(f"   Next: kickvoice research discover \"{name}\"  (find more official accounts)")
    console.hint(f"         kickvoice research collect \"{name}\"")


@research.command("discover")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Confirm every account found on the chosen record.")
@_platform_options
@click.pass_context
def research_discover(ctx, name, yes, **handles):
    """
    Find NAME's official accounts via Wikidata, and confirm which to use.

    Wikidata records the accounts a person publicly claims, so this never
    links anonymous or look-alike accounts. Give a known handle (e.g. --x)
    for an exact match; otherwise it searches by name.
    """
    from kickvoice.pipelines.research import discover
    from kickvoice.research.net import Fetcher, NetError
    from kickvoice.research.store import CANDIDATE, CONFIRMED, REJECTED

    config = _config(ctx)
    known = {k.removesuffix("_handle"): v for k, v in handles.items() if v}
    try:
        candidates = discover(Fetcher(config), config, name, known)
    except NetError as e:
        raise click.ClickException(f"Wikidata lookup failed: {e}") from e
    if not candidates:
        console.info(f"No Wikidata record found for '{name}'. Add accounts yourself with "
                     "`research add`.")
        return

    interactive = sys.stdin.isatty() and not yes
    chosen = candidates[0]
    if len(candidates) > 1 or interactive:
        console.echo()
        for i, c in enumerate(candidates, 1):
            console.echo(f"  [{i}] {c.label} ({c.qid}) — {c.description or 'no description'}"
                         f" · {len(c.accounts)} account(s)")
        if interactive:
            pick = click.prompt("Which one is them? (0 = none)", type=int, default=1)
            if pick == 0 or pick > len(candidates):
                console.info("Nothing linked.")
                return
            chosen = candidates[pick - 1]
        elif len(candidates) > 1 and not known:
            raise click.ClickException(
                "Several people match that name; re-run in a terminal to choose, or pass a "
                "known handle (e.g. --x) for an exact match.")

    with _store(ctx) as store:
        person = store.get_person(name) or store.add_person(name)
        person.wikidata_id = chosen.qid
        person.description = person.description or chosen.description
        store.update_person(person)
        found = [(p, h, u) for p, h, u in chosen.accounts]
        if chosen.wikipedia_title:
            found.insert(0, ("wikipedia", chosen.wikipedia_title,
                             _account_url("wikipedia", chosen.wikipedia_title)))
        for platform, handle, url in found:
            acct = store.add_account(person.id, platform, handle, url, status=CANDIDATE,
                                     found_via=f"wikidata {chosen.qid}")
            if acct.status == CONFIRMED:
                continue
            if yes:
                status = CONFIRMED
            elif interactive:
                status = CONFIRMED if click.confirm(f"  Use {platform}: {handle}?", default=True) else REJECTED
            else:
                continue
            store.set_account_status(acct.id, status)
        accounts = store.accounts(person.id)
    _print_accounts(accounts)
    if any(a.status == CANDIDATE for a in accounts):
        console.hint(f"\n   Confirm candidates: kickvoice research accounts \"{name}\" --confirm ID")


def _print_accounts(accounts):
    console.echo()
    console.table(["ID", "Platform", "Account", "Status", "Found via"],
                  [[a.id, a.platform, a.handle, a.status, a.found_via] for a in accounts])


@research.command("accounts")
@click.argument("name")
@click.option("--confirm", "confirm_ids", multiple=True, type=int, help="Account id to confirm.")
@click.option("--reject", "reject_ids", multiple=True, type=int, help="Account id to reject.")
@click.pass_context
def research_accounts(ctx, name, confirm_ids, reject_ids):
    """List NAME's accounts; confirm or reject candidates."""
    from kickvoice.research.store import CONFIRMED, REJECTED

    with _store(ctx) as store:
        person = _person(store, name)
        valid = {a.id for a in store.accounts(person.id)}
        for ids, status in ((confirm_ids, CONFIRMED), (reject_ids, REJECTED)):
            for account_id in ids:
                if account_id not in valid:
                    raise click.ClickException(f"No account {account_id} for {person.name}")
                store.set_account_status(account_id, status)
        accounts = store.accounts(person.id)
    if not accounts:
        console.info("No accounts yet.")
        return
    _print_accounts(accounts)


@research.command("collect")
@click.argument("name")
@click.option("--only", help="Comma-separated sources: wikipedia,youtube,x,reddit,website,twitch,transcripts,web")
@click.option("--voice", is_flag=True,
              help="Transcribe videos and keep only lines matching their voice profile (slow, precise).")
@click.option("--appearances", type=int, default=0,
              help="Also search N videos on other channels (debates, podcasts). Needs --voice.")
@click.option("--web-search", type=int, default=0, help="Fetch N web search results about them (BRAVE_API_KEY).")
@click.option("--max-videos", type=int, help="Videos per channel [default: 25].")
@click.option("--max-posts", type=int, help="Posts per account [default: 1000].")
@click.pass_context
def research_collect(ctx, name, only, voice, appearances, web_search, max_videos, max_posts):
    """
    Collect NAME's public record from every confirmed account.

    Re-running only adds what's new. Nothing is collected from accounts you
    haven't confirmed.
    """
    from kickvoice.pipelines.research import CollectOptions, collect

    config = _config(ctx)
    options = CollectOptions(
        only=tuple(s.strip() for s in only.split(",")) if only else (),
        voice=voice, appearances=appearances, web_search=web_search,
        max_videos=max_videos, max_posts=max_posts)
    with _store(ctx) as store:
        person = _person(store, name)
        console.heading(f"\n🔎 Collecting {person.name}'s public record")
        try:
            results = collect(store, person, config, options, progress=console.info)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        stats = store.stats(person.id)
    console.echo()
    for source, s in results.items():
        line = f"   {source:<12} new {s.new:>4} · updated {s.updated:>3} · unchanged {s.unchanged:>4}"
        console.info(line)
        for err in s.errors[:5]:
            console.hint(f"      ! {err}")
        if len(s.errors) > 5:
            console.hint(f"      ! … {len(s.errors) - 5} more")
    if not results:
        console.warn("No confirmed accounts to collect from. See `research accounts`.")
    console.info(f"\n   On record: {stats['documents']} documents, {stats['chunks']} excerpts "
                 f"({stats['chunks'] - stats['analyzed']} not analysed yet)")
    if stats["chunks"] > stats["analyzed"]:
        console.hint(f"   Next: kickvoice research analyze \"{person.name}\"")


@research.command("import")
@click.argument("name")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--about", is_flag=True, help="The file is ABOUT them (not their own words).")
@click.option("--source", default="file", show_default=True, help="Label for where it came from.")
@click.pass_context
def research_import(ctx, name, path, about, source):
    """
    Import text you have: JSONL/CSV of posts (text,id,url,date), an X data
    export (tweets.js), or a .txt/.md file.
    """
    from kickvoice.research.net import Fetcher
    from kickvoice.research.sources.base import SourceContext
    from kickvoice.research.sources.local import import_file

    config = _config(ctx)
    with _store(ctx) as store:
        person = _person(store, name)
        ctx_ = SourceContext(config, Fetcher(config), store, person, console.info)
        try:
            stats = import_file(ctx_, path, first_hand=not about, source=source)
        except (ValueError, KeyError) as e:
            raise click.ClickException(f"Couldn't import {path}: {e}") from e
    console.success(f"Imported {stats.total} item(s) ({stats.new} new)")


@research.command("analyze")
@click.argument("name")
@click.option("--limit", type=int, help="Analyse at most this many excerpts this run.")
@click.option("-y", "--yes", is_flag=True, help="Don't ask for cost confirmation.")
@click.pass_context
def research_analyze(ctx, name, limit, yes):
    """Extract NAME's claims and arguments, with verified quotes (uses the LLM)."""
    from kickvoice.research.extract import estimate_tokens, extract_claims
    from kickvoice.research.llm import LLMError, make_embedder

    config = _config(ctx)
    with _store(ctx) as store:
        person = _person(store, name)
        pending = store.chunks_to_analyze(person.id, limit=limit)
        if not pending:
            console.info("Nothing new to analyse. Collect more with `research collect`.")
            return
        tokens_in, tokens_out = estimate_tokens(pending, config)
        console.info(f"{len(pending)} excerpt(s) to analyse with {config.llm.analysis_model}: "
                     f"~{tokens_in:,} input + ~{tokens_out:,} output tokens.")
        if not yes:
            if not sys.stdin.isatty():
                raise click.ClickException("Confirm the cost with --yes when not in a terminal.")
            if not click.confirm("Proceed?", default=True):
                return
        llm = _llm(config)
        stats = extract_claims(store, person, llm, config, limit_chunks=limit,
                               progress=console.info)
        embedded = 0
        embedder = make_embedder(config)
        if embedder is not None:
            todo = store.claims_without_embedding(person.id)
            if todo:
                try:
                    vectors = embedder.embed([f"{c.topic}: {c.claim}" for c in todo])
                    store.set_claim_embeddings(zip([c.id for c in todo], vectors))
                    embedded = len(todo)
                except LLMError as e:
                    console.warn(f"Embeddings skipped: {e}")
    console.success(f"{stats.claims} claim(s) extracted from {stats.chunks} excerpt(s)"
                    + (f"; {embedded} embedded" if embedded else ""))
    if stats.rejected_quotes:
        console.hint(f"   {stats.rejected_quotes} claim(s) dropped: quote not found verbatim in source")
    if stats.failed_batches:
        console.warn(f"{stats.failed_batches} batch(es) failed and will be retried next run: "
                     f"{stats.errors[0]}")
    console.hint(f"   tokens used: {stats.input_tokens:,} in / {stats.output_tokens:,} out")


@research.command("list")
@click.pass_context
def research_list(ctx):
    """All research subjects."""
    with _store(ctx) as store:
        rows = []
        for p in store.list_persons():
            s = store.stats(p.id)
            rows.append([p.name, p.voice_profile or "—", s["documents"], s["claims"],
                         len(store.accounts(p.id, status="confirmed"))])
    if not rows:
        console.info("No research subjects. Add one: kickvoice research add NAME --x HANDLE")
        return
    console.echo()
    console.table(["Name", "Voice", "Documents", "Claims", "Accounts"], rows)
    console.echo()


@research.command("show")
@click.argument("name")
@click.option("--topic", help="Show claims on this topic.")
@click.pass_context
def research_show(ctx, name, topic):
    """What's on record for NAME: sources, accounts, topics (or one topic's claims)."""
    from kickvoice.research.store import AUTHORSHIP_LABELS

    with _store(ctx) as store:
        person = _person(store, name)
        stats = store.stats(person.id)
        console.heading(f"\n{person.name}" + (f" — {person.description}" if person.description else ""))
        if person.voice_profile:
            console.info(f"Voice profile: {person.voice_profile}")
        if person.wikidata_id:
            console.info(f"Wikidata: {person.wikidata_id}")
        console.info(f"Documents: {stats['documents']} "
                     f"({', '.join(f'{k} {v}' for k, v in sorted(stats['by_source'].items())) or '—'})")
        console.info(f"Excerpts: {stats['chunks']} ({stats['analyzed']} analysed) · "
                     f"Claims: {stats['claims']}")
        if topic:
            claims = store.claims(person.id, topic=topic)
            console.heading(f"\nClaims on '{topic}' ({len(claims)})")
            for c in claims:
                flag = "" if c.sincere else click.style(" [insincere]", fg="yellow")
                console.echo(f"- {c.claim}{flag}")
                console.hint(f"    \"{c.quote}\" — {(c.published_at or 'undated')[:10]}, "
                             f"{AUTHORSHIP_LABELS.get(c.authorship, c.authorship)}"
                             + (f", {c.url}" if c.url else ""))
            return
        topics = store.topics(person.id)
        if topics:
            console.heading("\nTop topics")
            for t, n in topics[:15]:
                console.info(f"  {n:>4}  {t}")
        accounts = store.accounts(person.id)
        if accounts:
            _print_accounts(accounts)


@research.command("search")
@click.argument("name")
@click.argument("query")
@click.option("--limit", type=int, default=10, show_default=True)
@click.pass_context
def research_search(ctx, name, query, limit):
    """Search NAME's record (claims and raw excerpts)."""
    from kickvoice.research.llm import make_embedder
    from kickvoice.research.retrieve import Retriever
    from kickvoice.research.store import AUTHORSHIP_LABELS

    with _store(ctx) as store:
        person = _person(store, name)
        hits = Retriever(store, person, make_embedder(_config(ctx))).search(query, k=limit)
    if not hits:
        console.info("No matches.")
        return
    for ev in hits:
        label = f"[{ev.ref}] {ev.source} {(ev.published_at or 'undated')[:10]} · " \
                f"{AUTHORSHIP_LABELS.get(ev.authorship, ev.authorship)}"
        console.echo(click.style(label, fg="bright_black"))
        if ev.kind == "claim":
            console.echo(f"  {ev.text}")
        console.echo(f"  \"{ev.quote}\"")
        if ev.url:
            console.hint(f"  {ev.url}")


@research.command("brief")
@click.argument("name")
@click.option("--topic", help="Only this topic.")
@click.option("--max-topics", type=int, default=8, show_default=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path),
              help="Write the brief here (markdown) [default: print].")
@click.pass_context
def research_brief(ctx, name, topic, max_topics, out_path):
    """Pre-debate brief: positions, quotes, contradictions, expected arguments, counters."""
    from kickvoice.research.brief import build_brief

    config = _config(ctx)
    with _store(ctx) as store:
        person = _person(store, name)
        if not store.topics(person.id):
            raise click.ClickException(
                f"No analysed claims for {person.name} yet — run `research collect` and "
                "`research analyze` first.")
        markdown = build_brief(store, person, _llm(config), config, topic=topic,
                               max_topics=max_topics, progress=console.hint)
    if out_path:
        out_path.write_text(markdown, encoding="utf-8")
        console.success(f"Brief written to {out_path}")
    else:
        console.echo(markdown)


@research.command("remove")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True)
@click.pass_context
def research_remove(ctx, name, yes):
    """Delete NAME and everything collected about them."""
    with _store(ctx) as store:
        person = _person(store, name)
        if not yes and not click.confirm(f"Delete {person.name} and all collected data?"):
            console.info("Cancelled.")
            return
        store.remove_person(person.id)
    console.success(f"Removed {person.name}")


# ── debate ───────────────────────────────────────────────────────

@click.command()
@click.option("--stream", help="Stream URL, e.g. https://kick.com/channel.")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path),
              help="Local recording (for practice/replays).")
@click.option("--opponent", required=True, help="Research subject you're debating.")
@click.option("--notes", type=click.Path(exists=True, path_type=Path),
              help="Your prep notes (markdown/text): facts and talking points to draw on.")
@click.option("--me", help="Your own voice profile (never treated as the opponent).")
@click.option("--any-speaker", is_flag=True,
              help="Respond to every voice except --me (if the opponent isn't enrolled).")
@click.option("--duration", type=float, help="Stop after this many seconds.")
@click.pass_context
def debate(ctx, stream, file_path, opponent, notes, me, any_speaker, duration):
    """
    Live counter-arguments while you debate OPPONENT.

    Transcribes the stream, detects the opponent's points by their voice,
    and suggests replies grounded in their own public record and your
    notes. Contradictions are only flagged against things they said
    themselves, with date and link.

    \b
    Example:
      kickvoice debate --stream https://kick.com/somechannel --opponent "Destiny" --notes prep.md
    """
    from kickvoice.pipelines.debate import (
        DebateAssistant, PointAggregator, debate_log_path, opponent_matcher,
    )
    from kickvoice.pipelines.transcribe import TranscriptionPipeline, speaker_label
    from kickvoice.research.counter import CounterEngine, load_notes
    from kickvoice.research.llm import make_embedder
    from kickvoice.research.retrieve import Retriever
    from kickvoice.transcripts.asr import ASRUnavailable

    config = _config(ctx)
    if bool(stream) == bool(file_path):
        raise click.UsageError("Provide exactly one of --stream or --file.")
    config.transcribe.chunk_duration_s = config.debate.chunk_duration_s
    llm = _llm(config)

    with _store(ctx) as store:
        person = _person(store, opponent)
        if not person.voice_profile and not any_speaker:
            raise click.ClickException(
                f"{person.name} has no voice profile, so their lines can't be recognised. "
                f"Link one (research add \"{person.name}\" --voice PROFILE) or use --any-speaker.")
        stats = store.stats(person.id)
        if stats["claims"] == 0:
            console.warn(f"No analysed claims for {person.name} — suggestions will rely on raw "
                         "excerpts and your notes. Run `research analyze` for better results.")
        engine = CounterEngine(store, person, llm, config,
                               retriever=Retriever(store, person, make_embedder(config)),
                               notes=load_notes(notes))
        aggregator = PointAggregator(
            opponent_matcher(person.voice_profile, me, any_speaker), speaker_label,
            min_words=config.debate.min_point_words)
        source = stream or str(file_path)
        log_path = debate_log_path(config, source)
        assistant = DebateAssistant(engine, aggregator, config, log_path)

        console.heading(f"\n🥊 Debate assist vs {person.name} — {stats['claims']} claims, "
                        f"{stats['documents']} documents on record")
        console.hint(f"   Suggestions are logged to {log_path}")
        assistant.start()
        try:
            TranscriptionPipeline(
                stream_url=stream, file_path=str(file_path) if file_path else None,
                config=config, on_utterance=assistant.on_utterance,
            ).run(max_duration_s=duration)
        except (ASRUnavailable, RuntimeError, ValueError) as e:
            raise click.ClickException(str(e)) from e
        finally:
            console.hint("\n   Finishing the last suggestion…")
            assistant.stop()
    console.info(f"\n   {len(assistant.suggestions)} suggestion(s)"
                 + (f", {assistant.skipped} stale point(s) skipped" if assistant.skipped else "")
                 + f". Log: {log_path}")
