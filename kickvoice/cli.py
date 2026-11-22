"""
kickvoice command line.

    kickvoice enroll "DrDisrespect" --file clip.wav
    kickvoice monitor --stream https://kick.com/channel
    kickvoice extract-profiles --stream https://kick.com/channel
    kickvoice review
    kickvoice transcribe --stream https://kick.com/channel
    kickvoice transcripts show SESSION
    kickvoice research add "Name" --x handle --voice PROFILE
    kickvoice debate --stream https://kick.com/channel --opponent "Name"
    kickvoice profiles list

Commands stay thin: they parse options, call a pipeline, and print the
result. The heavy imports happen inside each command so `--help` is instant.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from kickvoice import __version__, console
from kickvoice.cli_research import debate, research
from kickvoice.config import Config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_config(ctx: click.Context) -> Config:
    return ctx.obj["config"]


def split_multi(values: tuple[str, ...]) -> list[str]:
    """Allow both repeated options and comma-separated lists."""
    out: list[str] = []
    for value in values:
        out += [v.strip() for v in str(value).split(",") if v.strip()]
    return out


def resolve_name(argument: str | None, option: str | None, what: str = "NAME") -> str:
    """
    Accept both `kickvoice profiles info xQc` and the older
    `kickvoice profiles info --name xQc`.
    """
    if argument and option and argument != option:
        raise click.UsageError(f"Give {what} once, not both as an argument and --name.")
    name = argument or option
    if not name:
        raise click.UsageError(f"Missing {what}.")
    return name


def require_tty(what: str):
    if not sys.stdin.isatty():
        raise click.ClickException(f"{what} needs an interactive terminal.")


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="kickvoice")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
@click.option("--config", "config_path", type=click.Path(path_type=Path),
              help="Config file [default: ~/.kickvoice/config.json].")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: Path | None):
    """Speaker identification on Kick.com live streams."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(config_path)


# ── enroll ───────────────────────────────────────────────────────

@cli.command()
@click.argument("name", required=False)
@click.option("--name", "name_opt", help="Speaker name (alternative to the NAME argument).")
@click.option("--file", "files", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Audio/video file (repeatable).")
@click.option("--youtube", "--url", "urls", multiple=True,
              help="Video URL to download with yt-dlp (repeatable, or comma-separated).")
@click.option("--notes", default="", help="Free-text note stored with the profile.")
@click.pass_context
def enroll(ctx, name: str | None, name_opt: str | None,
           files: tuple[Path, ...], urls: tuple[str, ...], notes: str):
    """
    Create a voice profile for NAME from known audio.

    Two or three sources covering different moods (calm, excited, shouting)
    make a noticeably more robust profile.

    \b
    Examples:
      kickvoice enroll "DrDisrespect" --file clip.wav
      kickvoice enroll "xQc" --youtube https://youtu.be/AAA --youtube https://youtu.be/BBB
    """
    from kickvoice.pipelines.enroll import Enroller

    name = resolve_name(name, name_opt)
    sources = [str(f) for f in files] + split_multi(urls)
    if not sources:
        raise click.UsageError("Give at least one --file or --youtube source.")

    def progress(source, i, total):
        console.info(f"   [{i}/{total}] {source}")

    console.heading(f"\n🎤 Enrolling '{name}' from {len(sources)} source(s)")
    try:
        result = Enroller(config=get_config(ctx)).enroll(
            name, sources, notes=notes, on_progress=progress)
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e

    console.success(f"Enrolled '{name}' — {result.n_segments} segments, "
                    f"{console.fmt_duration(result.speech_seconds)} of speech")


# ── monitor ──────────────────────────────────────────────────────

@cli.command()
@click.option("--stream", help="Stream URL, e.g. https://kick.com/channel.")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path),
              help="Local audio/video file (for testing).")
@click.option("--threshold", type=float, help="Match threshold override [default: 0.60].")
@click.pass_context
def monitor(ctx, stream: str | None, file_path: Path | None, threshold: float | None):
    """
    Watch a stream and report who is speaking.

    \b
    Example:
      kickvoice monitor --stream https://kick.com/somechannel
    """
    from kickvoice.pipelines.monitor import Monitor

    config = get_config(ctx)
    if bool(stream) == bool(file_path):
        raise click.UsageError("Provide exactly one of --stream or --file.")
    if threshold is not None:
        config.identify.match_threshold = threshold

    try:
        Monitor(stream_url=stream, file_path=str(file_path) if file_path else None,
                config=config).run()
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


# ── extract-profiles ─────────────────────────────────────────────

@cli.command("extract-profiles")
@click.option("--stream", help="Stream URL, e.g. https://kick.com/channel.")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path),
              help="Local audio/video file (VOD, recording).")
@click.option("--duration", type=float,
              help="Seconds to listen [default: 300 for streams, whole file for --file; "
                   "0 = until the stream ends / Ctrl+C].")
@click.option("--min-speech", type=float,
              help="Ignore voices with less total speech, in seconds [default: 20].")
@click.option("--cluster-threshold", type=float,
              help="Voice grouping cut-off, as cosine distance [default: 0.70]. "
                   "Lower splits more, higher merges more.")
@click.option("--threshold", type=float,
              help="Similarity needed to match a known profile [default: 0.60].")
@click.option("--no-review", is_flag=True, help="Only collect voices; approve later.")
@click.pass_context
def extract_profiles(ctx, stream, file_path, duration, min_speech,
                     cluster_threshold, threshold, no_review):
    """
    Discover the speakers on a stream and turn them into profiles.

    Listens, groups the speech by voice, matches each voice against known
    profiles, then asks you to approve every new or updated profile.

    \b
    Examples:
      kickvoice extract-profiles --stream https://kick.com/somechannel
      kickvoice extract-profiles --file vod.mp4 --min-speech 10
    """
    from kickvoice.pipelines.extract import ProfileExtractor

    config = get_config(ctx)
    if bool(stream) == bool(file_path):
        raise click.UsageError("Provide exactly one of --stream or --file.")
    if threshold is not None:
        config.identify.match_threshold = threshold
    if min_speech is not None:
        config.extract.min_speaker_speech_s = min_speech
    if cluster_threshold is not None:
        config.extract.cluster_distance_threshold = cluster_threshold
    if duration is None and file_path:
        duration = 0      # files: the whole thing unless told otherwise

    try:
        extractor = ProfileExtractor(
            stream_url=stream, file_path=str(file_path) if file_path else None, config=config)
        session = extractor.run(max_duration_s=duration)
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if session is None:
        return
    if no_review or not sys.stdin.isatty():
        if not no_review:
            console.hint("\n   Not an interactive terminal — nothing saved as a profile yet.")
        console.hint(f"   Review later with: ./run.sh review --session {session.id}")
        return
    run_review(session, config)


# ── review ───────────────────────────────────────────────────────

@cli.command()
@click.option("--session", "session_ref",
              help="Session id, id prefix or path [default: newest with pending voices].")
@click.option("--list", "list_sessions_flag", is_flag=True, help="List extraction sessions.")
@click.pass_context
def review(ctx, session_ref: str | None, list_sessions_flag: bool):
    """
    Approve speakers found by extract-profiles.

    \b
    Examples:
      kickvoice review
      kickvoice review --list
      kickvoice review --session 20250601-201500
    """
    from kickvoice.pipelines.session import find_session, list_sessions

    config = get_config(ctx)
    root = config.extractions_dir

    if list_sessions_flag:
        sessions = list_sessions(root)
        if not sessions:
            console.info("No extraction sessions.")
            return
        console.echo()
        console.table(
            ["Session", "Speakers", "Pending", "Audio"],
            [[s.id, len(s.candidates), len(s.pending),
              console.fmt_duration(s.captured_seconds)] for s in sessions],
        )
        console.echo()
        return

    session = find_session(root, session_ref)
    if session is None:
        if session_ref:
            raise click.ClickException(f"Session not found (or ambiguous): {session_ref}")
        console.info("Nothing to review — no sessions with pending speakers.")
        return
    if not session.pending:
        console.info(f"All speakers in {session.id} have already been reviewed.")
        return
    require_tty("review")
    run_review(session, config)


def run_review(session, config: Config):
    from kickvoice.pipelines.review import Reviewer

    Reviewer(session, config=config).run()


# ── transcribe ───────────────────────────────────────────────────

ATTRIBUTION_CHOICE = click.Choice(["confident", "probable", "unknown"])


@cli.command()
@click.option("--stream", help="Stream URL, e.g. https://kick.com/channel.")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path),
              help="Local audio/video file (VOD, recording).")
@click.option("--duration", type=float,
              help="Seconds to transcribe [default: until the stream/file ends].")
@click.option("--model", help="Whisper model: tiny/base/small/medium/large-v3 [default: small].")
@click.option("--language", help="Spoken language code, e.g. en [default: auto-detect].")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), help="Run Whisper on CPU or GPU.")
@click.option("--drop-unknown", is_flag=True,
              help="Don't store text from unrecognised voices.")
@click.option("-q", "--quiet", is_flag=True, help="Don't print lines as they're transcribed.")
@click.pass_context
def transcribe(ctx, stream, file_path, duration, model, language, device, drop_unknown, quiet):
    """
    Transcribe a stream and label every line with who said it.

    Each line gets an attribution level: confident (strong voice match on
    enough audio), probable (matched, but short or contested), or unknown.
    Only enrolled speakers can be recognised.

    \b
    Examples:
      kickvoice transcribe --stream https://kick.com/somechannel --duration 1800
      kickvoice transcribe --file vod.mp4 --language en
    """
    from kickvoice.pipelines.transcribe import TranscriptionPipeline, print_utterance
    from kickvoice.transcripts.asr import ASRUnavailable

    config = get_config(ctx)
    if bool(stream) == bool(file_path):
        raise click.UsageError("Provide exactly one of --stream or --file.")
    tcfg = config.transcribe
    if model:
        tcfg.asr_model = model
    if language:
        tcfg.language = language
    if device:
        tcfg.asr_device = device
        if device == "cuda" and tcfg.asr_compute_type == "int8":
            tcfg.asr_compute_type = "float16"
    if drop_unknown:
        tcfg.keep_unattributed = False

    try:
        TranscriptionPipeline(
            stream_url=stream, file_path=str(file_path) if file_path else None,
            config=config, on_utterance=None if quiet else print_utterance,
        ).run(max_duration_s=duration)
    except (ASRUnavailable, RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e


@cli.group()
def transcripts():
    """Read, search, export and re-attribute transcripts."""


def open_transcripts(ctx):
    from kickvoice.transcripts.store import TranscriptStore

    return TranscriptStore(get_config(ctx).transcripts_db)


def resolve_session(store, ref: str):
    session = store.get_session(ref)
    if session is None:
        raise click.ClickException(f"Transcript session not found (or ambiguous): {ref}")
    return session


@transcripts.command("list")
@click.pass_context
def transcripts_list(ctx):
    """Show all transcribed sessions."""
    with open_transcripts(ctx) as store:
        sessions = store.list_sessions()
    if not sessions:
        console.info("No transcripts yet.")
        console.hint("   Create one: ./run.sh transcribe --stream URL")
        return
    console.echo()
    console.table(
        ["Session", "Audio", "Lines", "Speakers"],
        [[s.id, console.fmt_duration(s.captured_seconds), s.n_utterances,
          ", ".join(s.speakers) or "—"] for s in sessions],
    )
    console.echo()


@transcripts.command("show")
@click.argument("session_ref")
@click.option("--speaker", help="Only this speaker's lines.")
@click.option("--min-attribution", type=ATTRIBUTION_CHOICE, default="unknown", show_default=True,
              help="Hide lines attributed with less confidence than this.")
@click.pass_context
def transcripts_show(ctx, session_ref, speaker, min_attribution):
    """Print a session's transcript."""
    from kickvoice.pipelines.transcribe import print_utterance

    with open_transcripts(ctx) as store:
        session = resolve_session(store, session_ref)
        utterances = store.utterances(session.id, speaker, min_attribution)
    console.heading(f"\n{session.id} · {session.source} · started {session.started_at}")
    console.hint(f"ASR: {session.asr_model} · voices: {session.embedding_model}\n")
    for u in utterances:
        print_utterance(u)
    if not utterances:
        console.info("(no matching lines)")


@transcripts.command("search")
@click.argument("query")
@click.option("--speaker", help="Only this speaker's lines.")
@click.option("--min-attribution", type=ATTRIBUTION_CHOICE, default="probable", show_default=True)
@click.option("--limit", type=int, default=30, show_default=True)
@click.pass_context
def transcripts_search(ctx, query, speaker, min_attribution, limit):
    """
    Find lines containing QUERY across all sessions.

    \b
    Example:
      kickvoice transcripts search "gambling" --speaker xQc
    """
    with open_transcripts(ctx) as store:
        hits = store.search(query, speaker, min_attribution, limit)
        sessions = {s.id: s for s in store.list_sessions()}
    if not hits:
        console.info("No matches.")
        return
    from kickvoice.pipelines.transcribe import speaker_label

    for u in hits:
        who = speaker_label(u).rstrip("?")
        if u.speaker:
            who += f" ({u.attribution})"
        console.echo(f"{click.style(u.session_id, fg='bright_black')} "
                      f"[{console.fmt_clock(u.start_s)}] {who}: {u.text}")
    console.hint(f"\n{len(hits)} match(es) in {len({u.session_id for u in hits})} session(s) "
                 f"of {len(sessions)}")


@transcripts.command("speakers")
@click.pass_context
def transcripts_speakers(ctx):
    """How much transcribed speech is on record for each speaker."""
    with open_transcripts(ctx) as store:
        rows = store.speaker_summary()
    if not rows:
        console.info("No transcripts yet.")
        return
    console.echo()
    console.table(
        ["Speaker", "Lines", "Speech", "Confident speech", "Sessions"],
        [[r["speaker"] or "(unknown)", r["utterances"], console.fmt_duration(r["seconds"] or 0),
          console.fmt_duration(r["confident_seconds"] or 0), r["sessions"]] for r in rows],
    )
    console.echo()


@transcripts.command("export")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path),
              help="File to write (.jsonl, or .txt for plain text).")
@click.option("--session", "session_ref", help="Only this session [default: all].")
@click.option("--speaker", help="Only this speaker's lines.")
@click.option("--min-attribution", type=ATTRIBUTION_CHOICE, default="unknown", show_default=True)
@click.pass_context
def transcripts_export(ctx, out_path, session_ref, speaker, min_attribution):
    """
    Write transcripts to JSON Lines (one utterance per line) or plain text.

    JSONL is the hand-off format for analysis: each line carries the speaker,
    attribution level, timestamps, source and text.
    """
    import json

    from kickvoice.pipelines.transcribe import speaker_label

    count = 0
    with open_transcripts(ctx) as store:
        sessions = ([resolve_session(store, session_ref)] if session_ref
                    else store.list_sessions()[::-1])
        with open(out_path, "w", encoding="utf-8") as f:
            for session in sessions:
                for u in store.utterances(session.id, speaker, min_attribution):
                    if out_path.suffix == ".txt":
                        f.write(f"[{session.id} {console.fmt_clock(u.start_s)}] "
                                f"{speaker_label(u)}: {u.text}\n")
                    else:
                        f.write(json.dumps(u.to_dict(session), ensure_ascii=False) + "\n")
                    count += 1
    console.success(f"Exported {count} line(s) to {out_path}")


@transcripts.command("reattribute")
@click.option("--session", "session_ref", help="Only this session [default: all].")
@click.pass_context
def transcripts_reattribute(ctx, session_ref):
    """
    Re-match stored lines against the current profiles.

    Run this after enrolling someone new, fixing a profile, or changing the
    match thresholds. Uses the stored voice embeddings — no audio needed.
    """
    from kickvoice.speakers.profiles import ProfileStore
    from kickvoice.transcripts.refine import refine_session

    config = get_config(ctx)
    names, matrix = ProfileStore(config=config).embedding_matrix()
    changed = unchanged = 0
    with open_transcripts(ctx) as store:
        sessions = [resolve_session(store, session_ref)] if session_ref else store.list_sessions()
        for s in sessions:
            if s.embedding_model and s.embedding_model != config.embedding.model_source:
                raise click.ClickException(
                    f"Session {s.id} was voice-matched with '{s.embedding_model}', "
                    f"not '{config.embedding.model_source}' — embeddings aren't comparable.")
        for s in sessions:
            result = refine_session(store, s.id, names, matrix, config)
            changed += result.changed
            unchanged += result.unchanged
    console.success(f"Re-attributed {len(sessions)} session(s): {changed} line(s) changed, "
                    f"{unchanged} unchanged")


@transcripts.command("delete")
@click.argument("session_ref")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def transcripts_delete(ctx, session_ref, yes):
    """Delete a transcript session and all its lines."""
    with open_transcripts(ctx) as store:
        session = resolve_session(store, session_ref)
        if not yes and not click.confirm(f"Delete transcript {session.id}?"):
            console.info("Cancelled.")
            return
        store.delete_session(session.id)
    console.success(f"Deleted transcript {session.id}")


# ── profiles ─────────────────────────────────────────────────────

@cli.group()
def profiles():
    """List, inspect, remove, export and import speaker profiles."""


def open_store(ctx) -> "object":
    from kickvoice.speakers.profiles import ProfileStore

    return ProfileStore(config=get_config(ctx))


@profiles.command("list")
@click.pass_context
def profiles_list(ctx):
    """Show all enrolled speakers."""
    store = open_store(ctx)
    if not store.count:
        console.info("No enrolled speakers.")
        console.hint("   Add one: ./run.sh enroll NAME --file audio.wav")
        return
    rows = []
    for p in store.list_profiles():
        sources = ", ".join(p.source_urls[:2])
        if len(p.source_urls) > 2:
            sources += f" (+{len(p.source_urls) - 2} more)"
        rows.append([p.name, p.n_segments, console.fmt_duration(p.speech_seconds),
                     p.enrolled_at[:19], sources])
    console.echo()
    console.table(["Name", "Segments", "Speech", "Enrolled", "Sources"], rows)
    console.echo()


@profiles.command("info")
@click.argument("name", required=False)
@click.option("--name", "name_opt", help="Speaker name (alternative to the NAME argument).")
@click.pass_context
def profiles_info(ctx, name: str | None, name_opt: str | None):
    """Show everything stored about NAME."""
    name = resolve_name(name, name_opt)
    profile = open_store(ctx).find_profile(name)
    if not profile:
        raise click.ClickException(f"Profile not found: {name}")
    console.echo()
    console.heading(f"Speaker: {profile.name}")
    console.info(f"Enrolled: {profile.enrolled_at}")
    if profile.updated_at:
        console.info(f"Updated:  {profile.updated_at}")
    console.info(f"Segments: {profile.n_segments}")
    if profile.speech_seconds:
        console.info(f"Speech:   {console.fmt_duration(profile.speech_seconds)}")
    if profile.embedding_model:
        console.info(f"Model:    {profile.embedding_model}")
    console.info("Sources:")
    for url in profile.source_urls:
        console.info(f"  - {url}")
    if profile.notes:
        console.info(f"Notes: {profile.notes}")
    console.echo()


@profiles.command("remove")
@click.argument("name", required=False)
@click.option("--name", "name_opt", help="Speaker name (alternative to the NAME argument).")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def profiles_remove(ctx, name: str | None, name_opt: str | None, yes: bool):
    """Delete the profile for NAME."""
    name = resolve_name(name, name_opt)
    store = open_store(ctx)
    profile = store.find_profile(name)
    if not profile:
        raise click.ClickException(f"Profile not found: {name}")
    if not yes and not click.confirm(f"Remove profile '{profile.name}'?"):
        console.info("Cancelled.")
        return
    store.remove_profile(profile.name)
    console.success(f"Removed profile: {profile.name}")


@profiles.command("export")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path),
              help="Pack file to write.")
@click.option("--name", "names", multiple=True,
              help="Profile to include (repeatable, or comma-separated) [default: all].")
@click.pass_context
def profiles_export(ctx, out_path: Path, names: tuple[str, ...]):
    """
    Write profiles to a shareable pack file.

    \b
    Example:
      kickvoice profiles export --out streamers.json
    """
    from kickvoice.speakers.packs import PackError, export_pack

    store = open_store(ctx)
    try:
        count = export_pack(store, out_path, names=split_multi(names) or None,
                            config=get_config(ctx))
    except PackError as e:
        raise click.ClickException(str(e)) from e
    console.success(f"Exported {count} profile(s) to {out_path}")


@profiles.command("import")
@click.argument("pack", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--file", "pack_opt", type=click.Path(exists=True, path_type=Path),
              help="Pack file (alternative to the PACK argument).")
@click.option("--on-conflict", type=click.Choice(["skip", "overwrite", "merge"]),
              default="skip", show_default=True,
              help="What to do when a profile name already exists.")
@click.pass_context
def profiles_import(ctx, pack: Path | None, pack_opt: Path | None, on_conflict: str):
    """
    Load profiles from a pack file.

    \b
    Example:
      kickvoice profiles import streamers.json --on-conflict merge
    """
    from kickvoice.speakers.packs import PackError, import_pack

    pack = pack or pack_opt
    if not pack:
        raise click.UsageError("Give a pack file to import.")
    store = open_store(ctx)
    try:
        result = import_pack(store, pack, on_conflict=on_conflict, config=get_config(ctx))
    except PackError as e:
        raise click.ClickException(str(e)) from e

    for label, names in (("Added", result.added), ("Overwrote", result.overwritten),
                         ("Merged", result.merged), ("Skipped (already exist)", result.skipped)):
        if names:
            console.info(f"{label}: {', '.join(names)}")
    if result.skipped:
        console.hint("   (use --on-conflict merge|overwrite to update existing profiles)")
    if not any((result.added, result.overwritten, result.merged, result.skipped)):
        console.warn("Pack contained no profiles.")


cli.add_command(research)
cli.add_command(debate)


@cli.command("mcp")
@click.pass_context
def mcp_server(ctx):
    """
    Run kickvoice as an MCP server (stdio) for Claude Desktop, Claude Code,
    or any MCP host.

    The agent gets tools to listen to a stream, recognise and save voices,
    research speakers, and draft chat replies for you to paste.

    \b
    Claude Code:
      claude mcp add kickvoice -- /path/to/kickvoice/run.sh mcp
    """
    # stdout is the protocol channel from here on: everything human goes to stderr.
    console.use_stderr()
    try:
        from kickvoice.agent.server import serve_stdio
    except ImportError as e:
        raise click.ClickException(
            f"MCP support needs the 'mcp' package: pip install -e '.[agent]' ({e})") from e
    serve_stdio(get_config(ctx))


def main():
    # prog_name keeps the help text saying "kickvoice" however it was started
    # (./run.sh, python -m kickvoice, or the installed entry point).
    cli(obj={}, prog_name="kickvoice")


if __name__ == "__main__":
    main()
