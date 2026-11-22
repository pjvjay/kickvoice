#!/usr/bin/env python3
"""
kickvoice CLI — Speaker identification on live streams.

Usage:
    kickvoice enroll --name "Speaker" --file audio.wav
    kickvoice enroll --name "Speaker" --youtube "https://youtube.com/..."
    kickvoice monitor --stream "https://kick.com/channel"
    kickvoice monitor --file recording.wav
    kickvoice extract-profiles --stream "https://kick.com/channel"
    kickvoice review
    kickvoice profiles list
    kickvoice profiles remove --name "Speaker"
    kickvoice profiles export --out pack.json
    kickvoice profiles import --file pack.json
"""
import argparse
import logging
import sys
from pathlib import Path

from kickvoice.config import Config


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_enroll(args, config: Config):
    """Enroll a speaker."""
    from kickvoice.enroll.enroll import Enroller

    enroller = Enroller(config=config)

    if args.file:
        n = enroller.enroll_from_file(
            name=args.name,
            audio_path=args.file,
            notes=args.notes or "",
        )
        print(f"✅ Enrolled '{args.name}' from file ({n} segments)")

    elif args.youtube:
        urls = [u.strip() for u in args.youtube.split(",")]
        n = enroller.enroll_from_youtube(
            name=args.name,
            urls=urls,
            notes=args.notes or "",
        )
        print(f"✅ Enrolled '{args.name}' from {len(urls)} YouTube video(s) ({n} segments)")

    else:
        print("Error: provide --file or --youtube", file=sys.stderr)
        sys.exit(1)


def cmd_monitor(args, config: Config):
    """Start the live monitoring pipeline."""
    from kickvoice.live.monitor import Monitor

    monitor = Monitor(
        stream_url=args.stream,
        file_path=args.file,
        config=config,
    )
    monitor.run()


def cmd_extract(args, config: Config):
    """Discover speakers on a stream/file, then review them."""
    from kickvoice.extract.extractor import ProfileExtractor

    try:
        extractor = ProfileExtractor(
            stream_url=args.stream, file_path=args.file, config=config,
        )
        duration = args.duration
        if duration is None and args.file:
            duration = 0   # files: use the whole thing unless told otherwise
        session = extractor.run(max_duration_s=duration)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)

    if session is None:
        return
    if args.no_review:
        print(f"\n   Review later with: ./run.sh review --session {session.id}")
        return
    if not sys.stdin.isatty():
        print("\n   Not an interactive terminal — nothing was saved as a profile yet.")
        print(f"   Review later with: ./run.sh review --session {session.id}")
        return
    _run_review(session, config)


def cmd_review(args, config: Config):
    """Approve/reject speakers from a previous extract-profiles run."""
    from kickvoice.extract.extractor import fmt_duration
    from kickvoice.extract.session import find_session, list_sessions

    root = config.extractions_dir
    if args.list:
        sessions = list_sessions(root)
        if not sessions:
            print("No extraction sessions.")
            return
        print(f"\n{'Session':<45} {'Speakers':<10} {'Pending':<9} {'Audio'}")
        print("-" * 75)
        for s in sessions:
            print(f"{s.id:<45} {len(s.candidates):<10} {len(s.pending):<9} "
                  f"{fmt_duration(s.captured_seconds)}")
        print()
        return

    session = find_session(root, args.session)
    if session is None:
        if args.session:
            print(f"Session not found (or ambiguous): {args.session}", file=sys.stderr)
            sys.exit(1)
        print("Nothing to review — no sessions with pending speakers.")
        return
    if not session.pending:
        print(f"All speakers in {session.id} have already been reviewed.")
        return
    if not sys.stdin.isatty():
        print("review needs an interactive terminal.", file=sys.stderr)
        sys.exit(1)
    _run_review(session, config)


def _run_review(session, config: Config):
    from kickvoice.extract.review import Reviewer

    Reviewer(session, config=config).run()


def cmd_profiles(args, config: Config):
    """Manage speaker profiles."""
    from kickvoice.storage.profiles import ProfileStore

    store = ProfileStore(config=config)

    if args.action == "list":
        profiles = store.list_profiles()
        if not profiles:
            print("No enrolled speakers.")
            return
        print(f"\n{'Name':<20} {'Segments':<10} {'Enrolled':<20} {'Sources'}")
        print("-" * 75)
        for p in profiles:
            sources = ", ".join(p.source_urls[:2])
            if len(p.source_urls) > 2:
                sources += f" (+{len(p.source_urls) - 2} more)"
            print(f"{p.name:<20} {p.n_segments:<10} {p.enrolled_at[:19]:<20} {sources}")
        print()

    elif args.action == "remove":
        if not args.name:
            print("Error: --name required for remove", file=sys.stderr)
            sys.exit(1)
        if store.remove_profile(args.name):
            print(f"🗑  Removed profile: {args.name}")
        else:
            print(f"Profile not found: {args.name}", file=sys.stderr)
            sys.exit(1)

    elif args.action == "info":
        if not args.name:
            print("Error: --name required for info", file=sys.stderr)
            sys.exit(1)
        profile = store.get_profile(args.name)
        if not profile:
            print(f"Profile not found: {args.name}", file=sys.stderr)
            sys.exit(1)
        print(f"\nSpeaker: {profile.name}")
        print(f"Enrolled: {profile.enrolled_at}")
        print(f"Segments: {profile.n_segments}")
        print(f"Sources:")
        for url in profile.source_urls:
            print(f"  - {url}")
        if profile.speech_seconds:
            print(f"Speech: {profile.speech_seconds:.0f}s")
        if profile.notes:
            print(f"Notes: {profile.notes}")
        print()

    elif args.action == "export":
        from kickvoice.storage.packs import PackError, export_pack

        if not args.out:
            print("Error: --out required for export", file=sys.stderr)
            sys.exit(1)
        names = [n.strip() for n in args.name.split(",")] if args.name else None
        try:
            n = export_pack(store, args.out, names=names, config=config)
        except PackError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"📦 Exported {n} profile(s) to {args.out}")

    elif args.action == "import":
        from kickvoice.storage.packs import PackError, import_pack

        if not args.file:
            print("Error: --file required for import", file=sys.stderr)
            sys.exit(1)
        try:
            r = import_pack(store, args.file, on_conflict=args.on_conflict, config=config)
        except PackError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        for label, names in (("Added", r.added), ("Overwrote", r.overwritten),
                             ("Merged", r.merged), ("Skipped (already exist)", r.skipped)):
            if names:
                print(f"{label}: {', '.join(names)}")
        if r.skipped:
            print("  (use --on-conflict merge|overwrite to update existing profiles)")


def main():
    parser = argparse.ArgumentParser(
        prog="kickvoice",
        description="Speaker identification on Kick.com live streams",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (default: ~/.kickvoice/config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- enroll ---
    p_enroll = subparsers.add_parser("enroll", help="Enroll a speaker")
    p_enroll.add_argument("--name", required=True, help="Speaker name")
    p_enroll.add_argument("--file", type=str, help="Path to audio file")
    p_enroll.add_argument(
        "--youtube", type=str,
        help="YouTube URL(s), comma-separated",
    )
    p_enroll.add_argument("--notes", type=str, help="Optional notes")

    # --- monitor ---
    p_monitor = subparsers.add_parser("monitor", help="Monitor a live stream")
    p_monitor.add_argument("--stream", type=str, help="Kick.com stream URL")
    p_monitor.add_argument("--file", type=str, help="Local audio file (for testing)")
    p_monitor.add_argument(
        "--threshold", type=float, default=None,
        help="Match threshold override",
    )

    # --- extract-profiles ---
    p_extract = subparsers.add_parser(
        "extract-profiles",
        help="Discover the speakers on a stream and turn them into profiles",
        description=(
            "Listen to a stream (or file), group speech by voice, match each "
            "voice against known profiles, then ask you to approve creating or "
            "updating profiles."
        ),
    )
    src = p_extract.add_mutually_exclusive_group(required=True)
    src.add_argument("--stream", type=str, help="Stream URL (e.g. https://kick.com/channel)")
    src.add_argument("--file", type=str, help="Local audio/video file (VOD, recording)")
    p_extract.add_argument(
        "--duration", type=float, default=None,
        help="Seconds to listen (default: 300 for streams, whole file for --file; "
             "0 = until the stream ends / Ctrl+C)",
    )
    p_extract.add_argument(
        "--min-speech", type=float, default=None,
        help="Ignore voices with less total speech than this, in seconds (default: 20)",
    )
    p_extract.add_argument(
        "--cluster-threshold", type=float, default=None,
        help="Voice grouping cut-off as cosine distance (default: 0.70). "
             "Lower splits more, higher merges more",
    )
    p_extract.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity needed to match a known profile (default: 0.60)",
    )
    p_extract.add_argument(
        "--no-review", action="store_true",
        help="Only save candidates; approve later with `review`",
    )

    # --- review ---
    p_review = subparsers.add_parser(
        "review", help="Approve speakers found by extract-profiles",
    )
    p_review.add_argument(
        "--session", type=str,
        help="Session id/prefix or path (default: newest with pending speakers)",
    )
    p_review.add_argument("--list", action="store_true", help="List sessions")

    # --- profiles ---
    p_profiles = subparsers.add_parser("profiles", help="Manage speaker profiles")
    p_profiles.add_argument(
        "action", choices=["list", "remove", "info", "export", "import"],
        help="Action to perform",
    )
    p_profiles.add_argument(
        "--name", type=str,
        help="Speaker name (export: optional, comma-separated; default all)",
    )
    p_profiles.add_argument("--out", type=str, help="export: output pack file")
    p_profiles.add_argument("--file", type=str, help="import: pack file to load")
    p_profiles.add_argument(
        "--on-conflict", choices=["skip", "overwrite", "merge"], default="skip",
        help="import: what to do when a profile already exists (default: skip)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config
    config_path = Path(args.config) if args.config else None
    config = Config.load(config_path)

    # Apply overrides
    if args.command in ("monitor", "extract-profiles") and args.threshold:
        config.identify.match_threshold = args.threshold
    if args.command == "extract-profiles":
        if args.min_speech is not None:
            config.extract.min_speaker_speech_s = args.min_speech
        if args.cluster_threshold is not None:
            config.extract.cluster_distance_threshold = args.cluster_threshold

    # Dispatch
    if args.command == "enroll":
        cmd_enroll(args, config)
    elif args.command == "monitor":
        cmd_monitor(args, config)
    elif args.command == "extract-profiles":
        cmd_extract(args, config)
    elif args.command == "review":
        cmd_review(args, config)
    elif args.command == "profiles":
        cmd_profiles(args, config)


if __name__ == "__main__":
    main()
