#!/usr/bin/env python3
"""
kickvoice CLI — Speaker identification on live streams.

Usage:
    kickvoice enroll --name "Speaker" --file audio.wav
    kickvoice enroll --name "Speaker" --youtube "https://youtube.com/..."
    kickvoice monitor --stream "https://kick.com/channel"
    kickvoice monitor --file recording.wav
    kickvoice profiles list
    kickvoice profiles remove --name "Speaker"
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
        if profile.notes:
            print(f"Notes: {profile.notes}")
        print()


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

    # --- profiles ---
    p_profiles = subparsers.add_parser("profiles", help="Manage speaker profiles")
    p_profiles.add_argument(
        "action", choices=["list", "remove", "info"],
        help="Action to perform",
    )
    p_profiles.add_argument("--name", type=str, help="Speaker name")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config
    config_path = Path(args.config) if args.config else None
    config = Config.load(config_path)

    # Apply overrides
    if args.command == "monitor" and args.threshold:
        config.identify.match_threshold = args.threshold

    # Dispatch
    if args.command == "enroll":
        cmd_enroll(args, config)
    elif args.command == "monitor":
        cmd_monitor(args, config)
    elif args.command == "profiles":
        cmd_profiles(args, config)


if __name__ == "__main__":
    main()
