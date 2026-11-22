"""
YouTube: their own videos (captions) and, with a voice profile, their words
in any video (debates, podcasts, streams on other channels).

Two modes:
  captions  fast. Uses YouTube captions of videos on *their own channel*.
            Guests can be talking too, so this is "their channel (speaker
            unverified)".
  voice     slow but precise. Downloads the audio and runs kickvoice's
            transcription pipeline; only lines confidently matched to their
            voice profile are kept. Also the only mode used for appearances
            on other channels, where captions can't tell who is speaking.

All YouTube access goes through yt-dlp (already a kickvoice dependency).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from kickvoice.audio.ffmpeg import require_tool
from kickvoice.research.chunking import chunk_timed
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import SELF_CHANNEL, SELF_VOICE, Account, Document

logger = logging.getLogger(__name__)

TimedLines = list[tuple[float, float, str]]
VoiceTranscriber = Callable[[str, Path], TimedLines]   # (video url, workdir) → their lines


class YtDlp:
    """Thin subprocess wrapper (swappable in tests)."""

    def run(self, args: list[str], timeout: float = 600) -> str:
        cmd = [require_tool("yt-dlp", "pip install yt-dlp"), "--no-warnings", *args]
        logger.debug("yt-dlp %s", " ".join(args))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"yt-dlp timed out: {' '.join(args[-1:])}") from e
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[-400:]}")
        return result.stdout


def channel_videos_url(handle: str) -> str:
    """Channel id (UC…), @handle, or URL → the channel's videos tab."""
    handle = handle.strip()
    if handle.startswith("http"):
        if "youtube.com" not in handle or "?" in handle:
            return handle                      # other sites (Twitch) / explicit URLs as given
        base = handle.rstrip("/")
        return base if base.endswith(("/videos", "/streams")) else f"{base}/videos"
    if handle.startswith("@"):
        return f"https://www.youtube.com/{handle}/videos"
    if re.fullmatch(r"UC[\w-]{22}", handle):
        return f"https://www.youtube.com/channel/{handle}/videos"
    return f"https://www.youtube.com/@{handle}/videos"


def list_videos(ytdlp: YtDlp, url: str, limit: int) -> list[dict]:
    """Newest-first videos of a channel/playlist, or `ytsearchN:` results."""
    data = json.loads(ytdlp.run(["--flat-playlist", "-J", "-I", f"1:{limit}", url]))
    videos = []
    for entry in data.get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        videos.append({
            "id": entry["id"],
            "title": entry.get("title") or "",
            "url": entry.get("url") if str(entry.get("url", "")).startswith("http")
            else f"https://www.youtube.com/watch?v={entry['id']}",
            "channel_id": entry.get("channel_id") or data.get("channel_id"),
            "channel": entry.get("channel") or data.get("channel"),
            "duration": entry.get("duration"),
        })
    return videos


def fetch_captions(ytdlp: YtDlp, video_url: str, workdir: Path) -> tuple[dict, str | None]:
    """(metadata, vtt text or None). Prefers uploaded English subs over auto ones."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = ytdlp.run([
        "--no-simulate", "-j", "--skip-download",
        "--write-subs", "--write-auto-subs", "--sub-langs", "en.*,en", "--sub-format", "vtt",
        "-o", str(workdir / "%(id)s.%(ext)s"), video_url])
    meta = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
    files = sorted(workdir.glob(f"{meta.get('id', '*')}*.vtt"), key=_caption_rank)
    return meta, (files[0].read_text(encoding="utf-8", errors="replace") if files else None)


def fetch_metadata(ytdlp: YtDlp, video_url: str) -> dict:
    out = ytdlp.run(["-j", "--skip-download", video_url])
    return json.loads(out.strip().splitlines()[-1]) if out.strip() else {}


def _caption_rank(path: Path) -> tuple[int, str]:
    """Prefer 'ID.en.vtt', then regional English ('en-US'), then raw 'en-orig'."""
    lang = path.name.rsplit(".", 2)[-2] if path.name.count(".") >= 2 else ""
    if lang == "en":
        return (0, lang)
    if lang.endswith("-orig"):
        return (2, lang)
    return (1, lang)


_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
_TAG = re.compile(r"<[^>]+>")


def _seconds(ts: str) -> float:
    m = _TS.fullmatch(ts.strip())
    if not m:
        raise ValueError(ts)
    h, mnt, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(s) + int(ms) / 1000


def parse_vtt(vtt: str) -> TimedLines:
    """
    WebVTT → [(start, end, text)], de-duplicated.

    Parsed line by line (not by blank-line blocks): YouTube puts
    whitespace-only lines *inside* cues, which would otherwise split them.
    Auto-captions also repeat each line in the next cue (rolling display);
    keeping only lines that differ from the last two emitted ones removes
    the repeats without losing genuine repetition further apart.
    """
    rows = vtt.replace("\r\n", "\n").split("\n")
    lines: TimedLines = []
    recent: list[str] = []
    cue: tuple[float, float] | None = None
    for i, row in enumerate(rows):
        if "-->" in row:
            try:
                start_s, end_part = row.split("-->")
                cue = (_seconds(start_s), _seconds(end_part.split()[0]))
            except (ValueError, IndexError):
                cue = None
            continue
        if cue is None:
            continue                              # header (WEBVTT, Kind:, Language:)
        next_row = rows[i + 1] if i + 1 < len(rows) else ""
        prev_row = rows[i - 1] if i > 0 else ""
        if "-->" in next_row and not prev_row.strip():
            continue                              # a cue identifier line, not text
        text = re.sub(r"\s+", " ", _TAG.sub("", row)).strip()
        text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
        if not text or text in recent:
            continue
        lines.append((cue[0], cue[1], text))
        recent = (recent + [text])[-2:]
    return lines


def _published(meta: dict) -> str | None:
    d = meta.get("upload_date")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if d and len(d) == 8 else None


def collect_channel(
    ctx: SourceContext,
    account: Account,
    ytdlp: YtDlp | None = None,
    voice: VoiceTranscriber | None = None,
    limit: int | None = None,
) -> CollectStats:
    """Their own channel's videos — captions, or voice-verified lines."""
    ytdlp = ytdlp or YtDlp()
    stats = CollectStats()
    limit = limit or ctx.config.research.max_videos
    try:
        videos = list_videos(ytdlp, channel_videos_url(account.handle), limit)
    except RuntimeError as e:
        stats.errors.append(f"youtube {account.handle}: {e}")
        return stats
    ctx.progress(f"youtube: {len(videos)} video(s) on {account.handle}")
    for i, video in enumerate(videos, 1):
        ctx.progress(f"  [{i}/{len(videos)}] {video['title'][:70]}")
        _collect_video(ctx, video, ytdlp, voice, stats, own_channel=True)
    return stats


def collect_appearances(
    ctx: SourceContext,
    query: str,
    voice: VoiceTranscriber,
    exclude_channels: set[str],
    ytdlp: YtDlp | None = None,
    limit: int = 10,
) -> CollectStats:
    """
    Their words in other people's videos (debates, podcasts, interviews),
    found by search. Voice mode only: captions can't tell who is speaking.
    """
    ytdlp = ytdlp or YtDlp()
    stats = CollectStats()
    try:
        videos = list_videos(ytdlp, f"ytsearch{limit}:{query}", limit)
    except RuntimeError as e:
        stats.errors.append(f"youtube search: {e}")
        return stats
    videos = [v for v in videos if v.get("channel_id") not in exclude_channels]
    ctx.progress(f"youtube search '{query}': {len(videos)} appearance candidate(s)")
    for i, video in enumerate(videos, 1):
        ctx.progress(f"  [{i}/{len(videos)}] {video['title'][:70]}")
        _collect_video(ctx, video, ytdlp, voice, stats, own_channel=False)
    return stats


def _collect_video(ctx, video, ytdlp, voice, stats, own_channel: bool):
    with tempfile.TemporaryDirectory(prefix="kickvoice_yt_") as tmp:
        try:
            if voice is not None:
                meta = fetch_metadata(ytdlp, video["url"])
                lines = voice(video["url"], Path(tmp))
                authorship, mode = SELF_VOICE, "voice"
                if not lines:
                    ctx.progress("    (no lines matched their voice)")
                    return
            else:
                meta, vtt = fetch_captions(ytdlp, video["url"], Path(tmp))
                if not vtt:
                    stats.errors.append(f"no captions: {video['url']}")
                    return
                lines, authorship, mode = parse_vtt(vtt), SELF_CHANNEL, "captions"
        except RuntimeError as e:
            stats.errors.append(f"{video['url']}: {e}")
            return
    doc = Document(
        person_id=ctx.person.id, source="youtube", kind="video",
        external_id=f"{video['id']}:{mode}",
        text="\n".join(t for _, _, t in lines), authorship=authorship,
        url=f"https://www.youtube.com/watch?v={video['id']}",
        title=meta.get("title") or video.get("title", ""),
        published_at=_published(meta),
        meta={"channel": meta.get("channel") or video.get("channel"),
              "channel_id": meta.get("channel_id") or video.get("channel_id"),
              "mode": mode, "own_channel": own_channel,
              "duration": meta.get("duration") or video.get("duration")},
    )
    ctx.save(doc, chunk_timed(lines), stats)


def voice_transcriber(config, voice_profile: str) -> VoiceTranscriber:
    """
    Download a video's audio and keep only the lines confidently attributed
    to `voice_profile` by the transcription pipeline.
    """
    def transcribe(video_url: str, workdir: Path) -> TimedLines:
        from kickvoice.audio.download import download_audio
        from kickvoice.pipelines.transcribe import TranscriptionPipeline
        from kickvoice.transcripts.attribution import CONFIDENT

        audio = download_audio(video_url, workdir)
        pipeline = TranscriptionPipeline(file_path=str(audio), config=config,
                                         on_utterance=None, out=lambda _: None)
        session = pipeline.run()
        utterances = pipeline.transcripts.utterances(session.id, speaker=voice_profile,
                                                     min_attribution=CONFIDENT)
        # The lines now live in the research record (with the video's metadata);
        # drop the working transcript so the "transcripts" source doesn't
        # import the same speech a second time as a stream.
        pipeline.transcripts.delete_session(session.id)
        return [(u.start_s, u.end_s, u.text) for u in utterances]

    return transcribe
