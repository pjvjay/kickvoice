"""Split collected text into searchable, citable chunks."""
from __future__ import annotations

import re

from kickvoice.research.store import Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 1500) -> list[Chunk]:
    """
    Paragraph-aware chunks of at most ~max_chars. Paragraphs are kept whole
    when they fit; long ones are split at sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        sentence_buf = ""
        for sentence in _SENTENCE_END.split(para):
            if sentence_buf and len(sentence_buf) + len(sentence) + 1 > max_chars:
                pieces.append(sentence_buf)
                sentence_buf = ""
            while len(sentence) > max_chars:              # no punctuation at all
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            sentence_buf = f"{sentence_buf} {sentence}".strip()
        if sentence_buf:
            pieces.append(sentence_buf)

    chunks: list[Chunk] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > max_chars:
            chunks.append(Chunk(buf))
            buf = ""
        buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(Chunk(buf))
    return chunks


def chunk_timed(lines: list[tuple[float, float, str]], max_chars: int = 1200,
                max_gap_s: float = 20.0) -> list[Chunk]:
    """
    Group timed transcript lines into chunks that keep start/end times, so
    evidence can link to the exact moment in a video. A long silence also
    starts a new chunk.
    """
    chunks: list[Chunk] = []
    buf: list[tuple[float, float, str]] = []
    for start, end, text in lines:
        text = text.strip()
        if not text:
            continue
        size = sum(len(t) + 1 for _, _, t in buf)
        if buf and (size + len(text) > max_chars or start - buf[-1][1] > max_gap_s):
            chunks.append(Chunk(" ".join(t for _, _, t in buf), buf[0][0], buf[-1][1]))
            buf = []
        buf.append((start, end, text))
    if buf:
        chunks.append(Chunk(" ".join(t for _, _, t in buf), buf[0][0], buf[-1][1]))
    return chunks
