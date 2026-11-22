"""
Who said each word: align ASR words with speaker-identity scores.

Pure numpy so every rule here is unit-testable without models.

    1. Speaker windows (short, overlapping) carry a similarity score against
       every enrolled profile.
    2. Each word takes the mean scores of the windows overlapping it; the
       best profile above the match threshold wins, otherwise "unknown".
    3. Consecutive words with the same label (and no long pause) form a turn;
       runs are also split where the voice changes.
    4. Each turn is re-scored from its own, longer audio and given an
       attribution level: confident / probable / unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kickvoice.transcripts.asr import Word

CONFIDENT = "confident"  # strong match on enough audio — safe to analyse
PROBABLE = "probable"  # passes the match threshold, but short or contested
UNKNOWN = "unknown"  # no enrolled voice matched

ATTRIBUTION_LEVELS = (CONFIDENT, PROBABLE, UNKNOWN)


@dataclass
class ScoredWindow:
    start: float  # seconds, relative to the block
    end: float
    scores: np.ndarray  # similarity to each profile (len = n profiles)
    embedding: np.ndarray | None = None  # L2-normalised voice embedding


@dataclass
class LabelledWord:
    word: Word
    speaker: str | None
    score: float  # mean window similarity to `speaker` (or best)
    embedding: np.ndarray | None = None  # voice at this word (for change detection)


@dataclass
class Turn:
    """A run of words attributed to one voice."""

    words: list[LabelledWord]
    speaker: str | None  # label from the word-level pass
    # Filled in by `finalise_turn`:
    final_speaker: str | None = None
    similarity: float = 0.0
    attribution: str = UNKNOWN
    contested: bool = False  # word-level and turn-level voices disagreed
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def start(self) -> float:
        return self.words[0].word.start

    @property
    def end(self) -> float:
        return self.words[-1].word.end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return "".join(w.word.text for w in self.words).strip()

    @property
    def asr_confidence(self) -> float:
        return float(np.mean([w.word.probability for w in self.words]))


def words_in_speech(
    words: list[Word],
    speech: list[tuple[float, float]],
    tolerance: float = 0.2,
) -> list[Word]:
    """
    Drop words whose midpoint isn't inside VAD speech — Whisper invents text
    over music and silence ("Thanks for watching!"), which streams have
    plenty of.
    """
    return [w for w in words if any(s - tolerance <= w.mid <= e + tolerance for s, e in speech)]


def label_words(
    words: list[Word],
    windows: list[ScoredWindow],
    names: list[str],
    match_threshold: float,
) -> list[LabelledWord]:
    """Give each word the best-scoring profile of the windows overlapping it."""
    labelled = []
    for w in words:
        covering = [win for win in windows if win.start < w.end and w.start < win.end]
        embedding = _mean_unit([win.embedding for win in covering if win.embedding is not None])
        if not covering or not names:
            labelled.append(LabelledWord(w, None, 0.0, embedding))
            continue
        mean = np.mean([win.scores for win in covering], axis=0)
        best = int(np.argmax(mean))
        score = float(mean[best])
        labelled.append(LabelledWord(w, names[best] if score >= match_threshold else None, score, embedding))
    return labelled


def group_turns(
    words: list[LabelledWord],
    max_pause: float,
    change_distance: float | None = None,
) -> list[Turn]:
    """
    Consecutive same-speaker words → turns; long pauses also split.

    With `change_distance` set, a run is also split when a word's voice is
    further than that (cosine distance) from the turn so far. That separates
    two unrecognised people, and also a sound-alike whose words scored as the
    same profile. Over-splitting one person is harmless: the whole-session
    check regroups the pieces by voice.
    """
    turns: list[Turn] = []
    centroid: np.ndarray | None = None
    for lw in words:
        current = turns[-1] if turns else None
        same = (
            current is not None and current.speaker == lw.speaker and lw.word.start - current.end <= max_pause
        )
        if (
            same
            and change_distance is not None
            and centroid is not None
            and lw.embedding is not None
            and 1.0 - float(centroid @ lw.embedding) > change_distance
        ):
            same = False
        if same:
            current.words.append(lw)
            centroid = _mean_unit([x.embedding for x in current.words if x.embedding is not None])
        else:
            turns.append(Turn(words=[lw], speaker=lw.speaker))
            centroid = lw.embedding
    return turns


def _mean_unit(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else None


def classify(
    similarity: float,
    duration: float,
    match_threshold: float,
    high_threshold: float,
    min_confident_s: float,
    contested: bool = False,
) -> str:
    """The single rule for how far a voice match can be trusted."""
    if similarity < match_threshold:
        return UNKNOWN
    if similarity >= high_threshold and duration >= min_confident_s and not contested:
        return CONFIDENT
    return PROBABLE


def finalise_turn(
    turn: Turn,
    turn_scores: np.ndarray | None,
    names: list[str],
    match_threshold: float,
    high_threshold: float,
    min_confident_s: float,
    turn_embedding: np.ndarray | None = None,
) -> Turn:
    """
    Decide the turn's speaker and how much to trust it.

    `turn_scores` are the similarities of the *whole turn's* audio to each
    profile (None if the turn was too short to embed). A longer sample is more
    reliable than any single window, so it has the final say — but if it
    disagrees with the word-level label, the turn is capped at "probable".
    """
    turn.embedding = turn_embedding
    if not names:
        turn.final_speaker, turn.similarity, turn.attribution = None, 0.0, UNKNOWN
        return turn

    if turn_scores is None:
        # Too short to re-score: fall back to the word-level evidence, which
        # never counts as confident.
        scores = [w.score for w in turn.words]
        turn.similarity = float(np.mean(scores)) if scores else 0.0
        turn.final_speaker = turn.speaker
        turn.attribution = PROBABLE if turn.speaker else UNKNOWN
        return turn

    best = int(np.argmax(turn_scores))
    turn.similarity = float(turn_scores[best])
    turn.contested = turn.speaker is not None and turn.speaker != names[best]
    turn.attribution = classify(
        turn.similarity, turn.duration, match_threshold, high_threshold, min_confident_s, turn.contested
    )
    turn.final_speaker = names[best] if turn.attribution != UNKNOWN else None
    return turn


def speech_cut_point(
    speech: list[tuple[int, int]],
    n_samples: int,
    edge_tolerance: int,
) -> int | None:
    """
    Where to end a block so the last (still running) utterance isn't cut.

    If speech reaches the end of the block, return the start of that speech
    segment — the caller processes audio up to there and carries the rest
    into the next block. Returns None when no cut is needed (or possible:
    speech spans the whole block).
    """
    if not speech:
        return None
    start, end = speech[-1]
    if end < n_samples - edge_tolerance:
        return None  # block ends in silence — fine
    if start <= 0:
        return None  # one continuous utterance; can't help
    return start


def assign_groups_to_profiles(
    group_scores: np.ndarray,
    match_threshold: float,
) -> list[int | None]:
    """
    Give each voice group at most one profile, and each profile at most one
    group, greedily by similarity.

    `group_scores[g, p]` is the similarity of voice group g to profile p.
    Returns the profile index for each group (None = unmatched).

    The one-group-per-profile rule is what stops an unenrolled guest who
    sounds like the host (same mic, same room) from being attributed to the
    host: the host's own, closer voice group claims the profile first. If
    one person is split into two groups, the weaker group stays unknown —
    precision over recall.
    """
    n_groups = group_scores.shape[0]
    assigned: list[int | None] = [None] * n_groups
    if group_scores.size == 0:
        return assigned
    taken_profiles: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-group_scores, axis=None), group_scores.shape))[0]
    for g, p in order:
        if group_scores[g, p] < match_threshold:
            break
        if assigned[g] is None and p not in taken_profiles:
            assigned[g] = int(p)
            taken_profiles.add(int(p))
    return assigned
