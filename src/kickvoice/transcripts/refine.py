"""
Session-level attribution: decide speakers using the whole session at once.

Single lines are short and noisy, and two people on the same mic can score
alike against one profile. Looking at the whole session fixes both:

    1. Group the session's lines by voice (same clustering as extract-profiles).
    2. Each group's averaged voice is matched against the profiles, and each
       profile can claim at most one group (closest wins).
    3. Every line inherits its group's speaker. To be "confident", the group
       AND the line itself must both match strongly (plus length/contested
       checks). A line whose group matched but whose own voice is a weak
       match is only "probable".

Why both: in testing, one failure was a sound-alike guest whose lines scored
exactly like the host's line by line (only the group rule caught it), and
another was two similar voices merged into one group whose lines scored very
differently on their own (only the line check caught it).

Runs automatically at the end of `transcribe`, and for `transcripts reattribute`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kickvoice.config import Config
from kickvoice.speakers.cluster import cluster_speakers
from kickvoice.speakers.profiles import normalise
from kickvoice.transcripts.attribution import (
    PROBABLE,
    UNKNOWN,
    assign_groups_to_profiles,
    classify,
)
from kickvoice.transcripts.store import TranscriptStore, Utterance


@dataclass
class RefineResult:
    changed: int = 0  # speaker or attribution level changed
    unchanged: int = 0
    voice_groups: int = 0
    matched_groups: int = 0


def refine_session(
    store: TranscriptStore,
    session_id: str,
    names: list[str],
    matrix: np.ndarray,
    config: Config,
) -> RefineResult:
    """Re-attribute every line of one session against `names`/`matrix`."""
    utterances = store.utterances(session_id)
    decisions = decide(utterances, names, matrix, config)
    result = RefineResult()
    updates = []
    for u, (speaker, similarity, level, group) in zip(utterances, decisions, strict=True):
        if (speaker, level) != (u.speaker, u.attribution):
            result.changed += 1
        else:
            result.unchanged += 1
        updates.append((u.id, speaker, similarity, level, group))
    store.update_attributions(updates)
    groups = {d[3] for d in decisions if d[3] is not None}
    result.voice_groups = len(groups)
    result.matched_groups = len({(d[0], d[3]) for d in decisions if d[0] is not None and d[3] is not None})
    return result


def decide(
    utterances: list[Utterance],
    names: list[str],
    matrix: np.ndarray,
    config: Config,
) -> list[tuple[str | None, float, str, int | None]]:
    """(speaker, similarity, attribution, voice_group) for each utterance."""
    ident, tcfg = config.identify, config.transcribe
    sr = config.audio.sample_rate
    embedded = [i for i, u in enumerate(utterances) if u.embedding is not None]

    # 1. group the lines that have a voice embedding
    group_of: dict[int, int] = {}
    group_speaker: dict[int, str | None] = {}
    group_sim: dict[int, float] = {}
    if embedded:
        vectors = np.stack([normalise(utterances[i].embedding) for i in embedded])
        spans = [(int(utterances[i].start_s * sr), int(utterances[i].end_s * sr)) for i in embedded]
        clusters = cluster_speakers(
            vectors,
            spans,
            sr,
            distance_threshold=config.extract.cluster_distance_threshold,
            min_speech_s=0.0,
            max_windows=config.extract.max_cluster_windows,
        )
        centroids = np.stack([c.centroid for c in clusters]) if clusters else np.zeros((0, 1))
        # 2. one profile per group, one group per profile
        scores = centroids @ matrix.T if names and clusters else np.zeros((len(clusters), 0))
        assignment = assign_groups_to_profiles(scores, ident.match_threshold)
        for g, cluster in enumerate(clusters):  # g = 0 is the most talkative voice
            for j in cluster.indices:
                group_of[embedded[j]] = g + 1
            p = assignment[g]
            group_speaker[g + 1] = names[p] if p is not None else None
            group_sim[g + 1] = (
                float(scores[g, p]) if p is not None else (float(scores[g].max()) if scores.shape[1] else 0.0)
            )

    # 3. each line inherits its group's decision
    claimed = {s for s in group_speaker.values() if s}
    decisions = []
    for i, u in enumerate(utterances):
        group = group_of.get(i)
        if group is not None:
            speaker = group_speaker[group]
            if speaker is None:
                decisions.append((None, group_sim[group], UNKNOWN, group))
                continue
            own = float(normalise(u.embedding) @ matrix[names.index(speaker)])
            evidence = min(group_sim[group], own)  # both must be strong
            level = classify(
                evidence,
                u.duration,
                ident.match_threshold,
                ident.high_confidence_threshold,
                tcfg.min_confident_turn_s,
                contested=u.contested,
            )
            # The group says it's them, so a weak line is "probable", not unknown.
            decisions.append((speaker, own, level if level != UNKNOWN else PROBABLE, group))
        else:
            # Too short to embed: keep its word-level label only if that
            # speaker was found in this session, and never above "probable".
            if u.speaker and u.speaker in claimed and u.attribution != UNKNOWN:
                decisions.append((u.speaker, u.similarity, PROBABLE, None))
            else:
                decisions.append((None, u.similarity, UNKNOWN, None))
    return decisions
