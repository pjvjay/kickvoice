"""
Unsupervised speaker discovery: group short-window embeddings by voice.

Pure numpy/scipy so it can be tested without any ML models.

Approach (a lightweight speaker-diarization back-end):
  1. Each VAD speech segment is cut into short overlapping windows
     (default 3s / 1.5s hop) so that one embedding rarely spans two people.
  2. Windows are clustered with average-linkage agglomerative clustering on
     cosine distance, cut at a fixed distance threshold. The number of
     speakers does not need to be known in advance.
  3. Clusters with too little total speech are dropped (crosstalk, music,
     one-off TTS donations, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeakerCluster:
    """A group of windows believed to be the same voice."""

    indices: np.ndarray  # indices into the window list
    centroid: np.ndarray  # L2-normalised mean embedding
    speech_seconds: float  # union of window spans (no double counting)
    cohesion: float  # mean cosine sim of members to centroid
    first_sample: int  # earliest window start (absolute sample)

    @property
    def n_windows(self) -> int:
        return int(len(self.indices))


def make_windows(
    start: int,
    end: int,
    window: int,
    hop: int,
    min_len: int,
) -> list[tuple[int, int]]:
    """
    Split the sample span [start, end) into embedding windows.

    Segments shorter than `window` become a single window. Longer segments get
    windows every `hop` samples, plus a final window flush with the end so the
    tail is covered. Spans shorter than `min_len` yield nothing.
    """
    length = end - start
    if length < min_len or window <= 0 or hop <= 0:
        return []
    if length <= window:
        return [(start, end)]
    spans = []
    s = start
    while s + window <= end:
        spans.append((s, s + window))
        s += hop
    if spans[-1][1] < end:
        spans.append((end - window, end))
    return spans


def covered_seconds(spans: list[tuple[int, int]], sample_rate: int) -> float:
    """Total time covered by (possibly overlapping) sample spans."""
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total / sample_rate


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _agglomerate(x: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Average-linkage cosine clustering → integer labels (0-based)."""
    if len(x) == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(x, method="average", metric="cosine")
    return fcluster(z, t=distance_threshold, criterion="distance") - 1


def cluster_speakers(
    embeddings: np.ndarray,
    spans: list[tuple[int, int]],
    sample_rate: int,
    distance_threshold: float,
    min_speech_s: float,
    max_windows: int = 4000,
) -> list[SpeakerCluster]:
    """
    Cluster window embeddings into speakers.

    Args:
        embeddings: (N, D) window embeddings (any scale; normalised here)
        spans: N (start_sample, end_sample) absolute window positions
        sample_rate: samples per second for the spans
        distance_threshold: cosine distance at which to cut the dendrogram
        min_speech_s: drop clusters covering less speech than this
        max_windows: cluster at most this many windows directly; the rest
            are assigned to the nearest resulting cluster

    Returns:
        Clusters sorted by speech time (most talkative first).
    """
    if len(embeddings) == 0:
        return []
    if len(embeddings) != len(spans):
        raise ValueError("embeddings and spans must have the same length")

    x = _normalise(np.asarray(embeddings, dtype=np.float64))
    n = len(x)

    if n > max_windows:
        # Evenly spaced subsample keeps coverage across the whole capture.
        core = np.unique(np.linspace(0, n - 1, max_windows).astype(int))
        core_labels = _agglomerate(x[core], distance_threshold)
        labels = np.full(n, -1, dtype=int)
        labels[core] = core_labels
        centroids = _normalise(
            np.stack([x[core[core_labels == k]].mean(axis=0) for k in range(core_labels.max() + 1)])
        )
        rest = np.setdiff1d(np.arange(n), core)
        sims = x[rest] @ centroids.T
        best = sims.argmax(axis=1)
        ok = (1.0 - sims[np.arange(len(rest)), best]) <= distance_threshold
        labels[rest[ok]] = best[ok]
        logger.info(
            f"Clustered {len(core)} of {n} windows directly; "
            f"assigned {int(ok.sum())} of the remaining {len(rest)}"
        )
    else:
        labels = _agglomerate(x, distance_threshold)

    clusters: list[SpeakerCluster] = []
    dropped = 0
    for k in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == k)
        member_spans = [spans[i] for i in idx]
        speech = covered_seconds(member_spans, sample_rate)
        if speech < min_speech_s:
            dropped += 1
            continue
        centroid = _normalise(x[idx].mean(axis=0))
        cohesion = float((x[idx] @ centroid).mean())
        clusters.append(
            SpeakerCluster(
                indices=idx,
                centroid=centroid,
                speech_seconds=speech,
                cohesion=cohesion,
                first_sample=min(s for s, _ in member_spans),
            )
        )

    clusters.sort(key=lambda c: c.speech_seconds, reverse=True)
    logger.debug(
        f"{len(clusters)} speaker cluster(s) kept, {dropped} small cluster(s) "
        f"dropped (< {min_speech_s:.0f}s speech)"
    )
    return clusters


def representative_windows(
    cluster: SpeakerCluster,
    embeddings: np.ndarray,
    spans: list[tuple[int, int]],
    sample_rate: int,
    target_s: float,
) -> list[int]:
    """
    Pick the windows most typical of a cluster (closest to its centroid),
    non-overlapping, until `target_s` seconds are collected. Returns window
    indices in time order, ready to be stitched into a review clip.
    """
    x = _normalise(np.asarray(embeddings[cluster.indices], dtype=np.float64))
    order = np.argsort(-(x @ cluster.centroid))
    chosen: list[int] = []
    total = 0
    target = int(target_s * sample_rate)
    for j in order:
        i = int(cluster.indices[j])
        s, e = spans[i]
        if any(s < spans[c][1] and spans[c][0] < e for c in chosen):
            continue
        chosen.append(i)
        total += e - s
        if total >= target:
            break
    return sorted(chosen, key=lambda i: spans[i][0])
