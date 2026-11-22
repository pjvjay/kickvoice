import numpy as np
import pytest

from kickvoice.extract.cluster import (
    cluster_speakers, covered_seconds, make_windows, representative_windows,
)

SR = 16000


def test_make_windows_short_segment_is_single_window():
    assert make_windows(100, 100 + 2 * SR, 3 * SR, SR, SR) == [(100, 100 + 2 * SR)]


def test_make_windows_drops_too_short():
    assert make_windows(0, SR // 2, 3 * SR, SR, SR) == []


def test_make_windows_covers_tail():
    spans = make_windows(0, int(7.2 * SR), 3 * SR, int(1.5 * SR), SR)
    assert spans[0] == (0, 3 * SR)
    assert spans[-1] == (int(7.2 * SR) - 3 * SR, int(7.2 * SR))
    assert all(e - s == 3 * SR for s, e in spans)


def test_covered_seconds_counts_overlap_once():
    assert covered_seconds([(0, 3 * SR), (SR, 4 * SR), (10 * SR, 11 * SR)], SR) == 5.0
    assert covered_seconds([], SR) == 0.0


def _blobs(sizes, dim=64, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(sizes), dim))
    x, y = [], []
    for k, n in enumerate(sizes):
        x.append(centers[k] + rng.normal(scale=spread * np.linalg.norm(centers[k]) / np.sqrt(dim),
                                         size=(n, dim)))
        y += [k] * n
    return np.vstack(x), np.array(y)


def _spans(n, step=SR * 3 // 2, win=3 * SR):
    return [(i * step, i * step + win) for i in range(n)]


def test_recovers_speakers_and_orders_by_speech():
    x, y = _blobs([40, 25, 12])
    clusters = cluster_speakers(x, _spans(len(x)), SR, distance_threshold=0.5, min_speech_s=0)
    assert [c.n_windows for c in clusters] == [40, 25, 12]
    for c in clusters:
        assert len(set(y[c.indices])) == 1           # pure
        assert np.isclose(np.linalg.norm(c.centroid), 1.0)
        assert c.cohesion > 0.9


def test_small_clusters_dropped():
    x, _ = _blobs([40, 3])
    clusters = cluster_speakers(x, _spans(len(x)), SR, distance_threshold=0.5, min_speech_s=10)
    assert len(clusters) == 1 and clusters[0].n_windows == 40


def test_subsampling_assigns_remaining_windows():
    x, y = _blobs([300, 200])
    clusters = cluster_speakers(x, _spans(len(x)), SR, distance_threshold=0.5,
                                min_speech_s=0, max_windows=60)
    assert sorted(c.n_windows for c in clusters) == [200, 300]
    for c in clusters:
        assert len(set(y[c.indices])) == 1


def test_single_window_and_empty():
    assert cluster_speakers(np.zeros((0, 8)), [], SR, 0.5, 0) == []
    one = cluster_speakers(np.ones((1, 8)), [(0, 3 * SR)], SR, 0.5, 0)
    assert len(one) == 1 and one[0].speech_seconds == 3.0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        cluster_speakers(np.ones((2, 8)), [(0, 1)], SR, 0.5, 0)


def test_representative_windows_non_overlapping_time_ordered():
    x, _ = _blobs([30])
    spans = _spans(30)
    c = cluster_speakers(x, spans, SR, 0.5, 0)[0]
    ids = representative_windows(c, x, spans, SR, target_s=10)
    chosen = [spans[i] for i in ids]
    assert chosen == sorted(chosen)
    assert all(a[1] <= b[0] for a, b in zip(chosen, chosen[1:]))
    assert sum(e - s for s, e in chosen) >= 9 * SR
