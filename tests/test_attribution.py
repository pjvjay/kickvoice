"""Word → speaker alignment rules (pure functions, no models)."""
import numpy as np
import pytest

from kickvoice.transcripts.asr import Word
from kickvoice.transcripts.attribution import (
    CONFIDENT, PROBABLE, UNKNOWN, LabelledWord, ScoredWindow, Turn, assign_groups_to_profiles,
    classify,
    finalise_turn, group_turns, label_words, speech_cut_point, words_in_speech,
)

NAMES = ["Host", "Guest"]


def w(start, end, text=" x"):
    return Word(start, end, text, 0.9)


def win(start, end, host, guest):
    return ScoredWindow(start, end, np.array([host, guest]))


def test_words_outside_speech_are_dropped():
    words = [w(0.0, 0.4, " thanks"), w(1.0, 1.4, " real"), w(5.0, 5.3, " watching")]
    kept = words_in_speech(words, [(0.9, 2.0)])
    assert [x.text for x in kept] == [" real"]


def test_label_words_uses_overlapping_windows():
    windows = [win(0, 2, 0.9, 0.1), win(1, 3, 0.8, 0.2), win(3, 5, 0.1, 0.85)]
    labelled = label_words([w(0.2, 0.6), w(3.5, 4.0)], windows, NAMES, match_threshold=0.6)
    assert [l.speaker for l in labelled] == ["Host", "Guest"]
    assert labelled[0].score == pytest.approx(0.9)


def test_label_words_averages_conflicting_windows():
    windows = [win(0, 2, 0.9, 0.1), win(1, 3, 0.1, 0.7)]
    (lw,) = label_words([w(1.2, 1.6)], windows, NAMES, 0.6)
    # mean: host 0.5, guest 0.4 → neither passes 0.6
    assert lw.speaker is None and lw.score == pytest.approx(0.5)


def test_label_words_without_windows_or_profiles():
    assert label_words([w(0, 1)], [], NAMES, 0.6)[0].speaker is None
    assert label_words([w(0, 1)], [win(0, 2, 0.9, 0.1)], [], 0.6)[0].speaker is None


def lw(start, end, speaker, score=0.8):
    return LabelledWord(w(start, end, f" {speaker}"), speaker, score)


def test_group_turns_splits_on_speaker_and_pause():
    words = [lw(0, 0.5, "Host"), lw(0.5, 1.0, "Host"),
             lw(1.0, 1.5, "Guest"),
             lw(3.0, 3.5, "Guest"),              # 1.5s pause → new turn
             lw(3.6, 4.0, None)]
    turns = group_turns(words, max_pause=1.0)
    assert [(t.speaker, len(t.words)) for t in turns] == [
        ("Host", 2), ("Guest", 1), ("Guest", 1), (None, 1)]
    assert turns[0].text == "Host Host"
    assert turns[0].duration == pytest.approx(1.0)


@pytest.mark.parametrize("sim,duration,contested,expected", [
    (0.9, 5.0, False, CONFIDENT),
    (0.9, 1.0, False, PROBABLE),     # too short for confident
    (0.9, 5.0, True, PROBABLE),      # word/turn voices disagreed
    (0.65, 5.0, False, PROBABLE),    # passes match, not high threshold
    (0.5, 5.0, False, UNKNOWN),
])
def test_classify(sim, duration, contested, expected):
    assert classify(sim, duration, 0.6, 0.75, 2.0, contested) == expected


def turn_of(speaker, start=0.0, end=3.0, score=0.8):
    return Turn(words=[lw(start, end, speaker, score)], speaker=speaker)


def test_finalise_turn_confident_when_turn_audio_agrees():
    t = finalise_turn(turn_of("Host"), np.array([0.85, 0.1]), NAMES, 0.6, 0.75, 2.0)
    assert (t.final_speaker, t.attribution, t.contested) == ("Host", CONFIDENT, False)
    assert t.similarity == pytest.approx(0.85)


def test_finalise_turn_contested_is_capped():
    t = finalise_turn(turn_of("Host"), np.array([0.3, 0.9]), NAMES, 0.6, 0.75, 2.0)
    assert (t.final_speaker, t.attribution, t.contested) == ("Guest", PROBABLE, True)


def test_finalise_turn_unknown_word_label_can_be_recovered_by_turn_audio():
    t = finalise_turn(turn_of(None), np.array([0.9, 0.1]), NAMES, 0.6, 0.75, 2.0)
    assert (t.final_speaker, t.attribution) == ("Host", CONFIDENT)


def test_finalise_turn_below_threshold_is_unknown():
    t = finalise_turn(turn_of("Host"), np.array([0.5, 0.2]), NAMES, 0.6, 0.75, 2.0)
    assert (t.final_speaker, t.attribution) == (None, UNKNOWN)


def test_finalise_short_turn_never_confident():
    t = finalise_turn(turn_of("Host", 0, 0.6, score=0.95), None, NAMES, 0.6, 0.75, 2.0)
    assert (t.final_speaker, t.attribution) == ("Host", PROBABLE)
    t = finalise_turn(turn_of(None, 0, 0.6, score=0.3), None, NAMES, 0.6, 0.75, 2.0)
    assert t.attribution == UNKNOWN


def test_finalise_without_profiles_keeps_embedding():
    emb = np.ones(4)
    t = finalise_turn(turn_of(None), None, [], 0.6, 0.75, 2.0, turn_embedding=emb)
    assert t.attribution == UNKNOWN and t.embedding is emb


@pytest.mark.parametrize("speech,expected", [
    ([], None),                               # no speech
    ([(100, 500)], None),                     # ends in silence
    ([(100, 500), (700, 1000)], 700),         # last utterance runs to the edge
    ([(0, 1000)], None),                      # one utterance fills the block
    ([(100, 500), (700, 995)], 700),          # within tolerance of the edge
])
def test_speech_cut_point(speech, expected):
    assert speech_cut_point(speech, 1000, edge_tolerance=10) == expected


def test_unknown_runs_split_where_the_voice_changes():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    words = [LabelledWord(w(0, 0.5), None, 0, a), LabelledWord(w(0.5, 1.0), None, 0, a),
             LabelledWord(w(1.0, 1.5), None, 0, b), LabelledWord(w(1.5, 2.0), None, 0, b)]
    assert len(group_turns(words, max_pause=1.0)) == 1               # no voice check
    turns = group_turns(words, max_pause=1.0, change_distance=0.7)
    assert [len(t.words) for t in turns] == [2, 2]


def test_voice_change_check_tolerates_missing_embeddings_and_similar_voices():
    a = np.array([1.0, 0.0])
    close = np.array([0.9, 0.436])                                     # cos ≈ 0.9
    words = [LabelledWord(w(0, 0.5), None, 0, a), LabelledWord(w(0.5, 1.0), None, 0, None),
             LabelledWord(w(1.0, 1.5), None, 0, close)]
    assert len(group_turns(words, 1.0, change_distance=0.7)) == 1


def test_label_words_attaches_mean_voice_embedding():
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    windows = [ScoredWindow(0, 2, np.zeros(0), e1), ScoredWindow(1, 3, np.zeros(0), e2)]
    (lw,) = label_words([w(1.2, 1.6)], windows, [], 0.6)
    assert lw.speaker is None
    assert np.allclose(lw.embedding, np.array([1, 1]) / np.sqrt(2))


def test_each_profile_claims_at_most_one_voice_group():
    # group 0 = host (0.9), group 1 = sound-alike guest (0.8 vs host), group 2 = guest
    scores = np.array([[0.90, 0.10],
                       [0.80, 0.20],
                       [0.15, 0.70]])
    assert assign_groups_to_profiles(scores, match_threshold=0.6) == [0, None, 1]


def test_group_assignment_threshold_and_empty():
    assert assign_groups_to_profiles(np.array([[0.5, 0.55]]), 0.6) == [None]
    assert assign_groups_to_profiles(np.zeros((2, 0)), 0.6) == [None, None]
    assert assign_groups_to_profiles(np.zeros((0, 3)), 0.6) == []


def test_group_assignment_prefers_globally_best_pairs():
    # group 0 is slightly closer to A than B, but group 1 is much closer to A:
    # A goes to group 1, and group 0 falls back to B.
    scores = np.array([[0.80, 0.78],
                       [0.95, 0.10]])
    assert assign_groups_to_profiles(scores, 0.6) == [1, 0]
