#!/usr/bin/env python3
"""Unit tests for the aligner, metrics, and normalizer. Run: python test_scorer.py"""

from normalize import NormConfig, normalize_ref, normalize_hyp, normalize_lines
from score import align, wer_from_counts, mer_from_counts, char_distance, score_pair


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_align_basic():
    # ref: a b c d ; hyp: a x c d e  -> 1 sub (b->x), 1 ins (e), 0 del, 3 correct
    _, c = align(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
    assert c == {"S": 1, "D": 0, "I": 1, "C": 3}, c
    assert approx(wer_from_counts(c), 2 / 4)          # (1+0+1)/4
    assert approx(mer_from_counts(c), 2 / 5)          # (1+0+1)/(1+0+1+3)


def test_align_deletion():
    # ref: a b c ; hyp: a c  -> 1 deletion (b)
    _, c = align(["a", "b", "c"], ["a", "c"])
    assert c == {"S": 0, "D": 1, "I": 0, "C": 2}, c
    assert approx(wer_from_counts(c), 1 / 3)


def test_perfect_and_empty():
    _, c = align(["a", "b"], ["a", "b"])
    assert c == {"S": 0, "D": 0, "I": 0, "C": 2}
    assert approx(wer_from_counts(c), 0.0)
    _, c2 = align(["a", "b"], [])
    assert c2 == {"S": 0, "D": 2, "I": 0, "C": 0}
    assert approx(wer_from_counts(c2), 1.0)


def test_wer_can_exceed_one_but_mer_cannot():
    # ref: a ; hyp: x y z  -> 1 sub + 2 ins => S=1,I=2,C=0
    # WER = (1+0+2)/1 = 3.0 (exceeds 1); MER = 3/(1+0+2+0) = 1.0 (capped at 1)
    _, c = align(["a"], ["x", "y", "z"])
    assert c == {"S": 1, "D": 0, "I": 2, "C": 0}, c
    assert approx(wer_from_counts(c), 3.0)
    assert approx(mer_from_counts(c), 1.0)


def test_cer():
    assert char_distance("kitten", "sitting") == 3
    assert char_distance("abc", "abc") == 0


def test_normalizer_keeps_fillers_by_default():
    toks = normalize_ref("Um, I don't-- I don't know.")
    assert toks == ["um", "i", "don't", "i", "don't", "know"], toks
    dropped = normalize_ref("Um, I don't know.", NormConfig(drop_fillers=True))
    assert "um" not in dropped


def test_normalizer_strips_annotations():
    toks = normalize_ref("[music] He said hello (phone rings)")
    assert toks == ["he", "said", "hello"], toks


def test_normalizer_digits_hyp_side():
    # hypothesis with digits should spell out; ref keeps its words
    assert normalize_hyp("$9800") == ["nine", "thousand", "eight", "hundred"]
    assert normalize_hyp("15%") == ["fifteen", "percent"]
    # ref side leaves digits alone by default (protocol: write words in ref)
    assert normalize_ref("9800") == ["9800"]


def test_normalizer_hyphens_split():
    assert normalize_ref("gluten-free whole-grain") == ["gluten", "free", "whole", "grain"]


def test_score_pair_end_to_end():
    ref = "I don't know.\nMaybe tomorrow."          # 4 ref words
    hyp = "I dunno maybe today"                       # sub don't->dunno? actually 'i' 'dunno' 'maybe' 'today'
    stats, buckets = score_pair(ref, hyp, NormConfig())
    # ref tokens: i don't know maybe tomorrow (5). hyp: i dunno maybe today (4)
    # align: i=C, don't->dunno=S, know->del=D, maybe=C, tomorrow->today=S
    assert stats["n_ref"] == 5, stats
    assert stats["S"] == 2 and stats["D"] == 1 and stats["I"] == 0, stats
    assert len(buckets) == 2  # two utterance lines


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
