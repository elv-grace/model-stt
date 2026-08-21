"""Punctuation restoration.

The model itself is a 2.2 GB download, so nothing here loads it. Everything that
decides what the output text looks like -- how a whisper word is taken apart, how
its mark is combined with a prediction, how capitalisation follows -- is a pure
function over (word, label, confidence), and that is what these cover. The
integration tests drive WhisperSTT with a scripted punctuator to prove the
restored words reach both tracks and that a broken one cannot lose a file.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import segment, word
from src.backends import Transcription
from src.model import SENTENCE_TRACK, WORD_TRACK
from src.punctuate import (
    PunctuationConfig,
    affixes,
    apply_labels,
    merge_mark,
    recase,
)

FILE = "test-files/1.m4a"


def tracks(tags, track):
    return [t for t in tags if t.track == track]


# ---------------------------------------------------------------- affixes


@pytest.mark.parametrize("raw,expected", [
    (" pursuit", (" ", "pursuit", "")),
    (" pursuit.", (" ", "pursuit", ".")),
    (" world!?", (" ", "world", "!?")),
    ("Hello", ("", "Hello", "")),
    (" 好。", (" ", "好", "。")),
    (" ...", (" ", "...", "")),      # nothing but punctuation: kept as the core
    (" ", (" ", "", "")),
])
def test_affixes_splits_lead_core_and_mark(raw, expected):
    assert affixes(raw) == expected


def test_affixes_preserves_the_leading_space():
    """sentences.py rebuilds captions with "".join, so the space is load-bearing."""
    lead, core, trail = affixes(" pursuit.")
    assert lead + core + trail == " pursuit."


# ---------------------------------------------------------------- merge_mark


def test_agreement_on_a_sentence_end_keeps_whispers_mark():
    """The label set has no '!' -- overwriting would flatten every exclamation."""
    assert merge_mark("!", ".", 0.99, 0.9) == "!"
    assert merge_mark("。", ".", 0.99, 0.9) == "。"
    assert merge_mark("?", ".", 0.99, 0.9) == "?"


def test_insertion_into_an_unpunctuated_run_is_unconditional():
    """The style-lapse and lyric-mode repair; cannot make an unpunctuated run worse."""
    assert merge_mark("", ".", 0.51, 0.9) == "."
    assert merge_mark("", "?", 0.30, 0.9) == "?"
    assert merge_mark("", ",", 0.30, 0.9) == ","


def test_demotion_is_off_by_default():
    """Measured to merge across real boundaries -- "Melissa." + "Richard." became
    one caption -- and the worst merges were the most confident ones."""
    assert merge_mark(".", "0", 0.9999, 0.9) == "."
    assert merge_mark(".", ",", 0.9999, 0.9) == "."


def test_demotion_when_enabled_still_respects_the_threshold():
    assert merge_mark(".", "0", 0.95, 0.9, allow_demotion=True) == ""
    assert merge_mark(".", ",", 0.95, 0.9, allow_demotion=True) == ","
    assert merge_mark(".", "0", 0.85, 0.9, allow_demotion=True) == "."


def test_whispers_own_non_terminal_mark_survives():
    """Its comma placement is good and it uses marks the label set collapses."""
    assert merge_mark(",", "0", 0.99, 0.9) == ","
    assert merge_mark(";", "0", 0.99, 0.9) == ";"


# ---------------------------------------------------------------- recase


def test_recase_capitalises_sentence_starts_only():
    words = [word(" what", 0.0, 0.1), word(" now.", 0.1, 0.2), word(" go", 0.2, 0.3)]
    assert [w.word for w in recase(words)] == [" What", " now.", " Go"]


def test_recase_never_lowercases():
    """Whisper's proper nouns carry information this model cannot reproduce."""
    words = [word(" Stafford", 0.0, 0.1), word(" zips", 0.1, 0.2), word(" it.", 0.2, 0.3)]
    assert [w.word for w in recase(words)] == [" Stafford", " zips", " it."]


def test_recase_does_not_capitalise_after_an_abbreviation():
    """"Mr." is not a sentence end, so what follows it is not a sentence start.
    Whisper's own casing is what supplies the capital on a name here."""
    words = [word(" Mr.", 0.0, 0.1), word(" anthony", 0.1, 0.2)]
    assert [w.word for w in recase(words)] == [" Mr.", " anthony"]


def test_recase_does_not_capitalise_after_an_initial():
    words = [word(" J.", 0.0, 0.1), word(" R.", 0.1, 0.2), word(" tolkien", 0.2, 0.3)]
    assert [w.word for w in recase(words)][-1] == " tolkien"


# ---------------------------------------------------------------- apply_labels


def test_apply_labels_repairs_a_lowercase_run():
    raw = ["what", "should", "we", "do", "now", "what", "did", "you", "do"]
    words = [word(f" {w}", i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(raw)]
    labels = ["0", "0", "0", "0", ".", "0", "0", "0", "?"]
    out = apply_labels(words, labels, [0.99] * len(words), 0.9)
    assert "".join(w.word for w in out) == " What should we do now. What did you do?"


def test_apply_labels_leaves_timings_untouched():
    """This changes the characters on a word, never which word sits where."""
    words = [word(" a", 0.0, 0.4), word(" b", 0.5, 0.9)]
    out = apply_labels(words, [".", "0"], [0.99, 0.99], 0.9)
    assert [(w.start, w.end) for w in out] == [(0.0, 0.4), (0.5, 0.9)]
    assert [w.probability for w in out] == [w.probability for w in words]


def test_apply_labels_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        apply_labels([word(" a", 0.0, 0.4)], [".", "."], [0.9, 0.9], 0.9)


def test_a_punctuation_only_token_is_passed_through():
    words = [word(" hi", 0.0, 0.1), word(" ...", 0.1, 0.2)]
    out = apply_labels(words, ["0", "."], [0.99, 0.99], 0.9)
    assert out[1].word == " ..."


# ---------------------------------------------------------------- integration


class FakePunctuator:
    """Stands in for the model: applies a scripted rewrite to the word stream."""

    def __init__(self, rewrite):
        self.rewrite = rewrite
        self.seen = None

    def restore(self, words):
        self.seen = list(words)
        return self.rewrite(list(words))


def test_restored_punctuation_reaches_both_tracks(make_model):
    """One unpunctuated segment becomes two captions once punctuation is restored."""
    raw = [word(" hello", 0.0, 0.4), word(" world", 0.4, 0.9),
           word(" how", 1.0, 1.2), word(" are", 1.2, 1.4), word(" you", 1.4, 1.8)]
    transcription = Transcription(language="en", segments=[segment(raw)])

    def rewrite(words):
        marks = [".", "", "", "", "?"]
        return recase([replace(w, word=w.word + m) for w, m in zip(words, marks)])

    model = make_model(transcription, punctuator=FakePunctuator(rewrite))
    tags = model.tag(FILE)

    # "World" gets its capital from the full stop restored on "hello"
    assert [t.tag for t in tracks(tags, WORD_TRACK)] == [
        "Hello.", "World", "how", "are", "you?"
    ]
    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["Hello.", "World how are you?"]


def test_the_restorer_sees_one_stream_across_segments(make_model):
    """Whisper's segment boundaries are the cause of the pause-split failure, so a
    repair that could only see inside one segment could never undo it."""
    transcription = Transcription(language="en", segments=[
        segment([word(" but", 0.0, 0.4), word(" it's the pursuit", 0.4, 0.9)]),
        segment([word(" that's meaningful.", 2.1, 2.9)]),
    ])
    punctuator = FakePunctuator(lambda words: words)
    make_model(transcription, punctuator=punctuator).tag(FILE)

    assert [w.word for w in punctuator.seen] == [
        " but", " it's the pursuit", " that's meaningful."
    ]


def test_a_failing_punctuator_degrades_to_whispers_own(make_model, hello_world):
    """Worse captions beat no captions."""

    class Exploding:
        def restore(self, words):
            raise RuntimeError("cuda is on fire")

    tags = make_model(hello_world, punctuator=Exploding()).tag(FILE)
    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["Hello world.", "How are you?"]


def test_a_punctuator_returning_the_wrong_count_is_ignored(make_model, hello_world):
    tags = make_model(hello_world, punctuator=FakePunctuator(lambda w: w[:2])).tag(FILE)
    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["Hello world.", "How are you?"]


def test_disabled_config_builds_no_punctuator(make_model, hello_world):
    model = make_model(hello_world, punctuation=PunctuationConfig(enabled=False))
    assert model.punctuator is None
