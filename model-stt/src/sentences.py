"""Group word-level timings into sentence-level spans.

Whisper already emits punctuated, cased text, so unlike model-asr there is no
punctuation-restoration model in the loop: sentence boundaries are read straight
off the punctuation whisper produced. Punctuation is the only thing that ends a
caption, which is the same primary rule as model-asr's
ASRProducer._merge_to_sentences, so the two systems' sentence tracks stay
structurally comparable.

One deviation from model-asr, which has no equivalent: a length backstop. See
to_sentences. An earlier version had a second deviation -- any silence longer
than a threshold also ended a caption -- which is now gone, so this is closer to
model-asr than it was, not further.
"""
from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from .backends import Segment, Word

SENTENCE_DELIMITERS = (".", "?", "!", "。", "？", "！")

# A trailing '.' on one of these is an abbreviation, not a sentence end. Without
# this, "Mr. Anthony Eden's speech" is emitted as the two tags "Mr." and
# "Anthony Eden's speech." Deliberately conservative: entries like "no" are
# excluded because "No." is far more often a real one-word sentence than
# "number", and a missed split is cheaper than a wrongly-merged pair.
# Latin-script languages only. CJK writes abbreviations differently and ends
# sentences with 。, which is never ambiguous, so nothing here applies to it.
# Single-letter titles (French "M.", Spanish "D.") are covered by the initial
# rule below rather than listed here.
ABBREVIATIONS = frozenset({
    # en
    "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "hon", "sr", "jr", "st", "mt",
    "gen", "col", "capt", "lt", "sgt", "maj", "cmdr",
    "vs", "etc", "approx", "dept",
    # fr
    "mme", "mlle", "ste",
    # es
    "sra", "srta", "dna",
    # de
    "hr", "nr", "abb",
    # it
    "sig", "dott",
})


@dataclass(frozen=True)
class Sentence:
    start: float  # seconds
    end: float
    text: str
    # Aggregated from the words this sentence was built from, so a consumer of
    # the sentence track can weigh a caption without joining back to the word
    # track. None when whisper returned no word timings.
    #
    # These are *lexical* confidence: how sure the decoder was of the tokens it
    # chose. They are not a hallucination score, and were measured not to work
    # as one. What a low minimum does mark reliably is a word the model guessed at, 
    # e.g., a score=0.50 between neighbours at 0.999, and no two decodes of the audio
    # produced the same word.
    min_word_probability: Optional[float] = None
    mean_word_probability: Optional[float] = None


def terminates_sentence(raw: str) -> bool:
    """Whether this token ends a sentence, allowing for abbreviations."""
    text = raw.strip()
    if not text.endswith(SENTENCE_DELIMITERS):
        return False
    if not text.endswith("."):
        # '?' and '!' are never abbreviation markers
        return True

    stem = text[:-1].strip(string.punctuation + string.whitespace)
    if not stem:
        return True
    # A lone *Latin* letter is an initial ("J. R. R. Tolkien"). The script check
    # matters: str.isalpha() is Unicode-aware, so without it a single Han or
    # Hangul character ("好.", "네.") would be read as an initial and its sentence
    # break suppressed.
    if len(stem) == 1 and _is_latin_letter(stem):
        return False
    return stem.lower() not in ABBREVIATIONS


def _is_latin_letter(char: str) -> bool:
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:  # unnamed codepoint
        return False


def words_of(segments: List[Segment]) -> List[Word]:
    return [w for s in segments for w in s.words]


def _make(words: List[Word]) -> Optional[Sentence]:
    text = "".join(w.word for w in words).strip()
    if not text:
        return None
    probabilities = [w.probability for w in words]
    return Sentence(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        min_word_probability=round(min(probabilities), 4),
        mean_word_probability=round(sum(probabilities) / len(probabilities), 4),
    )


def _bounded(words: List[Word], max_words: int) -> List[List[Word]]:
    """Break an over-long run down until every piece is at most max_words long.

    Split at the widest internal pause. A pause is weak evidence of a boundary --
    which is exactly why it is no longer allowed to end a caption on its own --
    but for a run that has to be cut somewhere it is the best evidence available.

    Whisper's own segment boundaries were considered here and are worse. They are
    decoder windows, not utterances, and land mid-phrase: on NBAallstar one
    segment ends "...supporting more than a" and the next opens "hundred thousand
    residents." Cutting there would split a sentence between an article and its
    noun; cutting at a pause at least follows the audio.

    Cuts always fall between words -- a word is never divided.

    Iterative: a run long enough to need this can exceed a thousand words.
    """
    out: List[List[Word]] = []
    stack = [words]
    while stack:
        run = stack.pop()
        if len(run) < 2 or len(run) <= max_words:
            out.append(run)
            continue
        at = max(range(len(run) - 1), key=lambda i: run[i + 1].start - run[i].end)
        stack.append(run[at + 1:])
        stack.append(run[:at + 1])
    return out


def to_sentences(
    segments: List[Segment], max_gap_ms: float, max_words: int
) -> List[Sentence]:
    """Group words into sentences on punctuation, with two narrow backstops.

    Once whisper has supplied punctuation it is the ONLY thing that ends a
    caption, which is the same primary rule as model-asr's _merge_to_sentences. A
    punctuated sentence is kept whole however far apart its words are:

        "Oh,"  32.24-33.04    ... 5.44s of real silence ...
        "my"   38.48-38.84
        "God." 38.84-39.24    -> ONE caption, 32.24-39.24

    An earlier version also ended a caption on any silence longer than
    `max_gap_ms`, which tore that sentence into "Oh," and "my God.", and split
    "Good luck, guys." the same way. A pause is not a sentence boundary --
    speakers pause mid-sentence -- and with VAD off the timestamps either side of
    one are true, so there is nothing to repair. Equivalently: rather than
    splitting on a pause and then merging back any fragment that does not end in
    terminal punctuation, never make the split.

    Two cases remain where punctuation cannot be trusted, and only these:

      max_words   whisper's degenerate decode, which emits a long lowercase run
                  with no terminal punctuation anywhere -- documented in README
                  for large-v3 ("collapses the sentence track from 16 segments to
                  3") and seen on turbo, where it produced one 1595-word caption.
                  Re-decoding that audio returned properly punctuated text, so it
                  is a decode failure, not a property of the content; restoring
                  punctuation with a separate model would treat the symptom at the
                  cost of a ~2 GB dependency and a torch runtime this image does
                  not have.

                  This is only the TRIGGER for cutting, never the cut point -- the
                  cuts land on pauses (see _bounded), so it remains pause-based
                  segmentation with the threshold found adaptively rather than
                  fixed. A fixed threshold cannot work here: that caption's widest
                  internal pause was 3.4s, so anything above it would never split
                  at all. Such a cut can land mid-sentence, which is unavoidable
                  once the text has no sentence boundaries left to respect.
      max_gap_ms  a trailing run that reaches the end of the input without ever
                  terminating. Unlike the case above this may be a genuine
                  unfinished thought rather than a failure, so the original
                  dropped-full-stop rule still applies to it -- and this is the
                  one pause cut made here rather than in _bounded.

    Falls back to whisper's own segmentation when word timestamps are absent.
    """
    words = words_of(segments)
    if not words:
        return [
            Sentence(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]

    groups: List[List[Word]] = [[]]
    for word in words:
        groups[-1].append(word)
        if terminates_sentence(word.word):
            groups.append([])
    groups = [g for g in groups if g]

    if groups and not terminates_sentence(groups[-1][-1].word):
        trailing, run = groups.pop(), []
        max_gap_s = max_gap_ms / 1000.0
        for word in trailing:
            if run and word.start - run[-1].end > max_gap_s:
                groups.append(run)
                run = []
            run.append(word)
        if run:
            groups.append(run)

    sentences = []
    for group in groups:
        for piece in _bounded(group, max_words):
            sentence = _make(piece)
            if sentence:
                sentences.append(sentence)
    return sentences
