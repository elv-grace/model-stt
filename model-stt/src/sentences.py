"""Group word-level timings into sentence-level spans.

Whisper already emits punctuated, cased text, so unlike model-asr there is no
punctuation-restoration model in the loop: sentence boundaries are read straight
off the punctuation whisper produced. The grouping rule matches
model-asr's ASRProducer._merge_to_sentences so the two systems' sentence tracks
are structurally comparable.
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


def to_sentences(segments: List[Segment], max_gap_ms: float) -> List[Sentence]:
    """Merge words into sentences, splitting on punctuation or a long silence.

    Falls back to whisper's own segmentation when word timestamps are absent.
    """
    words = words_of(segments)
    if not words:
        return [
            Sentence(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]

    max_gap_s = max_gap_ms / 1000.0
    sentences: List[Sentence] = []
    current: List[Word] = []

    def flush() -> None:
        if not current:
            return
        text = "".join(w.word for w in current).strip()
        if text:
            probabilities = [w.probability for w in current]
            sentences.append(Sentence(
                start=current[0].start,
                end=current[-1].end,
                text=text,
                min_word_probability=round(min(probabilities), 4),
                mean_word_probability=round(sum(probabilities) / len(probabilities), 4),
            ))
        current.clear()

    for word in words:
        # a long silence ends the sentence even without punctuation, so a dropped
        # full stop cannot glue together two distant utterances
        if current and word.start - current[-1].end > max_gap_s:
            flush()
        current.append(word)
        if terminates_sentence(word.word):
            flush()

    flush()
    return sentences
