"""Group word-level timings into sentence-level spans.

Boundaries are read off the punctuation on the words -- by then re-decided by
src/punctuate.py, not whisper's own. Punctuation is the only thing that ends a
caption, plus a length backstop. See to_sentences."""
from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from .backends import Segment, Word

SENTENCE_DELIMITERS = (".", "?", "!", "。", "？", "！")

# Latin-script languages only.
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
    # Lexical confidence: how sure the decoder was of the tokens it chose.
    # Aggregated from the words this sentence was built from, so a consumer of
    # the sentence track can weigh a caption without joining back to the word
    # track. None when whisper returned no word timings.
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
    # A lone Latin letter is an initial ("J. R. R. Tolkien").
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

    Split at the widest internal pause. A pause is weak evidence of a boundary,
    but for a run that has to be cut somewhere it is the best evidence available.
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

    Punctuation is the ONLY thing that ends a caption -- the same primary rule as
    model-asr's _merge_to_sentences -- so a punctuated sentence is kept whole
    however far apart its words are. A pause never splits one: speakers pause
    mid-sentence, and with VAD off the timestamps either side are true.

    Falls back to whisper's own segmentation when word timestamps are absent."""
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
