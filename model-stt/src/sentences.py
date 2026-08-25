"""Group word-level timings into sentence-level spans.

Boundaries are read off the punctuation on the words -- by then re-decided by
src/punctuate.py, not whisper's own. Punctuation is the only thing that ends a
caption, plus a length backstop and, when diarization has labelled the words, a
change of speaker. See to_sentences."""
from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from .backends import Segment, Word

SENTENCE_DELIMITERS = (".", "?", "!", "。", "？", "！")

# Words a speaker change must have on both sides before it ends a caption.
# 2 rather than 1 because a one-word run is overwhelmingly a label flip inside
# one utterance rather than a turn; 2 rather than 3 because 3 also discards the
# real short exchanges diarization is there to catch. Measured in
# _split_on_speaker.
SPEAKER_SPLIT_MIN_WORDS = 2

# Silence a speaker change must also sit in. People do not swap mid-phrase with
# no pause at all, but a diarizer whose boundary is a word or two out of step
# with whisper's timings produces exactly that -- and it is the common case:
# across the seven-title run 87% of speaker splits fell between two contiguous
# words, and reading them found "You didn't" / "stay long." and "What are we" /
# "doing here?", one utterance in both cases. Requiring a real gap drops 90% of
# splits and keeps the ones that read as turns. Raise it towards a second to
# split only on unmistakable handovers; set it very high to stop splitting on
# speaker altogether and leave captions purely punctuation-shaped.
SPEAKER_SPLIT_MIN_GAP = 0.2  # seconds

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
    # Who was speaking, when diarization ran and placed someone here. One
    # caption never spans two known speakers, so this is a single value rather
    # than a set. None means diarization was off, or ran and found no speaker
    # over any word in this caption.
    speaker: Optional[str] = None


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
        # Whoever holds the most words, not whoever holds the first. A run is
        # nearly homogeneous by construction, but _split_on_speaker rejoins runs
        # too short to stand alone, and the label on those is the one least
        # worth believing -- reading the first word would let a rejoined
        # one-word flip name the whole caption.
        speaker=_dominant_speaker(words),
    )


def _dominant_speaker(words: List[Word]) -> Optional[str]:
    """The speaker holding the most words, earliest of them winning a tie."""
    counts: dict = {}
    for word in words:
        if word.speaker is not None:
            counts[word.speaker] = counts.get(word.speaker, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda speaker: counts[speaker])


def _split_on_speaker(
    words: List[Word],
    min_words: int = SPEAKER_SPLIT_MIN_WORDS,
    min_gap: float = SPEAKER_SPLIT_MIN_GAP,
) -> List[List[Word]]:
    """Break a run where the speaker changes, if the change looks like a turn.

    A word with no speaker is not evidence of a change. Diarization leaves gaps
    -- a word over music, a word in a pause the segmentation model called
    silence -- and splitting a caption at one would turn missing evidence into a
    visible boundary. Such a word joins the run it falls in and takes its label.
    Only two known, different speakers end a caption.

    Nor is a change on its own. A diarizer boundary a word or two out of step
    with whisper's timings looks exactly like a turn to a rule that only reads
    the label, and it is by far the common case. Two things separate the two,
    both required, and both measured over the seven-title run:

    - *Substance on each side.* Splits leaving a single word were 52% of every
      split diarization made, and reading them found "Thank" / "you." and "I'm"
      / "a human, not a rabbit." -- one utterance cut in two.
    - *A real pause.* 87% of splits fell between two contiguous words, which is
      not how people take turns. Requiring a gap drops 90% of splits, and what
      survives reads as handovers.

    Neither speaker_overlap nor speaker_coverage separates good splits from bad
    ones -- on this content most bad splits carry no overlap at all -- so these
    two are what is left. The cost is that a genuine interjection landing hard
    on the previous word rejoins it, which is where it sits with diarization
    off.

    This runs before _bounded, so it shortens runs and never lengthens them: a
    long single-speaker run still meets the same max_words backstop it does
    today, cut at its widest internal pause.
    """
    runs: List[List[Word]] = [[]]
    current: Optional[str] = None
    for word in words:
        if word.speaker is not None:
            if current is not None and word.speaker != current:
                runs.append([])
            current = word.speaker
        runs[-1].append(word)
    runs = [run for run in runs if run]

    # Decide each boundary rather than merge blindly. A boundary survives only
    # if both sides have substance AND they still disagree about who is talking
    # once the caption so far is taken as a whole -- absorbing a flip can leave
    # the next run naming the same speaker the caption already has, and that is
    # not a boundary at all.
    out: List[List[Word]] = []
    caption: List[Word] = []
    for run in runs:
        if not caption:
            caption = list(run)
            continue
        if (
            len(caption) >= min_words
            and len(run) >= min_words
            and run[0].start - caption[-1].end >= min_gap
            and _dominant_speaker(caption) != _dominant_speaker(run)
        ):
            out.append(caption)
            caption = list(run)
        else:
            caption.extend(run)
    if caption:
        out.append(caption)
    return out


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
    """Group words into sentences on punctuation, with three narrow backstops.

    Punctuation is the primary rule -- the same one as model-asr's
    _merge_to_sentences -- so a punctuated sentence is kept whole however far
    apart its words are. A pause never splits one: speakers pause mid-sentence,
    and with VAD off the timestamps either side are true.

    A change of speaker does end a caption, when diarization has labelled the
    words. Two people are two captions even mid-sentence, because one of them
    interrupting the other is exactly where whisper's punctuation is least
    reliable and where a merged caption is most obviously wrong. With
    diarization off, every word has speaker None and this rule never fires.

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
        # speaker first, then length: the backstop should count the words of the
        # caption that will actually be emitted
        for run in _split_on_speaker(group):
            for piece in _bounded(run, max_words):
                sentence = _make(piece)
                if sentence:
                    sentences.append(sentence)
    return sentences
