"""Restore punctuation over whisper's word stream.

to_sentences splits on punctuation, so a window with none becomes a single caption
holding many utterances.
Mirrored, Whisper decides punctuation per segment and it closes a segment at a pause, 
so a mid-sentence pause becomes a full stop: "but it's the pursuit." / "It's meaningful."

The repair is to run a token-classification model over the flat word
stream and let it re-decide, with no segment boundaries in view. This module is
the model-stt equivalent of model-asr's src/pretty.py, except instead of 
overwriting/replacing all the punctuation, it merges with Whisper's own (see merge_mark).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from .backends import Word
from .sentences import terminates_sentence

# Everything this module treats as attached punctuation. Deliberately includes
# the CJK marks: the model is multilingual and whisper emits 。？！ for Chinese
# and Japanese, which must survive the round trip like their Latin equivalents.
TRAILING = "".join([
    ".,!?;:-…",          # Latin, plus the ellipsis whisper uses for trailing off
    "。？！、，；：",   # 。？！、，；：
])
_TRAILING_RE = re.compile(f"[{re.escape(TRAILING)}]+$")

# Marks that end a sentence, as opposed to separating inside one. `terminates_sentence`
# decides the same question for a whole token including its abbreviation handling;
# this set is the character-level question asked of a single mark.
TERMINAL = ".!?…。？！"

# The model's label set: 0 (nothing), . , ? - :  -- note there is no `!`, and no
# CJK mark. That absence is the reason merge_mark exists rather than a plain
# overwrite: whisper's "Get out of here!" would come back "Get out of here."
MODEL_LABELS = ("0", ".", ",", "?", "-", ":")


@dataclass(frozen=True)
class PunctuationConfig:
    enabled: bool = True
    model: str = "oliverguhr/fullstop-punctuation-multilang-large"
    # Words per forward pass and how far the window advances. The model is a
    # 512-subtoken XLM-R; at ~1.5 subtokens per word, 200 words leaves headroom
    # for scripts that tokenize longer. The 50-word overlap is discarded on both
    # sides so no word is labelled from a window edge, where it has context on
    # only one side.
    window: int = 200
    stride: int = 150
    batch_size: int = 16
    # Whether the model may remove a full stop whisper emitted, merging two
    # captions. This is the repair for whisper ending a sentence at a mid-sentence
    # breath ("but it's the pursuit." / "It's meaningful."). OFF because
    # measuring it showed worse results.
    # Inserting punctuation into an unpunctuated run cannot make it worse, while
    # removing punctuation can, and does.
    demote_terminal: bool = False
    demote_confidence: float = 0.9
    # Drop the repair entirely rather than fail the file. A tagger that emits
    # slightly worse captions is better than one that emits none.
    required: bool = False


def affixes(raw: str) -> Tuple[str, str, str]:
    """Split a whisper word into (leading space, core, trailing punctuation).

    Whisper words carry their own leading space (" pursuit"), and sentences.py
    reassembles captions with "".join, so that space is load-bearing and has to
    survive. A token that is nothing but punctuation is returned as its own core
    so it is never silently dropped.
    """
    lead = raw[: len(raw) - len(raw.lstrip())]
    body = raw.strip()
    match = _TRAILING_RE.search(body)
    if not match or match.start() == 0:
        return lead, body, ""
    return lead, body[: match.start()], match.group()


def merge_mark(
    original: str,
    predicted: str,
    confidence: float,
    threshold: float,
    allow_demotion: bool = False,
) -> str:
    """Combine whisper's trailing mark with the model's prediction for one word.

    The asymmetry between the four cases is the whole design: adding punctuation
    to an unpunctuated run cannot make it worse, removing it can.

      both terminal          keep whisper's -- the label set has no `!` or CJK
                             mark, so overwriting flattens "Get out of here!"
      model terminal only    insert it, unconditionally
      whisper terminal only  keep whisper's, unless demote_terminal says otherwise
      neither terminal       prefer whisper's; it distinguishes ";" and "--",
                             which the label set collapses
    """
    orig_terminal = bool(original) and original[-1] in TERMINAL
    pred_terminal = predicted in (".", "?")

    if orig_terminal and pred_terminal:
        return original
    if pred_terminal and not orig_terminal:
        return predicted
    if orig_terminal and not pred_terminal:
        if not allow_demotion or confidence < threshold:
            return original
        return "" if predicted == "0" else predicted
    return original or ("" if predicted == "0" else predicted)


def recase(words: List[Word]) -> List[Word]:
    """Capitalise the first letter of every sentence. Never lowercases anything."""
    out: List[Word] = []
    start_of_sentence = True
    for word in words:
        lead, core, trail = affixes(word.word)
        if start_of_sentence and core:
            for i, char in enumerate(core):
                if char.isalpha():
                    core = core[:i] + char.upper() + core[i + 1:]
                    break
        if core or trail:
            start_of_sentence = terminates_sentence(core + trail)
        out.append(replace(word, word=lead + core + trail))
    return out


def apply_labels(
    words: List[Word],
    labels: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
    allow_demotion: bool = False,
) -> List[Word]:
    """Rewrite a word stream from per-word predictions. Pure; no model involved."""
    if len(labels) != len(words) or len(confidences) != len(words):
        raise ValueError(
            f"expected one label and confidence per word, got {len(words)} words, "
            f"{len(labels)} labels, {len(confidences)} confidences"
        )
    rewritten = []
    for word, label, confidence in zip(words, labels, confidences):
        lead, core, trail = affixes(word.word)
        # A token with nothing to punctuate -- empty, or punctuation all the way
        # through, which affixes hands back as its own core so it is not dropped.
        # Appending a mark to "..." would give "....".
        if not any(char.isalnum() for char in core):
            rewritten.append(word)
            continue
        mark = merge_mark(trail, label, confidence, threshold, allow_demotion)
        rewritten.append(replace(word, word=lead + core + mark))
    return recase(rewritten)


class PunctuationRestorer:
    """XLM-RoBERTa token classifier over the flat word stream.

    Loaded once per process. Held separately from the decoder so a caller that
    does not want it (bench comparisons against raw whisper output) simply does
    not build one.
    """

    def __init__(
        self,
        cfg: PunctuationConfig,
        weights_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        source = _local_weights(cfg.model, weights_dir) or cfg.model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForTokenClassification.from_pretrained(source)
        # fp16 halves 2.2 GB of weights sitting alongside the decoder's own GPU
        # allocation. Token classification reads an argmax over six labels, so
        # reduced precision cannot change a decision that was not already a tie.
        if device != "cpu":
            model = model.half()
        self.model = model.to(device).eval()
        self.id2label = model.config.id2label
        logger.info(f"punctuation: loaded {cfg.model} on {device}")

    def predict(self, cores: Sequence[str]) -> Tuple[List[str], List[float]]:
        """Label every word with the mark that should follow it.

        Windows overlap and only each window's interior is kept, so no word is
        labelled from a position where it had context on one side only. The first
        and last windows keep their outer edges because there is no context there
        to be had.
        """
        window, stride = self.cfg.window, self.cfg.stride
        margin = max(0, (window - stride) // 2)

        spans = []
        start = 0
        while True:
            spans.append(start)
            if start + window >= len(cores):
                break
            start += stride

        labels: List[str] = [""] * len(cores)
        confidences: List[float] = [0.0] * len(cores)
        for batch_start in range(0, len(spans), self.cfg.batch_size):
            batch = spans[batch_start: batch_start + self.cfg.batch_size]
            chunks = [list(cores[s: s + window]) for s in batch]
            encoded = self.tokenizer(
                chunks,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            with self.torch.no_grad():
                logits = self.model(
                    **{k: v.to(self.device) for k, v in encoded.items()}
                ).logits
            probabilities = logits.float().softmax(-1)
            best = probabilities.max(-1)

            for row, offset in enumerate(batch):
                word_ids = encoded.word_ids(row)
                # a word's label is read off its LAST subtoken: punctuation
                # attaches after the whole word, and only the final piece has seen
                # all of it
                per_word = {}
                for position, word_index in enumerate(word_ids):
                    if word_index is not None:
                        per_word[word_index] = position
                lo = 0 if offset == 0 else margin
                hi = len(chunks[row]) if offset + window >= len(cores) else window - margin
                for local in range(lo, min(hi, len(chunks[row]))):
                    position = per_word.get(local)
                    if position is None:
                        # word truncated away by max_length; leave whisper's own
                        # punctuation in place rather than guessing
                        continue
                    labels[offset + local] = self.id2label[int(best.indices[row, position])]
                    confidences[offset + local] = float(best.values[row, position])

        return [label or "0" for label in labels], confidences

    def restore(self, words: List[Word]) -> List[Word]:
        if not words:
            return words
        cores = [affixes(word.word)[1] for word in words]
        # A stream that is only punctuation has nothing to label and would make
        # the tokenizer produce zero word ids.
        if not any(cores):
            return words
        labels, confidences = self.predict(cores)
        return apply_labels(
            words,
            labels,
            confidences,
            self.cfg.demote_confidence,
            self.cfg.demote_terminal,
        )


def _local_weights(model: str, weights_dir: Optional[str]) -> Optional[str]:
    """Where download_weights.py puts this model, if it was run.

    Falls back to the hub id so a dev machine with network access still works;
    the image is expected to have the cache baked in or mounted, matching how the
    whisper weights are handled.
    """
    if not weights_dir:
        return None
    path = os.path.join(os.path.expanduser(weights_dir), "punctuation", model.replace("/", "--"))
    return path if os.path.isfile(os.path.join(path, "config.json")) else None


def build_punctuator(
    cfg: PunctuationConfig, weights_dir: Optional[str], device: Optional[str]
) -> Optional[PunctuationRestorer]:
    """Load the restorer, or return None if it is disabled or unavailable."""
    if not cfg.enabled:
        return None
    try:
        return PunctuationRestorer(cfg, weights_dir=weights_dir, device=device)
    except Exception as exc:  # noqa: BLE001 - any load failure degrades the same way
        if cfg.required:
            raise
        logger.warning(
            f"punctuation restoration unavailable ({type(exc).__name__}: {exc}); "
            "captions will use whisper's own punctuation"
        )
        return None
