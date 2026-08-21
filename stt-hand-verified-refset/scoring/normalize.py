"""
Shared text normalization for the hand-verified STT reference set.

BOTH the transcription protocol and the scorer depend on these rules. If you
change anything here, update protocol/transcription_protocol.md to match, or the
verbatim references and the scorer will silently disagree.

Design goal: a normalization that is fair to a *verbatim* reference. In
particular, fillers ("um", "uh") are KEPT by default, because the whole point of
the reference set is to measure the gap a subtitle (which drops fillers) hides.

Everything here is deterministic and dependency-free.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Words a transcriber may write for hesitations/fillers. Kept as tokens by
# default (verbatim). Scored-away only when drop_fillers=True.
FILLERS = {
    "um", "uh", "erm", "er", "hmm", "mm", "mhm", "uh-huh", "mm-hmm",
    "huh", "eh", "ah", "oh", "hm",
}

# Bracket/paren spans are ANNOTATIONS, not spoken words: [music], [laughter],
# (phone), (unintelligible). They are removed before scoring. Keep them in the
# reference for human readers and for auditing what was non-speech.
_BRACKET_RE = re.compile(r"[\[\(\{][^\]\)\}]*[\]\)\}]")

# Standalone integer/decimal token, optionally decorated with $ , . %.
_NUMBER_RE = re.compile(r"^\$?-?\d[\d,]*(?:\.\d+)?%?$")

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    """Integer -> spoken words for 0..999,999. Heuristic; documented as such."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        w = _TENS[n // 10]
        return w if n % 10 == 0 else f"{w} {_ONES[n % 10]}"
    if n < 1000:
        rest = n % 100
        head = f"{_ONES[n // 100]} hundred"
        return head if rest == 0 else f"{head} {_int_to_words(rest)}"
    if n < 1_000_000:
        rest = n % 1000
        head = f"{_int_to_words(n // 1000)} thousand"
        return head if rest == 0 else f"{head} {_int_to_words(rest)}"
    return str(n)  # large numbers left as digits (rare in dialogue)


def _spell_number_token(tok: str) -> str:
    """Best-effort spell-out of a numeric token -> space-joined words.

    Convenience for hypotheses that emit digits ("9800") when the reference has
    words. It cannot resolve genuine ambiguity ("1900" -> "nineteen hundred" vs
    "one thousand nine hundred"), so the protocol asks you to WRITE NUMBERS AS
    WORDS in the reference; by default this only touches the hypothesis side.
    """
    had_percent = tok.endswith("%")
    core = tok.replace("$", "").replace(",", "").rstrip("%")
    if "." in core:
        left, right = core.split(".", 1)
        left_w = _int_to_words(int(left)) if left.lstrip("-").isdigit() else left
        right_w = " ".join(_ONES[int(d)] for d in right if d.isdigit())
        out = f"{left_w} point {right_w}".strip()
    else:
        out = _int_to_words(int(core)) if core.lstrip("-").isdigit() else core
    if had_percent:
        out += " percent"
    return out


@dataclass
class NormConfig:
    lowercase: bool = True
    strip_annotations: bool = True    # remove [..]/(..)/{..} spans
    split_hyphens: bool = True        # "gluten-free" -> "gluten free"
    drop_fillers: bool = False        # KEEP fillers by default (verbatim)
    spell_digits_in_hyp: bool = True  # spell out digit tokens (hypothesis side)
    spell_digits_in_ref: bool = False
    keep_apostrophes: bool = True     # keep contractions ("don't") intact


def _clean_token(tok: str, cfg: NormConfig) -> list[str]:
    tok = tok.replace("’", "'")  # curly -> straight apostrophe
    if cfg.keep_apostrophes:
        tok = re.sub(r"[^\w']+", " ", tok, flags=re.UNICODE)
    else:
        tok = re.sub(r"[^\w]+", " ", tok, flags=re.UNICODE)
    pieces = []
    for piece in tok.split():
        piece = piece.strip("'")  # drop stray edge apostrophes/quote marks
        if piece:
            pieces.append(piece)
    return pieces


def _normalize(text: str, cfg: NormConfig, is_ref: bool) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    if cfg.strip_annotations:
        text = _BRACKET_RE.sub(" ", text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.split_hyphens:
        for dash in ("-", "–", "—"):
            text = text.replace(dash, " ")

    spell = cfg.spell_digits_in_ref if is_ref else cfg.spell_digits_in_hyp
    out: list[str] = []
    for tok in text.split():
        if spell and _NUMBER_RE.match(tok):
            out.extend(_spell_number_token(tok).split())
            continue
        for piece in _clean_token(tok, cfg):
            if cfg.drop_fillers and piece in FILLERS:
                continue
            out.append(piece)
    return out


def normalize_ref(text: str, cfg: NormConfig | None = None) -> list[str]:
    return _normalize(text, cfg or NormConfig(), is_ref=True)


def normalize_hyp(text: str, cfg: NormConfig | None = None) -> list[str]:
    return _normalize(text, cfg or NormConfig(), is_ref=False)


def normalize_lines(text: str, cfg: NormConfig | None = None, is_ref: bool = True):
    """Return a list of token-lists, one per non-empty input line.

    Used by the scorer to attribute alignment operations back to reference
    utterances for bootstrap resampling. Keep one utterance per line in your
    reference files.
    """
    cfg = cfg or NormConfig()
    lines = []
    for line in text.splitlines():
        toks = _normalize(line, cfg, is_ref=is_ref)
        if toks:
            lines.append(toks)
    return lines


if __name__ == "__main__":
    import sys
    print(" ".join(normalize_ref(sys.stdin.read(), NormConfig())))
