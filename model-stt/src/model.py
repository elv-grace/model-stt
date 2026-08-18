"""Multilingual speech-to-text as an AVModel tagger.

Transcription only: faster-whisper (CTranslate2) running large-v3-turbo, which
detects the spoken language per file and transcribes it in that language.

Output tracks:
  ""              word-level tags
  "auto_captions" sentence-level tags

Every tag carries additional_info["language"], so downstream consumers never have
to infer a track's language from the config that produced it.

Timestamps are relative to the file passed to tag(), matching the rest of the
tagger runtime.

Translation is out of scope and its machinery is commented out rather than
deleted -- search this file for "DISABLED (translation)", and see
"restoring translation" in README.md for the full list of places.
"""
from __future__ import annotations

import json
import math
import statistics
import string
import subprocess
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from loguru import logger

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.av import AVModel
from common_ml.utils.metrics import timeit

from .backends import DecodeOptions, Segment, WhisperBackend, Word, build_backend
from .sentences import Sentence, to_sentences
# DISABLED (translation): path C's LLM translator
# from .translate import LLMTranslator, TranslatorConfig

WORD_TRACK = ""
SENTENCE_TRACK = "auto_captions"
# DISABLED (translation): English sentence track emitted by path C
# TRANSLATION_TRACK = "translation"

# Shortest span a real speech segment can occupy. Whisper emits a final
# end-of-audio segment that repeats the preceding sentence; openai-whisper gives
# it empty text (filtered by the empty-text check), but faster-whisper gives it
# real text collapsed into ~20ms with every word zero-duration. It reports high
# confidence, so the no_speech/logprob checks do not catch it -- five words in
# 20ms is impossible, and that is what makes it detectable.
MIN_SEGMENT_MS = 100

# How much of the previous file's transcript to carry forward as initial_prompt.
# Default with context disabled is 0.
# When enabled:
# tag() also retries without the prompt if a prompted decode returns nothing,
# since this boundary was measured on two English fixtures and should not be
# assumed to hold everywhere.
CONTEXT_CHARS = 100


@dataclass(frozen=True)
class RuntimeConfig:
    # "faster-whisper" (CTranslate2, what the image ships) | "openai" (bench only;
    # the container does not install torch/openai-whisper).
    #
    # The two produced byte-identical text on 116 FLEURS utterances -- but only on
    # clean short speech with VAD off on both, which is no longer what we ship.
    # On 600s of crowd noise they diverge, and *not* in CT2's favour by default:
    # unfiltered CT2 produced 31 repeated hallucinated segments against
    # openai-whisper's 19. openai has no VAD to enable (it detects silence from the
    # decoder's own no_speech_prob), so enabling CT2's takes it to 0 -- the choice
    # rests on VAD being available at all, not on the runtime being better raw.
    backend: str = "faster-whisper"
    model_name: str = "large-v3-turbo"
    language: Optional[str] = None  # None => detect per file

    # DISABLED (translation): task was "transcribe" | "translate", translator was
    # "whisper" (path B) | "llm" (path C). With translation out of scope, task has
    # exactly one valid value, so both fields and the model-substitution logic they
    # required are gone.
    # task: str = "transcribe"
    # translator: str = "whisper"

    # Emission, not computation: word timings are always requested from the
    # backend because the sentence track needs them to place its boundaries.
    word_level: bool = True
    sentence_level: bool = True

    # Feed the previous file's transcript tail forward as initial_prompt to
    # recover context lost by per-file transcription.
    #
    # Off by default. Measured on this fixture set, content loss from decoding
    # files independently is not systematic (chunked vs whole-file transcript
    # length ranged 75%-166%, median ~112%: whole-file decoding has its own
    # suppression failure mode). The measured harm from carrying context is
    # severe, in two modes, both reproducing on either backend:
    #   1. length -- a >=200 char prompt made whisper return zero segments for a
    #      file that yields 87 tags without one
    #   2. contamination -- a Korean tail carried into a French song produced
    #      looping Korean hallucinations and lost 88% of the words (204 -> 24)
    # Both are guarded in _transcribe_with_guards. Enable only for genuinely
    # contiguous same-language segments, and measure.
    carry_context: bool = False

    # Off by default, and coupled to vad_filter below: with VAD off the decoder
    # sees the noise it used to be shielded from, and a *conditioned* decode locks
    # onto a phrase and repeats it. The loop is prompt feedback rather than a
    # silence artifact, so turning conditioning off does not merely suppress the
    # repeats, it recovers the speech underneath them. Over NBAallstar 1680-1740s
    # a conditioned decode returned "Thank you." on repeat where an unconditioned
    # one returned "The USA First World Tournament... Voting will be over at
    # the...". Turn it back on together with vad_filter (see the clean-audio
    # profile), not on its own.
    condition_on_previous_text: bool = False

    # Silero VAD, faster-whisper only. OFF by default -- it was the default for a
    # while, and the reversal is the single largest measured change in this file,
    # so it is worth stating why.
    #
    # VAD fails silently. Audio it discards never reaches the decoder, so
    # over-filtering leaves no low-confidence segment and no repetition signal
    # behind, only missing time that nothing in the output marks. Measured over
    # 13.2h of fixtures, turning it off returned 6.6% more words and recovered 30
    # minutes of arena PA announcements from NBAallstar that Silero scored as
    # non-speech at threshold 0.5 *and* 0.25 alike -- no threshold recovers them,
    # because speech-over-crowd and crowd-without-speech are the same thing to
    # Silero and sit minutes apart in that file. The cost was 32 extra
    # artifact-shaped captions against 460 extra substantive ones, and it is
    # *faster*, since running Silero over a whole file costs more than decoding
    # the parts it would have discarded.
    #
    # Turn it on for content with distinguishable silence -- scripted film, studio
    # recordings -- where suppressing stock-phrase artifacts is worth more than
    # the recall. That is the clean-audio profile in config.yml. The thresholds
    # below are consulted only when it is on, and are deliberately not Silero's
    # own (0.5 / 400); see DecodeOptions for the sweep.
    vad_filter: bool = False
    vad_threshold: float = 0.25
    vad_min_silence_ms: int = 2000
    vad_speech_pad_ms: int = 2000

    # Largest silence (seconds) that may sit *inside* one segment before it is
    # treated as a VAD excision the timestamps were mapped across. See
    # _repair_straddles. Measured over 4096 genuine intra-segment word gaps with
    # VAD off across four files (film, animation, sports), exactly one exceeded
    # 2.0s, so this fires on artifacts and effectively never on real speech.
    straddle_gap_max: float = 2.0

    # hallucination controls: a segment is dropped only when whisper is both
    # confident it is silence and unconfident in what it decoded
    hallucination_silence_threshold: Optional[float] = 2.0
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0

    # Repetition guard, using whisper's own signals. A segment is dropped only
    # when the decode fell back at least once (temperature > 0) AND the text it
    # settled on is still repetitive (compression_ratio above whisper's own 2.4
    # threshold). Requiring both keeps it high-precision: a clean first-pass
    # decode is never touched, however repetitive its text legitimately is.
    compression_ratio_max: float = 2.4

    # Isolated-artifact guard. Whisper emits stock phrases over non-speech
    # ("Thank you.") because they are frequent in its subtitle training data.
    # Nothing in the decoder's own signals marks them: measured over NBAallstar,
    # crowd-noise "Thank you." carries no_speech_prob 0.000 and avg_logprob -0.69
    # against -0.45 for real speech in the same file, so every per-segment guard
    # above passes it.
    #
    # What does separate them is company. Real speech arrives in conversational
    # density; artifacts sit alone. Over NBAallstar, "Thank you." captions were a
    # median 18.9s from their nearest neighbour (36 of 55 more than 10s away),
    # against 0.2s for substantive captions (1588 of 1661 within 2s) -- a two
    # order of magnitude separation, which is why the threshold below is not a
    # knife-edge.
    #
    # Deliberately NOT a list of known whisper phrases. That was tried, and it
    # worked, but the list could only be assembled by reading this fixture set --
    # it would have been tuned on the data it was measured against, and would not
    # transfer to another language, where whisper has its own stock phrases.
    # Instead each file calibrates itself: a short text that recurs in the file
    # AND is *characteristically* isolated (median isolation over the threshold)
    # is that file's artifact. A phrase that is normally conversational fails the
    # median test even when a few instances happen to stand alone, so real
    # dialogue is protected by construction.
    #
    # Held out one file at a time, the learned phrase set stayed stable and no
    # validated-real caption was dropped in any held-out file. Isolation *without*
    # the recurrence test was measured to delete real lines that legitimately
    # stand alone between long musical stretches -- "Whoa.", "Hello.",
    # "Again, Moss?", "One, two, three!", each confirmed genuine by re-decoding
    # its audio -- so both conditions are required.
    #
    # Set isolated_artifact_gap to 0 to disable.
    isolated_artifact_gap: float = 10.0
    isolated_artifact_max_words: int = 3
    # a one-off is left alone: recurrence is what distinguishes a decoder tic
    # from a short line that merely happens to sit in a quiet stretch
    isolated_artifact_min_count: int = 4

    # NOT ADDED, deliberately: a cross-segment guard dropping runs of identical
    # text. It looks like the obvious fix for a looping decode, which the
    # per-segment guards above structurally cannot catch (each repeat is short,
    # clean, and individually non-repetitive). But the runs sit on real audio.
    # Over NBAallstar 1680-1740s a conditioned decode repeated "Thank you."
    # while the same audio decoded with condition_on_previous_text=False
    # returned "The USA First World Tournament... Voting will be over at the...".
    # Dropping the run would have deleted that speech; turning conditioning off
    # recovers it. Fix the cause, do not delete the symptom -- see the two-stage
    # profile in config.yml.

    # Make the decoder reproducible by replacing whisper's *sampled* retry rungs
    # with deterministic ones. faster-whisper only. See _DeterministicFallback for
    # the measurements; in short it recovers most of the repetition suppression
    # (55 repeated segments against the sampled ladder's 30, and 228 with no retry
    # at all) and all of the reproducibility, at no cost on FLEURS.
    #
    # Off by default because the residual gap is real. Turn it on when
    # reproducibility is worth more than the last of the suppression: benchmarking,
    # diffing two runs, cache keys, QA sign-off. The `reproducible` profile in
    # config.yml does exactly that and nothing else.
    deterministic_fallback: bool = False

    beam_size: Optional[int] = None
    compute_type: str = "float16"  # faster-whisper only
    device: Optional[str] = None


class WhisperSTT(AVModel):
    def __init__(
        self,
        cfg: RuntimeConfig,
        models: Dict,
        weights_dir: str,
        sentence_gap_ms: float = 5000,
        max_caption_words: int = 100,
    ):
        self.cfg = cfg
        self.sentence_gap_ms = sentence_gap_ms
        self.max_caption_words = max_caption_words
        _validate(cfg, models)

        # DISABLED (translation): was _resolve_model(), which substituted
        # translate-capable weights when translation was requested with turbo.
        self.effective_model_name = cfg.model_name

        with timeit(f"loading {self.effective_model_name} via {cfg.backend}"):
            self.backend: WhisperBackend = build_backend(
                backend=cfg.backend,
                model_name=self.effective_model_name,
                models=models,
                weights_dir=weights_dir,
                device=cfg.device,
                compute_type=cfg.compute_type,
            )

        self._prev_tail: Optional[str] = None
        self._prev_language: Optional[str] = None

    def reset_context(self) -> None:
        """Drop carried context. Call between unrelated assets.

        The tagger runtime feeds contiguous segments of one asset, so context
        carries safely within a run; a benchmark or batch job that walks
        unrelated files must reset between them or a previous file's language
        can contaminate the next.
        """
        self._prev_tail = None
        self._prev_language = None

    def _transcribe_with_guards(self, fpath: str, opts: DecodeOptions):
        """Decode, re-running without the carried prompt if it looks harmful.

        A carried initial_prompt can override the audio outright: it either
        suppresses the file entirely, or drags the output into the prompt's
        language. Both cost one extra decode to detect and recover from, which is
        cheap next to losing the file's tags.
        """
        result = self.backend.transcribe(fpath, opts)
        if opts.initial_prompt is None:
            return result

        if not result.segments:
            reason = "no segments returned"
        elif (
            self._prev_language
            and result.language
            and result.language != self._prev_language
        ):
            # the prompt came from a different language than this file is in
            reason = f"language changed {self._prev_language} -> {result.language}"
        else:
            return result

        logger.warning(f"{fpath}: carried context looks harmful ({reason}); re-decoding without it")
        return self.backend.transcribe(fpath, replace(opts, initial_prompt=None))

    def tag(self, fpath: str) -> List[Tag]:
        if not has_audio_stream(fpath):
            logger.warning(f"{fpath} has no audio stream, skipping")
            return []

        opts = DecodeOptions(
            # DISABLED (translation): was "translate" for path B
            task="transcribe",
            language=self.cfg.language,
            word_timestamps=True,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            initial_prompt=self._prev_tail if self.cfg.carry_context else None,
            beam_size=self.cfg.beam_size,
            no_speech_threshold=self.cfg.no_speech_prob_max,
            logprob_threshold=self.cfg.avg_logprob_min,
            hallucination_silence_threshold=self.cfg.hallucination_silence_threshold,
            vad_filter=self.cfg.vad_filter,
            vad_threshold=self.cfg.vad_threshold,
            vad_min_silence_ms=self.cfg.vad_min_silence_ms,
            vad_speech_pad_ms=self.cfg.vad_speech_pad_ms,
            deterministic_fallback=self.cfg.deterministic_fallback,
        )

        result = self._transcribe_with_guards(fpath, opts)
        # before _filter_segments: a straddled segment has an inflated span, so
        # the sub-100ms repair there must see the corrected timings, not the
        # inflated ones
        segments = result.segments
        # Only with VAD on. A straddle is restore_speech_timestamps mapping a
        # segment's words back across an excision, and that runs only when
        # faster-whisper was handed speech chunks. With VAD off the timestamps are
        # already in source-media time, so every intra-segment gap is REAL and
        # repairing one does active harm: measured over 13.2h decoded without VAD
        # it fired 9 times, all false positives. On the clearest, whisper merged an
        # "Oh," at 32.24s with a "my God." at 38.48s into one segment; the repair
        # anchored on "Oh," and emitted the phrase at 32.24-33.80, placing
        # "my God." five seconds from where it was said. Another collapsed a
        # segment to zero duration.
        if self.cfg.vad_filter:
            segments = _repair_straddles(segments, self.cfg.straddle_gap_max)
        segments = self._filter_segments(segments)
        # last: isolation is measured against the segments that survive, so a
        # neighbour dropped as silence above must not still count as company
        segments = self._drop_isolated_artifacts(segments)

        # detected per file; path B used to force "en" here since whisper's
        # translate task emits English regardless of the source language
        text_language = result.language

        if self.cfg.carry_context:
            self._prev_tail = _context_tail(" ".join(s.text.strip() for s in segments))
            self._prev_language = result.language

        tags: List[Tag] = []
        if self.cfg.word_level:
            tags.extend(self._word_tags(segments, fpath, text_language))

        sentences = to_sentences(segments, self.sentence_gap_ms, self.max_caption_words)
        if self.cfg.sentence_level:
            tags.extend(self._sentence_tags(sentences, fpath, text_language, SENTENCE_TRACK))

        # DISABLED (translation): path C emitted an English sentence track here,
        #   if self.translator is not None and sentences:
        #       translated = self.translator.translate(sentences, result.language)
        #       tags.extend(self._sentence_tags(translated, fpath, "en", TRANSLATION_TRACK))

        tags.sort(key=lambda t: (t.start_time, t.track))
        return tags

    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        """Drop silence and hallucinations; repair collapsed timestamps.

        Sub-100ms segments are physically impossible speech, but they are not all
        the same thing. Measured over 13.2h of real media, 13 occurred: some were
        verbatim repeats of the preceding segment (whisper's end-of-audio
        artifact, e.g. 'Bye!' between 'Bye!' and 'Bye!'), and some were genuine
        short interjections ('Right?', 'Good.') whose word alignment collapsed.
        Dropping both loses real content; keeping both duplicates text. So the
        repeats are dropped and the rest are kept with a repaired span.
        """
        kept: List[Segment] = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue

            # whisper's own heuristic: assert silence only when the no-speech
            # probability is high AND the decode was itself low-confidence, so a
            # confident transcription over noisy audio is not thrown away
            if (
                seg.no_speech_prob > self.cfg.no_speech_prob_max
                and seg.avg_logprob < self.cfg.avg_logprob_min
            ):
                logger.debug(f"dropping likely-silent segment at {seg.start:.1f}s")
                continue

            # whisper already retries repetitive decodes at higher temperature;
            # if it fell back and the result is *still* repetitive, the fallback
            # failed and the text is looping
            if seg.temperature > 0 and seg.compression_ratio > self.cfg.compression_ratio_max:
                logger.debug(
                    f"dropping repetitive segment at {seg.start:.1f}s "
                    f"(compression_ratio={seg.compression_ratio:.2f}, "
                    f"temperature={seg.temperature:.1f}): {text[:40]!r}"
                )
                continue

            if self._to_milliseconds(seg.end - seg.start) >= MIN_SEGMENT_MS:
                kept.append(seg)
                continue

            if kept and _is_repeat_of(text, kept[-1].text):
                logger.debug(
                    f"dropping repeated {seg.end - seg.start:.3f}s segment at "
                    f"{seg.start:.1f}s: {text[:40]!r}"
                )
                continue

            # real speech with a collapsed span: give it a plausible duration
            # without letting it run into whatever comes next
            end = seg.start + MIN_SEGMENT_MS / 1000
            next_start = segments[i + 1].start if i + 1 < len(segments) else None
            if next_start is not None:
                end = min(end, max(next_start, seg.end))
            logger.debug(
                f"repairing {seg.end - seg.start:.3f}s span at {seg.start:.1f}s "
                f"-> {end - seg.start:.3f}s: {text[:40]!r}"
            )
            kept.append(replace(seg, end=end, words=_stretch_words(seg.words, seg.start, end)))

        return kept

    def _drop_isolated_artifacts(self, segments: List[Segment]) -> List[Segment]:
        """Drop a file's recurring stock phrases where they stand alone.

        Two conditions, both required, neither sufficient. See
        RuntimeConfig.isolated_artifact_gap for the measurements.
        """
        gap = self.cfg.isolated_artifact_gap
        if gap <= 0 or not segments:
            return segments

        def isolation(i: int) -> float:
            before = segments[i].start - segments[i - 1].end if i > 0 else math.inf
            after = segments[i + 1].start - segments[i].end if i + 1 < len(segments) else math.inf
            return min(before, after)

        candidates: Dict[str, List[int]] = {}
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text or len(text.split()) > self.cfg.isolated_artifact_max_words:
                continue
            key = _normalize_text(text)
            if key:
                candidates.setdefault(key, []).append(i)

        drop = set()
        for key, indices in candidates.items():
            if len(indices) < self.cfg.isolated_artifact_min_count:
                continue
            # the phrase must be isolated *as a rule*, not just in places
            if statistics.median(isolation(i) for i in indices) < gap:
                continue
            isolated = [i for i in indices if isolation(i) >= gap]
            logger.debug(
                f"dropping {len(isolated)} isolated occurrences of {key!r} "
                f"({len(indices)} total in file)"
            )
            drop.update(isolated)

        return [seg for i, seg in enumerate(segments) if i not in drop]

    def _word_tags(
        self, segments: List[Segment], fpath: str, language: Optional[str]
    ) -> List[Tag]:
        tags = []
        for seg in segments:
            for word in seg.words:
                text = word.word.strip()
                if not text:
                    continue
                tags.append(
                    Tag(
                        start_time=self._to_milliseconds(word.start),
                        end_time=self._to_milliseconds(word.end),
                        source_media=fpath,
                        tag=text,
                        track=WORD_TRACK,
                        additional_info={
                            "language": language,
                            "probability": round(word.probability, 4),
                            "avg_logprob": round(seg.avg_logprob, 4),
                            "no_speech_prob": round(seg.no_speech_prob, 4),
                            # repetition signals, so a consumer can filter on the
                            # same evidence _filter_segments uses
                            "compression_ratio": round(seg.compression_ratio, 3),
                            "temperature": round(seg.temperature, 2),
                        },
                    )
                )
        return tags

    def _sentence_tags(
        self, sentences: List[Sentence], fpath: str, language: Optional[str], track: str
    ) -> List[Tag]:
        tags = []
        for s in sentences:
            if not s.text.strip():
                continue
            info = {"language": language, "model": self.effective_model_name}
            # omitted rather than null when whisper returned no word timings, so
            # a consumer can tell "not measured" from "measured and low"
            if s.min_word_probability is not None:
                info["min_word_probability"] = s.min_word_probability
                info["mean_word_probability"] = s.mean_word_probability
            tags.append(
                Tag(
                    start_time=self._to_milliseconds(s.start),
                    end_time=self._to_milliseconds(s.end),
                    source_media=fpath,
                    tag=s.text,
                    track=track,
                    additional_info=info,
                )
            )
        return tags


def _normalize_text(text: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed."""
    stripped = text.strip().lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(stripped.split())


def _is_repeat_of(fragment: str, previous: str) -> bool:
    """Whether a collapsed fragment is just a repeat of the previous segment.

    Whisper's end-of-audio artifact is often a *near* repeat that drops a leading
    word ("The prince didn't answer." after "But the prince didn't answer."), so
    exact equality misses it. Containment alone is too loose in the other
    direction -- a genuine "No!" is contained in a preceding "No, I don't know" --
    so containment only counts once the fragment is long enough to be unlikely by
    chance. Short interjections must match exactly.
    """
    a, b = _normalize_text(fragment), _normalize_text(previous)
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a.split()) >= 3 and (a in b or b in a)


def _repair_straddles(segments: List[Segment], max_gap: float) -> List[Segment]:
    """Pull back words whose timestamps were mapped across a VAD excision.

    faster-whisper decodes the VAD-*concatenated* audio and then maps each word
    back to original time (restore_speech_timestamps), resolving each word to a
    speech chunk by its midpoint. A word whose alignment lands within a frame or
    two of a chunk boundary resolves to the wrong side, and because the segment's
    span is taken from its first and last word, one misplaced word stretches the
    whole segment across the excision.

    Measured on spiderman-into-the-spiderverse-10min, the segment "Good luck,
    guys." was emitted spanning 126.71s -> 333.92s: "Good" resolved to the chunk
    before a 207s excision, "luck," and "guys." to the chunk after. Decoding the
    same audio unfiltered puts the whole phrase at 333.06-333.90.

    A segment is one contiguous utterance by construction -- whisper cannot emit a
    segment spanning more than its 30s window -- so when its words fall into runs
    separated by more than `max_gap`, exactly one run is in the right place and
    the rest are misplaced. The run holding the most spoken time wins (it has the
    most evidence behind it, and on the case above it is also the correct one),
    and the others are laid back against it in order, each word keeping its own
    duration.

    Re-anchoring rather than splitting is what keeps the sentence track whole:
    Without it the segment's span stays inflated across the excision, and every
    word after the cut is reported at a time it was not spoken.
    """
    repaired: List[Segment] = []
    for i, seg in enumerate(segments):
        words = seg.words
        if len(words) < 2:
            repaired.append(seg)
            continue

        # split into runs of words with no excision-sized gap between them
        runs: List[List[Word]] = [[words[0]]]
        for prev, word in zip(words, words[1:]):
            if word.start - prev.end > max_gap:
                runs.append([])
            runs[-1].append(word)
        if len(runs) == 1:
            repaired.append(seg)
            continue

        anchor = max(runs, key=lambda run: (sum(w.end - w.start for w in run), len(run)))
        pivot = runs.index(anchor)
        before = [w for run in runs[:pivot] for w in run]
        after = [w for run in runs[pivot + 1:] for w in run]

        # do not run into the neighbouring segments, which keep their own timings
        floor = repaired[-1].end if repaired else 0.0
        ceiling = segments[i + 1].start if i + 1 < len(segments) else None

        fixed = (
            _pack_words(before, end=anchor[0].start, floor=floor)
            + list(anchor)
            + _pack_words(after, start=anchor[-1].end, ceiling=ceiling)
        )
        logger.debug(
            f"repairing segment straddling a VAD cut at {seg.start:.1f}s: "
            f"{seg.start:.1f}-{seg.end:.1f}s -> {fixed[0].start:.1f}-{fixed[-1].end:.1f}s: "
            f"{seg.text.strip()[:40]!r}"
        )
        repaired.append(replace(seg, start=fixed[0].start, end=fixed[-1].end, words=fixed))

    return repaired


def _pack_words(
    words: List[Word],
    start: Optional[float] = None,
    end: Optional[float] = None,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> List[Word]:
    """Lay words back-to-back against an anchor, compressing if space is short.

    Exactly one of `start` (pack forwards from here) or `end` (pack backwards to
    here) is given. `floor`/`ceiling` are the neighbouring segments' edges; when
    the words do not fit between the anchor and that edge they are scaled down
    uniformly rather than allowed to overlap.
    """
    if not words:
        return []

    total = sum(w.end - w.start for w in words)
    if end is not None:
        start = end - total
        if floor is not None and start < floor:
            start = floor
            total = max(end - floor, 0.0)
    elif ceiling is not None and start + total > ceiling:
        total = max(ceiling - start, 0.0)

    # scale > 1 never happens: total only ever shrinks above
    scale = total / sum(w.end - w.start for w in words) if total > 0 else 0.0
    packed: List[Word] = []
    cursor = start
    for word in words:
        duration = (word.end - word.start) * scale
        packed.append(replace(word, start=cursor, end=cursor + duration))
        cursor += duration
    return packed


def _stretch_words(words: List[Word], start: float, end: float) -> List[Word]:
    """Spread words evenly across a repaired span.

    A collapsed segment has every word on the same timestamp, so there is no
    real timing to preserve -- evenly spaced is the least misleading guess, and
    it keeps word tags inside their segment.
    """
    if not words:
        return words
    step = (end - start) / len(words)
    return [
        replace(w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]


def _context_tail(text: str, limit: int = CONTEXT_CHARS) -> Optional[str]:
    """The trailing context to prime the next file with, at most `limit` chars.

    Starts at a sentence boundary where one falls inside the window, else at a
    word boundary, so the prompt never opens on a fragment: a raw slice yielded
    prompts like "hen there was a 1914-18 war", and the fragment-opening variant
    measured worse than sentence-aligned prompts of comparable length.
    """
    text = text.strip()
    if not text:
        return None
    if len(text) <= limit:
        return text

    window = text[-limit:]
    # earliest boundary in the window gives the longest well-formed suffix
    starts = [window.find(d) + len(d) for d in (". ", "? ", "! ") if d in window]
    if not starts:
        space = window.find(" ")
        starts = [space + 1] if space != -1 else []
    if not starts:
        return window

    return window[min(starts):].strip() or None


# DISABLED (translation): _is_llm_translation() selected path C, and
# _resolve_model() substituted translate-capable weights when translation was
# requested with turbo (which accepts task="translate" but returns untranslated
# source-language text). Neither is reachable without a task/translator field.


def _validate(cfg: RuntimeConfig, models: Dict) -> None:
    if cfg.backend not in ("openai", "faster-whisper"):
        raise ValueError(f"backend must be 'openai' or 'faster-whisper', got {cfg.backend!r}")
    if cfg.model_name not in models:
        raise ValueError(f"unknown model {cfg.model_name!r}; known: {sorted(models)}")


def has_audio_stream(fpath: str) -> bool:
    """True if ffprobe reports at least one audio stream.

    whisper shells out to ffmpeg and gets an empty buffer for a video with no
    audio, which surfaces as an opaque decode error much later. Check up front.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        fpath,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
        return bool(json.loads(out).get("streams"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"could not probe {fpath} for audio streams ({e}); attempting anyway")
        return True
