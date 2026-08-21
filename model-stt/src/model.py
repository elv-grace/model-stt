"""Multilingual speech-to-text as an AVModel tagger.

Transcription only: faster-whisper (CTranslate2) running large-v3-turbo, which
detects the spoken language per file and transcribes it in that language.

Output tracks:
  ""              word-level tags
  "auto_captions" sentence-level tags

Every tag carries additional_info["language"] so downstream consumers never have
to infer a track's language from the config that produced it, 
additional_info["model"], and additional_info["min/mean_word_probability"].

Timestamps are relative to the file passed to tag(), matching the rest of the
tagger runtime.

Translation is out of scope and its machinery is commented out rather than
deleted -- search this file for "DISABLED (translation)", and see
"restoring translation" in README.md.
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
from .punctuate import PunctuationConfig, PunctuationRestorer, build_punctuator
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
    """Decoder and filtering settings, overridden per run by config.yml profiles.

    Comments here say what a field is and why its default is what it is. Tuning
    guidance is in config.yml beside the profile that sets it; the measurements
    behind every default are in README.md.
    """

    backend: str = "faster-whisper" # | "openai"
    model_name: str = "large-v3-turbo"
    language: Optional[str] = None  # detect per file

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

    # Carry the previous file's transcript tail forward as initial_prompt. Off:
    # a long prompt can suppress a file entirely and a foreign-language one can
    # override the audio. Both are guarded in _transcribe_with_guards.
    carry_context: bool = False

    # Paired with vad_filter, not independent: with VAD off the decoder sees noise
    # it was shielded from, and a conditioned decode locks onto a phrase and
    # repeats it. Flip both together or neither -- a test guards the coupling.
    condition_on_previous_text: bool = False

    # Silero VAD, faster-whisper only. Off because it fails silently: audio it
    # discards never reaches the decoder, so over-filtering leaves no signal
    # behind, only missing time. Thresholds are swept, not Silero's own.
    vad_filter: bool = False
    vad_threshold: float = 0.25
    vad_min_silence_ms: int = 2000
    vad_speech_pad_ms: int = 2000

    # Largest silence that may sit inside one segment before it is read as a VAD
    # excision the timestamps were mapped across. See _repair_straddles.
    straddle_gap_max: float = 2.0

    # A segment is dropped only when whisper is both confident it is silence and
    # unconfident in what it decoded -- either alone is not evidence.
    hallucination_silence_threshold: Optional[float] = 2.0
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0

    # Repetition guard: drop only when the decode already fell back
    # (temperature > 0) AND the text it settled on is still repetitive. Both
    # conditions keep it off a clean first-pass decode, however repetitive.
    compression_ratio_max: float = 2.4

    # Isolated-artifact guard. Whisper emits stock phrases ("Thank you.") over
    # non-speech and no decoder signal marks them; what separates them is company,
    # since real speech arrives in conversational density and artifacts sit alone.
    # Each file calibrates its own artifact set, so a phrase that is conversational
    # in one file is protected there and dropped in another. 0 disables.
    isolated_artifact_gap: float = 10.0
    isolated_artifact_max_words: int = 3
    isolated_artifact_min_count: int = 4

    # Replace whisper's sampled retry rungs with deterministic ones, making output
    # reproducible. faster-whisper only; see _DeterministicFallback.
    deterministic_fallback: bool = True

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
        punctuation: Optional[PunctuationConfig] = None,
        punctuator: Optional[PunctuationRestorer] = None,
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

        # Injectable so tests can drive the pipeline without loading 2.2 GB of
        # XLM-R, and so bench/ can compare against whisper's raw punctuation by
        # passing an explicitly disabled config.
        if punctuator is not None:
            self.punctuator: Optional[PunctuationRestorer] = punctuator
        else:
            self.punctuator = build_punctuator(
                punctuation or PunctuationConfig(),
                weights_dir=weights_dir,
                device=cfg.device,
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
        language. Both cost one extra decode to detect and recover from.
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
        # This order is load-bearing. Straddle repair first, so _filter_segments'
        # sub-100ms check sees corrected spans rather than inflated ones -- and
        # only with VAD on, since without it every intra-segment gap is real and
        # repairing one moves real speech. Isolation is measured after filtering,
        # so a neighbour dropped as silence cannot still count as company.
        # Punctuation last, before to_sentences reads it.
        segments = result.segments
        if self.cfg.vad_filter:
            segments = _repair_straddles(segments, self.cfg.straddle_gap_max)
        segments = self._filter_segments(segments)
        segments = self._drop_isolated_artifacts(segments)
        segments = self._repunctuate(segments)

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

    def _repunctuate(self, segments: List[Segment]) -> List[Segment]:
        """Re-decide punctuation over the whole file's words at once.

        Flattened across segments deliberately. Words are rewritten in place, keeping every timing untouched: this changes
        the characters attached to a word, never which word sits where. The
        segment's own text is rebuilt to match so the word track and the sentence
        track cannot disagree. A failure degrades to whisper's own punctuation rather than losing the file."""
        if self.punctuator is None or not segments:
            return segments

        words = [word for seg in segments for word in seg.words]
        if not words:
            return segments

        try:
            with timeit(f"restoring punctuation over {len(words)} words"):
                restored = self.punctuator.restore(words)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"punctuation restoration failed ({type(exc).__name__}: {exc}); "
                "keeping whisper's punctuation"
            )
            return segments

        if len(restored) != len(words):
            logger.warning(
                f"punctuation restoration returned {len(restored)} words for "
                f"{len(words)}; keeping whisper's punctuation"
            )
            return segments

        out: List[Segment] = []
        cursor = 0
        for seg in segments:
            if not seg.words:
                out.append(seg)
                continue
            chunk = restored[cursor: cursor + len(seg.words)]
            cursor += len(seg.words)
            out.append(replace(seg, words=chunk, text="".join(w.word for w in chunk)))
        return out

    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        """Drop silence and hallucinations (repetitions); repair collapsed timestamps."""
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
            # the phrase must be isolated as a rule, not just in places
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

    Whisper's end-of-audio artifact is often a near repeat that drops a leading
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

    restore_speech_timestamps resolves each word to a speech chunk by its
    midpoint, so a word aligned within a frame of a chunk boundary lands on the
    wrong side -- and since a segment's span comes from its first and last word,
    one misplaced word stretches the segment across the excision.

    A segment is one contiguous utterance by construction (whisper cannot exceed
    its 30s window), so when its words fall into runs separated by more than
    `max_gap`, exactly one run is correctly placed. The run holding the most
    spoken time wins; the others are laid back against it in order, each word
    keeping its duration. VAD-only -- see the caller. Details in README.
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
    audio, which surfaces as an opaque decode error much later. Check up front instead.
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
