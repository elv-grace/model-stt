"""Whisper speech-to-text as an AVModel tagger.

Three configurations, selected by (model_name, task, translator):

  A  transcribe / large-v3-turbo   fast multilingual ASR. The default.
  B  translate  / large-v3         whisper's native X->English speech translation.
                                   Requires non-turbo weights: large-v3-turbo was
                                   distilled without the translation task and
                                   silently returns source-language text, so a
                                   turbo request here is auto-upgraded.
  C  translate  / large-v3-turbo   turbo transcription, then LLM translation of the
     + translator=llm              sentence track. Emits both the source-language
                                   transcript and the English translation.

Output tracks:
  ""              word-level tags (source language for A/C, English for B)
  "auto_captions" sentence-level tags, same language as the word track
  "translation"   English sentence-level tags, path C only

Every tag carries additional_info["language"], so downstream consumers never have
to infer a track's language from the config that produced it.

Timestamps are relative to the file passed to tag(), matching the rest of the
tagger runtime.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from loguru import logger

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.av import AVModel
from common_ml.utils.metrics import timeit

from .backends import DecodeOptions, Segment, WhisperBackend, build_backend, resolve_weights
from .sentences import Sentence, to_sentences
from .translate import LLMTranslator, TranslatorConfig

WORD_TRACK = ""
SENTENCE_TRACK = "auto_captions"
TRANSLATION_TRACK = "translation"

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
    backend: str = "openai"  # "openai" | "faster-whisper"
    model_name: str = "large-v3-turbo"
    task: str = "transcribe"  # "transcribe" | "translate"
    translator: str = "whisper"  # "whisper" (path B) | "llm" (path C)
    language: Optional[str] = None  # None => detect per file

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
    condition_on_previous_text: bool = True

    # hallucination controls: a segment is dropped only when whisper is both
    # confident it is silence and unconfident in what it decoded
    hallucination_silence_threshold: Optional[float] = 2.0
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0

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
        translate_fallback: Optional[str] = None,
        translator_cfg: Optional[TranslatorConfig] = None,
    ):
        self.cfg = cfg
        self.sentence_gap_ms = sentence_gap_ms
        _validate(cfg, models)

        # Resolved before build_backend so the substituted weights follow the
        # substituted name: weight lookup is keyed on model_name alone.
        self.effective_model_name = _resolve_model(cfg, models, translate_fallback, weights_dir)

        with timeit(f"loading {self.effective_model_name} via {cfg.backend}"):
            self.backend: WhisperBackend = build_backend(
                backend=cfg.backend,
                model_name=self.effective_model_name,
                models=models,
                weights_dir=weights_dir,
                device=cfg.device,
                compute_type=cfg.compute_type,
            )

        self.translator: Optional[LLMTranslator] = None
        if _is_llm_translation(cfg):
            self.translator = LLMTranslator(translator_cfg or TranslatorConfig())

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

        # whisper's own task is "translate" only for path B. Path C transcribes in
        # the source language and translates the text afterwards.
        whisper_task = (
            "translate"
            if self.cfg.task == "translate" and not _is_llm_translation(self.cfg)
            else "transcribe"
        )

        opts = DecodeOptions(
            task=whisper_task,
            language=self.cfg.language,
            word_timestamps=True,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            initial_prompt=self._prev_tail if self.cfg.carry_context else None,
            beam_size=self.cfg.beam_size,
            no_speech_threshold=self.cfg.no_speech_prob_max,
            logprob_threshold=self.cfg.avg_logprob_min,
            hallucination_silence_threshold=self.cfg.hallucination_silence_threshold,
        )

        result = self._transcribe_with_guards(fpath, opts)
        segments = [s for s in result.segments if self._keep(s)]

        # path B emits English regardless of what language was detected
        text_language = "en" if whisper_task == "translate" else result.language

        if self.cfg.carry_context:
            self._prev_tail = _context_tail(" ".join(s.text.strip() for s in segments))
            self._prev_language = result.language

        tags: List[Tag] = []
        if self.cfg.word_level:
            tags.extend(self._word_tags(segments, fpath, text_language))

        sentences = to_sentences(segments, self.sentence_gap_ms)
        if self.cfg.sentence_level:
            tags.extend(self._sentence_tags(sentences, fpath, text_language, SENTENCE_TRACK))

        if self.translator is not None and sentences:
            translated = self.translator.translate(sentences, result.language)
            tags.extend(self._sentence_tags(translated, fpath, "en", TRANSLATION_TRACK))

        tags.sort(key=lambda t: (t.start_time, t.track))
        return tags

    def _keep(self, seg: Segment) -> bool:
        if not seg.text.strip():
            return False
        if self._to_milliseconds(seg.end - seg.start) < MIN_SEGMENT_MS:
            # an end-of-audio repetition collapsed onto a single timestamp; it is
            # reported as confident, so only the impossible duration reveals it
            logger.debug(
                f"dropping degenerate {seg.end - seg.start:.3f}s segment at "
                f"{seg.start:.1f}s: {seg.text.strip()[:40]!r}"
            )
            return False
        # whisper's own heuristic: assert silence only when the no-speech
        # probability is high AND the decode was itself low-confidence, so a
        # confident transcription over noisy audio is not thrown away
        if (
            seg.no_speech_prob > self.cfg.no_speech_prob_max
            and seg.avg_logprob < self.cfg.avg_logprob_min
        ):
            logger.debug(f"dropping likely-silent segment at {seg.start:.1f}s")
            return False
        return True

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
                        },
                    )
                )
        return tags

    def _sentence_tags(
        self, sentences: List[Sentence], fpath: str, language: Optional[str], track: str
    ) -> List[Tag]:
        return [
            Tag(
                start_time=self._to_milliseconds(s.start),
                end_time=self._to_milliseconds(s.end),
                source_media=fpath,
                tag=s.text,
                track=track,
                additional_info={"language": language, "model": self.effective_model_name},
            )
            for s in sentences
            if s.text.strip()
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


def _is_llm_translation(cfg: RuntimeConfig) -> bool:
    return cfg.task == "translate" and cfg.translator == "llm"


def _validate(cfg: RuntimeConfig, models: Dict) -> None:
    if cfg.task not in ("transcribe", "translate"):
        raise ValueError(f"task must be 'transcribe' or 'translate', got {cfg.task!r}")
    if cfg.translator not in ("whisper", "llm"):
        raise ValueError(f"translator must be 'whisper' or 'llm', got {cfg.translator!r}")
    if cfg.backend not in ("openai", "faster-whisper"):
        raise ValueError(f"backend must be 'openai' or 'faster-whisper', got {cfg.backend!r}")
    if cfg.model_name not in models:
        raise ValueError(f"unknown model {cfg.model_name!r}; known: {sorted(models)}")


def _resolve_model(
    cfg: RuntimeConfig, models: Dict, translate_fallback: Optional[str], weights_dir: str
) -> str:
    """The model to actually load, upgrading weights that cannot translate.

    large-v3-turbo accepts task="translate" and returns untranslated
    source-language text, so a turbo translation request is substituted rather
    than served: silently-wrong output is worse than a slower model.
    """
    if not (cfg.task == "translate" and cfg.translator == "whisper"):
        return cfg.model_name
    if models[cfg.model_name].get("translate"):
        return cfg.model_name

    if not translate_fallback:
        raise ValueError(
            f"{cfg.model_name} is not trained for translation and no translate_fallback "
            f"is configured. Set one in config.yml, pick a model with translate: true, "
            f"or use translator='llm'."
        )
    if translate_fallback not in models or not models[translate_fallback].get("translate"):
        raise ValueError(
            f"translate_fallback {translate_fallback!r} is not a known model with translate: true"
        )

    logger.warning(
        f"{cfg.model_name} was not trained for translation and would return "
        f"source-language text; using {translate_fallback} instead. Expect "
        f"substantially slower inference than {cfg.model_name}."
    )

    # An unstaged substitute means a multi-GB pull at startup, which fails outright
    # in an offline or read-only-cache deployment. Say so before it happens.
    weights = resolve_weights(translate_fallback, cfg.backend, models, weights_dir)
    if not weights.staged:
        logger.error(
            f"{translate_fallback} weights are not staged in {weights_dir}; they will be "
            f"downloaded at load time. Bake {translate_fallback} into images that serve "
            f"translation (see build.sh WEIGHTS)."
        )
    return translate_fallback


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
