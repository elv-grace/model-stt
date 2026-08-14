"""Whisper inference backends behind one interface.

Two runtimes decode the same large-v3 / large-v3-turbo weights:

openai          the reference implementation (openai-whisper). Requires torch.
                Slowest, but it is the accuracy baseline and the best-documented
                word-timestamp path, so it is what the benchmark compares against.
faster-whisper  CTranslate2. No torch dependency, typically 3-5x faster at the
                same accuracy with a much smaller resident footprint. Production.

Both are normalized to the Transcription/Segment/Word dataclasses below so that
model.py never branches on which runtime produced a result.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

# whisper's own temperature fallback ladder: each successive temperature is tried
# when a decode trips the compression-ratio or logprob threshold
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class Word:
    start: float  # seconds, relative to the start of the decoded audio
    end: float
    word: str
    probability: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: List[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Transcription:
    language: Optional[str]
    segments: List[Segment]


@dataclass(frozen=True)
class DecodeOptions:
    task: str = "transcribe"
    language: Optional[str] = None
    # Backend-level lever only. model.py always requests word timestamps: every Tag
    # needs a start/end, and the sentence track needs word timings to place its
    # boundaries (without them we fall back to whisper's much coarser segment
    # boundaries). The benchmark flips this to measure what the DTW pass costs.
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    beam_size: Optional[int] = None
    temperature: Sequence[float] = TEMPERATURE_FALLBACK
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    hallucination_silence_threshold: Optional[float] = 2.0


@dataclass(frozen=True)
class WeightRef:
    """Where a backend should load weights from.

    The two runtimes take different things, so this cannot collapse to one string:

    openai-whisper  load_model() only attaches DTW alignment heads when handed a
                    *registered model name*; handed a file path it sets
                    alignment_heads=None and word timestamps degrade. So `ref` is
                    the model name and the staging directory travels separately in
                    `download_root`, where _download() sha256-checks the pre-staged
                    file and skips the network.
    faster-whisper  CTranslate2 conversions carry their own alignment heads, so
                    there is no name requirement -- but bare size names resolve
                    through a repo mapping that changes between library versions.
                    `ref` is therefore always an explicit local directory or an
                    explicit HuggingFace repo id, never a size name.
    """
    ref: str
    download_root: Optional[str] = None
    staged: bool = False  # True when weights were found on disk, i.e. no network needed


class WhisperBackend(ABC):
    """A loaded set of whisper weights that can transcribe a media file."""

    name: str

    @abstractmethod
    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        pass


class OpenAIWhisperBackend(WhisperBackend):
    name = "openai"

    def __init__(self, weights: WeightRef, device: Optional[str] = None):
        # imported lazily so an image built without torch can still use faster-whisper
        import torch
        import whisper

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("cuda not available, running whisper on cpu")

        logger.info(
            f"loading openai-whisper {weights.ref} from {weights.download_root} on {device}"
        )
        self.model = whisper.load_model(
            weights.ref, device=device, download_root=weights.download_root
        )
        self.device = device

    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        kwargs: Dict[str, Any] = dict(
            task=opts.task,
            language=opts.language,
            word_timestamps=opts.word_timestamps,
            condition_on_previous_text=opts.condition_on_previous_text,
            initial_prompt=opts.initial_prompt,
            temperature=tuple(opts.temperature),
            compression_ratio_threshold=opts.compression_ratio_threshold,
            logprob_threshold=opts.logprob_threshold,
            no_speech_threshold=opts.no_speech_threshold,
            fp16=self.device != "cpu",
        )
        if opts.word_timestamps:
            # only honoured alongside word timestamps
            kwargs["hallucination_silence_threshold"] = opts.hallucination_silence_threshold
        if opts.beam_size:
            kwargs["beam_size"] = opts.beam_size

        result = self.model.transcribe(fpath, **kwargs)

        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                avg_logprob=s["avg_logprob"],
                no_speech_prob=s["no_speech_prob"],
                words=[
                    Word(
                        start=w["start"],
                        end=w["end"],
                        word=w["word"],
                        probability=w["probability"],
                    )
                    for w in s.get("words", [])
                ],
            )
            for s in result["segments"]
        ]
        return Transcription(language=result.get("language"), segments=segments)


class FasterWhisperBackend(WhisperBackend):
    name = "faster-whisper"

    def __init__(
        self,
        weights: WeightRef,
        device: Optional[str] = None,
        compute_type: str = "float16",
        cpu_threads: int = 0,
    ):
        from faster_whisper import WhisperModel

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                # torch is not a faster-whisper dependency; assume the GPU the
                # container was given and let ctranslate2 raise if it is absent
                device = "cuda"
        if device == "cpu" and compute_type == "float16":
            logger.warning("float16 is not supported on cpu, falling back to int8")
            compute_type = "int8"

        logger.info(f"loading faster-whisper {weights.ref} on {device} ({compute_type})")
        self.model = WhisperModel(
            weights.ref,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        try:
            return self._run(fpath, opts)
        except IndexError as e:
            # faster-whisper's find_alignment() raises IndexError when the DTW pass
            # gets an empty frame array, reproducible on sung audio at full 30s
            # length. Degrade to segment-level timings rather than lose the file:
            # to_sentences() falls back to segment boundaries when words are absent.
            if not opts.word_timestamps:
                raise
            logger.warning(
                f"word alignment failed on {fpath} ({e}); "
                f"re-decoding without word timestamps (segment-level timings only)"
            )
            return self._run(fpath, replace(opts, word_timestamps=False))

    def _run(self, fpath: str, opts: DecodeOptions) -> Transcription:
        segment_iter, info = self.model.transcribe(
            fpath,
            task=opts.task,
            language=opts.language,
            word_timestamps=opts.word_timestamps,
            condition_on_previous_text=opts.condition_on_previous_text,
            initial_prompt=opts.initial_prompt,
            # faster-whisper has no "greedy" sentinel; beam_size=1 is greedy
            beam_size=opts.beam_size or 1,
            temperature=list(opts.temperature),
            compression_ratio_threshold=opts.compression_ratio_threshold,
            # note the name differs from openai-whisper's logprob_threshold
            log_prob_threshold=opts.logprob_threshold,
            no_speech_threshold=opts.no_speech_threshold,
            hallucination_silence_threshold=(
                opts.hallucination_silence_threshold if opts.word_timestamps else None
            ),
        )

        # transcribe() returns a generator; nothing is decoded until it is drained
        segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=[
                    Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                    for w in (s.words or [])
                ],
            )
            for s in segment_iter
        ]
        return Transcription(language=info.language, segments=segments)


def resolve_weights(
    model_name: str, backend: str, models: Dict, weights_dir: str
) -> WeightRef:
    """Locate staged weights for (model_name, backend), else fall back to a remote ref."""
    entry = models.get(model_name)
    if entry is None:
        raise ValueError(f"unknown model {model_name!r}; known: {sorted(models)}")

    if backend == "openai":
        root = os.path.join(weights_dir, "openai")
        staged = os.path.isfile(os.path.join(root, entry["openai"]))
        if not staged:
            logger.warning(
                f"{entry['openai']} not staged in {root}; whisper will download "
                f"{model_name} at load time"
            )
        # the NAME, not the path -- see WeightRef
        return WeightRef(ref=model_name, download_root=root, staged=staged)

    if backend == "faster-whisper":
        local = os.path.join(weights_dir, "faster-whisper", entry["ct2"])
        if os.path.isdir(local):
            return WeightRef(ref=local, staged=True)
        logger.warning(
            f"{local} not staged; faster-whisper will pull {entry['ct2_repo']} from HuggingFace"
        )
        return WeightRef(ref=entry["ct2_repo"], staged=False)

    raise ValueError(f"unknown backend {backend!r}; expected 'openai' or 'faster-whisper'")


def build_backend(
    backend: str,
    model_name: str,
    models: Dict,
    weights_dir: str,
    device: Optional[str] = None,
    compute_type: str = "float16",
) -> WhisperBackend:
    weights = resolve_weights(model_name, backend, models, weights_dir)
    if backend == "openai":
        return OpenAIWhisperBackend(weights, device=device)
    return FasterWhisperBackend(weights, device=device, compute_type=compute_type)
