"""Whisper inference backends behind one interface.

Two runtimes decode the same large-v3 / large-v3-turbo weights:

openai          the reference implementation (openai-whisper). Requires torch.
                Slowest, but it is the accuracy baseline and the best-documented
                word-timestamp path, so it is what the benchmark compares against.
faster-whisper  CTranslate2. Typically 3-5x faster at the same accuracy with a
                much smaller resident footprint. Production (containerized).
                The library needs no torch -- though the image now ships it
                anyway for src/punctuate.py, so that is no longer what keeps the
                openai backend out; speed, GPU footprint, VAD and the hookable
                fallback are.

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
# creates stochastic output from sampling randomness, so default shipped is a deterministic ladder instead (see _DeterministicFallback)
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
    # whisper's two repetition signals, carried through because they are the
    # principled way to spot a hallucinated segment: compression_ratio above
    # compression_ratio_threshold means the text is repetitive, and temperature
    # above 0 means the decode fell back at least once before settling.
    compression_ratio: float = 0.0
    temperature: float = 0.0
    words: List[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Transcription:
    language: Optional[str]
    segments: List[Segment]


@dataclass(frozen=True)
class DecodeOptions:
    task: str = "transcribe"
    language: Optional[str] = None
    # Backend-level lever only. model.py always requests word timestamps.
    # The benchmark flips this to measure what the DTW pass costs.
    word_timestamps: bool = True
    # off with vad_filter below; see RuntimeConfig.condition_on_previous_text
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    # faster-whisper defaults to 5, openai-whisper defaults to None and picks GreedyDecoder.
    # With VAD on, beam 1 vs 5 is within noise, so greedy wins on speed and keeps the two backends byte-identical.
    # Without VAD, beam=5 was worse (finds more confident hallucinations in non-speech).
    beam_size: Optional[int] = None # greedy
    temperature: Sequence[float] = TEMPERATURE_FALLBACK
    # Swap the sampled retry rungs for deterministic ones, making the decoder
    # reproducible. faster-whisper only; see _DeterministicFallback.
    deterministic_fallback: bool = False
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    hallucination_silence_threshold: Optional[float] = 2.0
    # Silero VAD. faster-whisper only.
    # Off here and in RuntimeConfig (default, see reasoning there).
    # Turned on in the clean-audio profile.
    # 0.25/2000 is the best VAD-on operating point found.
    vad_filter: bool = False
    vad_threshold: float = 0.25 # controls how loud a pause must be before VAD opens a speech region (decode stability)
    vad_min_silence_ms: int = 2000 # controls how long a pause must run before VAD closes an already-open speech region
    # Padding applied either side of every detected speech region. Doubles as the
    # bridge that stops short excisions from splitting an utterance: regions less
    # than 2*vad_speech_pad_ms apart merge, so no cut shorter than 4s survives.
    vad_speech_pad_ms: int = 2000


@dataclass(frozen=True)
class WeightRef:
    """Where a backend should load weights from.

    Cannot collapse to one string, because the runtimes want different forms:
    openai-whisper needs a *registered model name* (a path sets
    alignment_heads=None and degrades word timestamps), so the directory travels
    separately in `download_root`; faster-whisper needs an explicit directory or
    repo id, never a size name, since that mapping shifts between versions.
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
        # transcribe() runs once per media file, so the vad_filter warning below
        # is emitted once per model rather than once per file
        self._warned_vad = False
        self._warned_deterministic = False

    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        if opts.deterministic_fallback and not self._warned_deterministic:
            logger.warning(
                "openai-whisper has no deterministic-fallback path; the sampled "
                "temperature ladder still applies and output stays irreproducible."
            )
            self._warned_deterministic = True

        if opts.vad_filter and not self._warned_vad:
            logger.warning(
                "openai-whisper has no VAD; vad_filter is ignored. It falls back to "
                "the decoder's own no_speech_prob, which does not suppress "
                "hallucinations over music or crowd noise."
            )
            self._warned_vad = True

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
                compression_ratio=s.get("compression_ratio", 0.0),
                temperature=s.get("temperature", 0.0) or 0.0,
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


class _DeterministicFallback:
    """Replaces whisper's *sampled* retry rungs with a deterministic escalation.

    The retry's value is the reject-and-retry loop, not the randomness -- sampling
    is only how faster-whisper makes a retry come out differently. This proxy sits
    in front of the CTranslate2 model and swaps each sampled rung for the
    corresponding entry below, keeping the loop and dropping the randomness.

    It is coupled to faster-whisper's call into CTranslate2 and should be
    re-checked on upgrade. If those keyword names change it stops intervening
    rather than misbehaving, since it only fires when `sampling_temperature` is
    present. Measurements in README.
    """

    # index i replaces temperature ladder rung i. Rung 0 is never reached (a
    # temperature of 0 does not take faster-whisper's sampling branch) and is
    # present only to keep the indices aligned with the temperature list.
    RUNGS = (
        {"beam_size": 1},
        {"beam_size": 5, "patience": 1.0},
        {"beam_size": 5, "patience": 1.0, "repetition_penalty": 1.15},
        {"beam_size": 5, "patience": 1.0, "repetition_penalty": 1.35},
        {"beam_size": 5, "patience": 1.0, "repetition_penalty": 1.35,
         "no_repeat_ngram_size": 4},
        {"beam_size": 8, "patience": 2.0, "repetition_penalty": 1.6,
         "no_repeat_ngram_size": 3},
    )

    def __init__(self, inner):
        self._inner = inner
        self.enabled = False
        self.temperatures: List[float] = []

    def __getattr__(self, name):
        # only reached for names not on the proxy itself
        return getattr(self._inner, name)

    def generate(self, *args, **kwargs):
        temperature = kwargs.get("sampling_temperature")
        if self.enabled and temperature:
            try:
                rung = self.temperatures.index(temperature)
            except ValueError:
                rung = len(self.RUNGS) - 1
            kwargs.pop("sampling_temperature", None)
            kwargs.pop("sampling_topk", None)
            kwargs.pop("num_hypotheses", None)
            # these two arrive from the caller's options; the rung overrides them
            kwargs.update(self.RUNGS[min(rung, len(self.RUNGS) - 1)])
        return self._inner.generate(*args, **kwargs)


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
        # Installed unconditionally and armed per decode from DecodeOptions, so
        # the flag can vary call to call without rebuilding the model.
        #
        # generate() takes an extra Python call, and
        # every other attribute faster-whisper reads off the CTranslate2 object
        # (encode, align, detect_language, device, is_multilingual) now resolves
        # through __getattr__ rather than directly.
        self._fallback = _DeterministicFallback(self.model.model)
        self.model.model = self._fallback

    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        try:
            return self._run(fpath, opts)
        except IndexError as e:
            # faster-whisper's find_alignment() raises IndexError when the DTW pass
            # gets an empty frame array.
            # Degrade to segment-level timings rather than lose the file:
            # to_sentences() falls back to segment boundaries when words are absent.
            if not opts.word_timestamps:
                raise
            logger.warning(
                f"word alignment failed on {fpath} ({e}); "
                f"re-decoding without word timestamps (segment-level timings only)"
            )
            return self._run(fpath, replace(opts, word_timestamps=False))

    def _run(self, fpath: str, opts: DecodeOptions) -> Transcription:
        self._fallback.enabled = opts.deterministic_fallback
        self._fallback.temperatures = list(opts.temperature)
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
            vad_filter=opts.vad_filter,
            vad_parameters=(
                dict(
                    threshold=opts.vad_threshold,
                    min_silence_duration_ms=opts.vad_min_silence_ms,
                    speech_pad_ms=opts.vad_speech_pad_ms,
                )
                if opts.vad_filter else None
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
                compression_ratio=s.compression_ratio,
                temperature=getattr(s, "temperature", 0.0) or 0.0,
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
