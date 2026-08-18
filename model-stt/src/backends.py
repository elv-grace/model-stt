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
    # Backend-level lever only. model.py always requests word timestamps: every Tag
    # needs a start/end, and the sentence track needs word timings to place its
    # boundaries (without them we fall back to whisper's much coarser segment
    # boundaries). The benchmark flips this to measure what the DTW pass costs.
    word_timestamps: bool = True
    # off with vad_filter below; see RuntimeConfig.condition_on_previous_text
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    # None => greedy. Deliberate, not an oversight: faster-whisper defaults to 5,
    # openai-whisper defaults to None and picks GreedyDecoder (decoding.py:546).
    # Swept with VAD on, beam 1 vs 5 is within noise (en 4.21 vs 4.33, fr 6.82 vs
    # 6.19 WER at n~38), so greedy wins on speed and keeps the two backends
    # byte-identical. Without VAD, beam=5 was worse -- it finds *more* confident
    # hallucinations in non-speech.
    beam_size: Optional[int] = None
    temperature: Sequence[float] = TEMPERATURE_FALLBACK
    # Swap the sampled retry rungs for deterministic ones, making the decoder
    # reproducible. faster-whisper only; see _DeterministicFallback.
    deterministic_fallback: bool = False
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    hallucination_silence_threshold: Optional[float] = 2.0

    # Silero VAD. faster-whisper only; openai-whisper has no equivalent and
    # ignores this. Off here and in RuntimeConfig -- see the reasoning there, and
    # turn it on via the clean-audio profile.
    #
    # Two corrections to what earlier versions of this comment claimed, both worth
    # keeping so they are not re-derived:
    #
    # 1. VAD is NOT conservative. vad_min_silence_ms only controls how long a pause
    #    must run before VAD *closes* an already-open speech region; audio VAD
    #    never opens in the first place is discarded whatever its length. On
    #    spiderman-into-the-spiderverse-10min at threshold 0.5, four stretches
    #    totalling 292s of real dialogue under a loud score never reached the
    #    decoder.
    # 2. The "600s of crowd noise" fixture this was tuned against was NOT crowd
    #    noise. It is NBAallstar 1200-1800s, and roughly the first 340s of it is
    #    the All-Star player introductions, transcribed accurately down to the
    #    spellings of Antetokounmpo and Gilgeous-Alexander. An earlier sweep scored
    #    those as hallucinations, which made VAD look free and VAD-off look
    #    catastrophic. It is neither. Only the last ~230s is genuine non-speech.
    #
    # The sweep below is on spiderman-into-the-spiderverse-10min, whose four
    # dropped stretches are known real dialogue, so "words" is a recall measure:
    #
    #   threshold  pad    kept   words   stretches recovered
    #   0.50       400     184s    495    0 of 4        <- silero defaults
    #   0.25       400     238s    585    2 of 4
    #   0.15      1000     318s    652    3 of 4
    #   0.25      2000     298s    625    4 of 4        <- clean-audio profile
    #   off         --     591s    801    4 of 4        <- default
    #
    # 0.25/2000 is the best VAD-on operating point found: it recovers all four and
    # 26% more words. Going below 0.25 buys little and costs decode stability
    # (7-22 segments needing temperature fallback, against 0 at 0.25).
    vad_filter: bool = False
    vad_threshold: float = 0.25
    vad_min_silence_ms: int = 2000
    # Padding applied either side of every detected speech region. Doubles as the
    # bridge that stops short excisions from splitting an utterance: regions less
    # than 2*vad_speech_pad_ms apart merge, so no cut shorter than 4s survives.
    vad_speech_pad_ms: int = 2000


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
            # say so rather than ignoring it silently: bench uses the DecodeOptions
            # defaults, so an openai run would request VAD, not get it, and be
            # compared against a CT2 run that did -- without anyone noticing.
            # Measured on 600s of crowd noise: openai 19 repeated segments, CT2
            # unfiltered 31, CT2 with VAD 0. openai is the better *unfiltered*
            # decoder here and still loses, because it has no VAD to enable.
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
    """Replaces whisper's *sampled* retry with a deterministic escalation.

    Whisper retries a decode that trips compression_ratio_threshold or
    logprob_threshold, walking up a temperature ladder. Every rung above 0 samples
    (faster-whisper: beam_size=1, sampling_topk=0, num_hypotheses=best_of), and
    that sampling cannot be seeded -- ctranslate2.set_random_seed has no effect on
    it, verified with forced sampling and with the seed set before model
    construction, and Whisper.generate() has no per-call seed. So the shipped
    decoder is not reproducible.

    The retry's value is not the randomness, though: it is the reject-and-retry
    loop around it. Sampling is only how faster-whisper makes a retry come out
    differently. Escalating deterministic parameters instead keeps the loop and
    drops the randomness. This proxy sits in front of the CTranslate2 model and
    swaps each sampled rung for the corresponding entry below.

    Measured on 600s of NBAallstar (repeated segments, lower is better):

        no retry at all (temperature=[0.0])   228   reproducible
        deterministic ladder                   55   reproducible
        shipped sampled ladder                 30   NOT reproducible

    So it recovers most of the suppression and all of the reproducibility. FLEURS
    en is unchanged at 4.33% WER / 2.08% CER, because clean speech rarely trips a
    threshold and so rarely reaches a rung above 0.

    Off by default: the residual gap to the sampled ladder is real, and on real
    media the run-to-run variation the sampled ladder causes is 10-18% WER without
    a matching swing in defect counts. Turn it on when reproducibility is worth
    more than the last of the repetition suppression -- benchmarking, diffing two
    runs, cache keys, QA sign-off.

    This reaches into faster-whisper's call into CTranslate2, so it is coupled to
    that call's keyword arguments and should be re-checked on upgrade. If those
    names change, the proxy stops intervening rather than misbehaving: `enabled`
    only fires when a `sampling_temperature` keyword is actually present.
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
        # It is not free when disabled: generate() takes an extra Python call, and
        # every *other* attribute faster-whisper reads off the CTranslate2 object
        # (encode, align, detect_language, device, is_multilingual) now resolves
        # through __getattr__ rather than directly. Measured at +58ms on a 3.9s
        # decode, +1.5%, with byte-identical output. Cheap enough to leave in
        # place; not zero, as an earlier version of this comment claimed.
        self._fallback = _DeterministicFallback(self.model.model)
        self.model.model = self._fallback

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
