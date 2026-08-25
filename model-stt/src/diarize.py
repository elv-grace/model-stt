"""Speaker diarization over the transcript's timeline.

Diarization answers "who spoke when", which is a different question from "what
was said" and is answered by a different model. It runs beside the decoder, never
in front of it: whisper transcribes the whole file, and this module only attaches
a label to what came back. **No word is ever dropped for lacking a speaker.** A
word the diarizer cannot place gets `speaker: None`, which is information, not
a reason to delete it.

The engine is speakrs (https://github.com/avencera/speakrs), a Rust
implementation of pyannote's community-1 pipeline. It has no Python bindings, so
diarize/ wraps it in a small binary and this module drives that binary: ffmpeg
decodes the file to 16 kHz mono PCM, the PCM goes over a pipe, and one JSON
object comes back. See diarize/src/main.rs for the protocol.

Two things about the output are worth knowing before reading further.

Labels are per-run. speakrs clusters each call on its own, so its SPEAKER_00 in
one part of an asset is unrelated to SPEAKER_00 in the next -- and the tagger
runtime feeds parts, not whole assets. The binary therefore also returns a mean
embedding per speaker, and this module keeps a registry of them across calls,
renaming each part's local labels to asset-wide ones. `reset()` drops the
registry, and must be called between unrelated assets for the same reason
WhisperSTT.reset_context() must.

Labels are not always trustworthy, and the untrustworthy places are knowable.
The segmentation model sees ten seconds at a time and can represent at most
three speakers in that window, at most two talking at once -- pyannote's powerset
shape, fixed by the weights. That bounds voices per window, not per recording:
windows step every one or two seconds and clustering stitches them together, so
a twelve-person film is fine as long as any given ten seconds holds three or
fewer voices. Where it breaks is sustained crosstalk -- a dinner table, a crowd,
a panel -- and it breaks by *mislabelling*, since the losing voice's frames go to
whichever speaker scored highest. That is invisible in a label, so every caption
carries `speaker_overlap`: how much of its span had more than one speaker
talking. High overlap is where a label deserves doubt.

There is also no music/speech distinction anywhere in this pipeline. Sung vocals
usually read as speech and a singer who sings enough gets their own label, which
no field distinguishes from a character's. Instrumental score is the easier case:
whisper's stock phrases over it usually fall outside every speaker turn, so they
come back with no speaker and low `speaker_coverage`.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

# The only rate speakrs' segmentation and embedding models accept.
SAMPLE_RATE = 16_000

# Where the label registry starts counting. Asset-wide labels are handed out in
# order of first appearance, which is not the order any one part uses.
LABEL_FORMAT = "SPEAKER_{:02d}"


@dataclass(frozen=True)
class DiarizationConfig:
    """Settings for the diarization pass. Surfaced in config.yml.

    Off by default: it is a second model on the same GPU, and a caption track
    without speakers is still a caption track.
    """

    enabled: bool = False

    # Built by diarize/build.sh. Relative paths resolve against the repo root,
    # so the default works from a checkout and the container overrides it.
    binary: str = "diarize/target/release/stt-diarize"

    # speakrs' model bundle, staged by download_weights.py. Relative paths
    # resolve against weights_dir. Leaving it unset makes the binary download
    # from HuggingFace at run time, which is both a network dependency at
    # container start and, in speakrs 0.5.0, broken for CUDA -- see
    # "staging the bundle" in README.md.
    models_dir: Optional[str] = "speakrs"

    # libonnxruntime.so, staged by diarize/build.sh. Unset falls back to
    # $ORT_DYLIB_PATH and then to the binary's own search path.
    ort_lib: Optional[str] = None

    # cuda | cpu. See the DISABLED note in diarize/src/main.rs for why upstream's
    # cuda-fast is not offered.
    mode: str = "cuda"

    # Bridge same-speaker turns separated by less than this. 0 keeps speakrs'
    # own segmentation, which already merges what its clustering thinks is one
    # turn.
    merge_gap: float = 0.0

    # Drop turns shorter than this before assignment. Making activations
    # exclusive can leave one-frame slivers where two speakers trade the top
    # score mid-word; those are not turns and should not win a word. Measured:
    # turns under 0.15s are 10.4% of turns but 0.3% of speech time. See
    # config.yml for the full table.
    min_duration: float = 0.15

    # Dendrogram merge distance for speakrs' clustering; None keeps its 0.6.
    # Lower splits more readily, and it is the only granularity lever exposed.
    # It does NOT fix short-response under-segmentation: 0.6 down to 0.2 all
    # merge the same two voices. See "what it gets wrong" in README.md.
    ahc_threshold: Optional[float] = None

    # Cosine similarity at which a part's speaker is judged to be one already
    # seen in an earlier part. The same-person and different-person
    # distributions overlap, so this trades identities recovered against
    # speakers wrongly merged rather than separating them; 0.65 recovers ~60% of
    # cross-part identities at ~1% wrong merges. The measurement and the full
    # sweep are in config.yml. Embeddings are wespeaker-voxceleb-resnet34,
    # L2-normalised.
    match_threshold: float = 0.65

    # Seconds before the diarizer is killed and the file goes out unlabelled.
    # None waits forever.
    timeout: Optional[float] = 1800.0

    # Retries for a failed run. speakrs' concurrent CUDA path aborts on a heap
    # corruption roughly once in 40 runs, with no pattern in the input. Native
    # memory corruption in code this repo does not own is also why this runs as
    # a subprocess; a retry turns a rare crash into a rare slowdown.
    retries: int = 1

    # false => a missing binary or a failed run logs a warning and the tagger
    # emits its usual tracks without speakers. true => the tagger fails instead.
    required: bool = False


@dataclass(frozen=True)
class SpeakerTurn:
    start: float  # seconds, relative to the start of the file
    end: float
    speaker: str


class Diarization:
    """One file's answer to "who spoke when", queryable by span.

    `turns` is exclusive: sorted, non-overlapping, one speaker at a time. That
    is what makes a word's speaker a single answer. `speech` and `overlapped`
    are derived from the raw turns, before that collapse, and are what the
    coverage and overlap questions are answered from.
    """

    def __init__(
        self,
        turns: Sequence[SpeakerTurn],
        raw_turns: Sequence[SpeakerTurn],
        duration: float,
    ):
        self.turns: List[SpeakerTurn] = list(turns)
        self.duration = duration
        self._starts = [t.start for t in self.turns]
        self._speech = _union(raw_turns or turns)
        self._overlapped = _overlapped_regions(raw_turns)

    @property
    def speakers(self) -> List[str]:
        return sorted({t.speaker for t in self.turns})

    def speaker_at(self, start: float, end: float) -> Optional[str]:
        """The speaker holding most of this span, or None if no turn touches it.

        Most, not first: a word straddling a turn boundary belongs to whoever
        was talking for more of it.
        """
        if not self.turns:
            return None

        # A collapsed word -- whisper emits zero-duration words, and _filter_segments
        # only repairs whole segments -- has no span to weigh, so ask which turn
        # contains the instant instead.
        if end <= start:
            index = bisect_right(self._starts, start) - 1
            if index < 0:
                return None
            turn = self.turns[index]
            return turn.speaker if turn.end >= start else None

        best: Optional[str] = None
        best_overlap = 0.0
        # turns are sorted and disjoint, so ends rise with starts: once a turn
        # ends before this span begins, every earlier one does too
        for turn in reversed(self.turns[: bisect_right(self._starts, end)]):
            if turn.end <= start:
                break
            overlap = min(turn.end, end) - max(turn.start, start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
        return best

    def coverage(self, start: float, end: float) -> float:
        """Fraction of this span that anyone was speaking over.

        Low coverage on a caption means whisper produced words where the
        diarizer heard no speech -- score, effects, silence. It is the closest
        thing here to a hallucination signal.
        """
        return _covered_fraction(self._speech, start, end)

    def overlap(self, start: float, end: float) -> float:
        """Fraction of this span with more than one speaker talking.

        The exclusive timeline had to pick one of them. This says how much of
        the span that choice was made over.
        """
        return _covered_fraction(self._overlapped, start, end)


class SpeakerDiarizer:
    """Runs the diarization binary and keeps labels stable across an asset."""

    def __init__(self, cfg: DiarizationConfig, weights_dir: Optional[str] = None):
        self.cfg = cfg
        self.binary = _resolve(cfg.binary, base=_repo_root())
        if not os.path.isfile(self.binary):
            raise FileNotFoundError(
                f"diarization binary not found at {self.binary}; build it with "
                "diarize/build.sh"
            )

        self.models_dir: Optional[str] = None
        if cfg.models_dir:
            self.models_dir = _resolve(cfg.models_dir, base=weights_dir or _repo_root())
            if not os.path.isdir(self.models_dir):
                raise FileNotFoundError(
                    f"speakrs models not found at {self.models_dir}; stage them with "
                    "`python download_weights.py --diarization`"
                )

        self.ort_lib = _resolve(cfg.ort_lib, base=_repo_root()) if cfg.ort_lib else None
        self._env = self._build_env()

        # asset-wide label -> unit-norm centroid, and the seconds of speech that
        # centroid was averaged over
        self._centroids: Dict[str, List[float]] = {}
        self._weights: Dict[str, float] = {}
        # labels are handed out in order of first appearance and never reused
        # within an asset, including to speakers that never got a centroid
        self._next_label = 0

    def reset(self) -> None:
        """Forget every speaker seen so far. Call between unrelated assets."""
        self._centroids.clear()
        self._weights.clear()
        self._next_label = 0

    def diarize(self, fpath: str) -> Optional[Diarization]:
        """Diarize one file, or return None if the attempt failed.

        Returning None rather than raising is the point: the transcript is
        already good without speakers, and a diarizer that takes the file down
        with it is worse than no diarizer.
        """
        try:
            payload = self._run_with_retries(fpath)
        except Exception as exc:  # noqa: BLE001 - every failure degrades the same way
            if self.cfg.required:
                raise
            logger.warning(
                f"{fpath}: diarization failed ({type(exc).__name__}: {exc}); "
                "tags will carry no speaker"
            )
            return None

        raw = [_turn(t) for t in payload.get("turns", [])]
        local = [_turn(t) for t in payload.get("segments", [])]
        if not local:
            logger.info(f"{fpath}: diarization found no speech")
            return Diarization([], [], payload.get("duration", 0.0))

        mapping = self._globalize(local, payload.get("embeddings", {}))
        turns = [SpeakerTurn(t.start, t.end, mapping[t.speaker]) for t in local]
        raw_turns = [
            SpeakerTurn(t.start, t.end, mapping.get(t.speaker, t.speaker)) for t in raw
        ]

        logger.info(
            f"{fpath}: {len(turns)} speaker turns, "
            f"{len({t.speaker for t in turns})} speakers "
            f"({', '.join(sorted({t.speaker for t in turns}))})"
        )
        return Diarization(turns, raw_turns, payload.get("duration", 0.0))

    def _run_with_retries(self, fpath: str) -> dict:
        """Re-run a crashed diarization, but never a timed-out one.

        The crash worth retrying is speakrs' intermittent heap corruption on the
        concurrent CUDA path, which the same audio survives on the next attempt.
        A timeout is not that: it means the work did not fit in the budget, and
        doing it again will not make it fit.
        """
        last: Exception = RuntimeError("no attempt made")
        for attempt in range(max(self.cfg.retries, 0) + 1):
            try:
                return self._run(fpath)
            except subprocess.TimeoutExpired:
                raise
            except Exception as exc:  # noqa: BLE001 - retried the same way regardless
                last = exc
                if attempt < self.cfg.retries:
                    logger.warning(
                        f"{fpath}: diarization attempt {attempt + 1} failed "
                        f"({type(exc).__name__}: {exc}); retrying"
                    )
        raise last

    def _run(self, fpath: str) -> dict:
        """Decode to PCM and pipe it through the diarizer, streaming both ways.

        ffmpeg writes straight into the binary rather than through this process:
        an hour of 16 kHz mono is 115 MB, and there is no reason for it to pass
        through Python. `-map 0:a:0` picks the same audio stream faster-whisper
        decodes, so the two models cannot end up on different tracks of a
        multi-track file.
        """
        # ffmpeg's diagnostics go to a file, not a pipe: nothing reads that pipe
        # until after wait(), and a file ffmpeg has plenty to complain about
        # would fill it and deadlock.
        with tempfile.TemporaryFile() as ffmpeg_errors:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-nostdin", "-v", "error",
                    "-i", fpath,
                    "-vn", "-map", "0:a:0",
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-f", "s16le", "-",
                ],
                stdout=subprocess.PIPE,
                stderr=ffmpeg_errors,
            )
            try:
                proc = subprocess.Popen(
                    self._command(),
                    stdin=ffmpeg.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._env,
                )
            except Exception:
                ffmpeg.kill()
                ffmpeg.wait()
                raise

            # this process must not hold the read end, or ffmpeg never sees the
            # pipe close when the diarizer exits early
            assert ffmpeg.stdout is not None
            ffmpeg.stdout.close()

            try:
                out, err = proc.communicate(timeout=self.cfg.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                ffmpeg.kill()
                proc.communicate()
                raise
            finally:
                ffmpeg.wait()

            ffmpeg_errors.seek(0)
            ffmpeg_detail = ffmpeg_errors.read().decode(errors="replace").strip()

        if proc.returncode != 0:
            detail = err.decode(errors="replace").strip() or f"exit {proc.returncode}"
            raise RuntimeError(f"{detail} (ffmpeg: {ffmpeg_detail})" if ffmpeg_detail else detail)

        # A diarizer that succeeded on a truncated decode is worse than one that
        # failed: the labels look complete and the tail is missing. Note that
        # ffmpeg also exits non-zero on the SIGPIPE it takes when the diarizer
        # dies first, which is why this is checked second.
        if ffmpeg.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited {ffmpeg.returncode}: {ffmpeg_detail or 'no detail'}"
            )

        for line in err.decode(errors="replace").splitlines():
            logger.debug(f"stt-diarize: {line}")
        return json.loads(out)

    def _command(self) -> List[str]:
        command = [self.binary, "--mode", self.cfg.mode]
        if self.models_dir:
            command += ["--models-dir", self.models_dir]
        if self.ort_lib:
            command += ["--ort-lib", self.ort_lib]
        if self.cfg.merge_gap > 0:
            command += ["--merge-gap", str(self.cfg.merge_gap)]
        if self.cfg.min_duration > 0:
            command += ["--min-duration", str(self.cfg.min_duration)]
        if self.cfg.ahc_threshold is not None:
            command += ["--ahc-threshold", str(self.cfg.ahc_threshold)]
        return command

    def _build_env(self) -> Dict[str, str]:
        """The loader path the CUDA execution provider needs.

        ONNX Runtime's CUDA provider dlopens libcublasLt.so.12 and libcudnn.so.9
        the same way CTranslate2 does, so the container's LD_LIBRARY_PATH already
        covers it. The pip nvidia wheels are added here anyway so a checkout
        works without the caller exporting anything.
        """
        env = dict(os.environ)
        paths = _nvidia_lib_dirs()
        if self.ort_lib:
            env["ORT_DYLIB_PATH"] = self.ort_lib
            paths.append(os.path.dirname(self.ort_lib))
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)
        if paths:
            env["LD_LIBRARY_PATH"] = ":".join(paths)
        return env

    def _globalize(
        self, turns: Sequence[SpeakerTurn], embeddings: Dict[str, Sequence[float]]
    ) -> Dict[str, str]:
        """Map this file's speaker labels onto asset-wide ones.

        Best-first and one-to-one: every (local, known) pair is scored, the
        strongest match above the threshold is taken, and both sides are then
        spent. Greedy rather than optimal because the alternative is a
        Hungarian assignment over at most a handful of speakers, and because a
        wrong greedy match costs one part's labels, not the asset's.

        A local speaker with no centroid -- possible only if every window it was
        assigned had a non-finite embedding -- gets a fresh label and is not
        registered, since there is nothing to match it against later.
        """
        speech = _speech_seconds(turns)
        local = {
            speaker: vector
            for speaker, vector in embeddings.items()
            if speaker in speech and vector
        }

        candidates = sorted(
            (
                (_cosine(vector, self._centroids[known]), speaker, known)
                for speaker, vector in local.items()
                for known in self._centroids
            ),
            reverse=True,
        )

        mapping: Dict[str, str] = {}
        taken: set = set()
        for score, speaker, known in candidates:
            if score < self.cfg.match_threshold:
                break
            if speaker in mapping or known in taken:
                continue
            mapping[speaker] = known
            taken.add(known)
            logger.debug(f"matched {speaker} to {known} (cosine {score:.3f})")

        # loudest first, so the label numbers follow how much a new speaker
        # actually says rather than which turn happened to come first
        for speaker in sorted(speech, key=lambda s: (-speech[s], s)):
            if speaker not in mapping:
                mapping[speaker] = LABEL_FORMAT.format(self._next_label)
                self._next_label += 1

        for speaker, label in mapping.items():
            if speaker in local:
                self._register(label, local[speaker], speech.get(speaker, 0.0))
        return mapping

    def _register(self, label: str, vector: Sequence[float], weight: float) -> None:
        """Fold one part's centroid into the asset-wide one, weighted by speech.

        A speaker heard for two minutes in one part and two seconds in another
        should not have those count equally: the long one is the better estimate
        of the voice.
        """
        weight = max(weight, 1e-6)
        known = self._centroids.get(label)
        if known is None:
            self._centroids[label] = _normalize(list(vector))
            self._weights[label] = weight
            return

        total = self._weights[label] + weight
        blended = [
            (existing * self._weights[label] + new * weight) / total
            for existing, new in zip(known, vector)
        ]
        self._centroids[label] = _normalize(blended)
        self._weights[label] = total


def build_diarizer(
    cfg: DiarizationConfig, weights_dir: Optional[str]
) -> Optional[SpeakerDiarizer]:
    """Load the diarizer, or return None if it is disabled or unavailable."""
    if not cfg.enabled:
        return None
    try:
        return SpeakerDiarizer(cfg, weights_dir=weights_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure degrades the same way
        if cfg.required:
            raise
        logger.warning(
            f"diarization unavailable ({type(exc).__name__}: {exc}); "
            "tags will carry no speaker"
        )
        return None


def _turn(payload: dict) -> SpeakerTurn:
    return SpeakerTurn(
        start=float(payload["start"]),
        end=float(payload["end"]),
        speaker=str(payload["speaker"]),
    )


def _speech_seconds(turns: Sequence[SpeakerTurn]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + max(turn.end - turn.start, 0.0)
    return totals


def _union(turns: Sequence[SpeakerTurn]) -> List[Tuple[float, float]]:
    """Merge every turn into disjoint spans of "someone is talking"."""
    spans = sorted((t.start, t.end) for t in turns if t.end > t.start)
    merged: List[Tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlapped_regions(turns: Sequence[SpeakerTurn]) -> List[Tuple[float, float]]:
    """Spans where two or more speakers are talking at once.

    A sweep over turn edges: depth rises on a start and falls on an end, and a
    region is overlapped wherever depth stayed above one. Ends sort before
    starts at equal time so that a turn ending exactly where another begins is
    a handover, not an overlap.
    """
    events: List[Tuple[float, int]] = []
    for turn in turns:
        if turn.end > turn.start:
            events.append((turn.start, 1))
            events.append((turn.end, -1))
    if not events:
        return []

    events.sort(key=lambda event: (event[0], event[1]))
    regions: List[Tuple[float, float]] = []
    depth = 0
    opened: Optional[float] = None
    for time, delta in events:
        was_overlapped = depth > 1
        depth += delta
        if depth > 1 and not was_overlapped:
            opened = time
        elif depth <= 1 and was_overlapped and opened is not None:
            if time > opened:
                regions.append((opened, time))
            opened = None
    return regions


def _covered_fraction(
    intervals: Sequence[Tuple[float, float]], start: float, end: float
) -> float:
    """How much of [start, end] the (disjoint, sorted) intervals cover."""
    span = end - start
    if span <= 0 or not intervals:
        return 0.0

    starts = [interval[0] for interval in intervals]
    # the interval before the first one starting after `start` may still reach in
    index = max(bisect_right(starts, start) - 1, 0)
    covered = 0.0
    for interval_start, interval_end in intervals[index:]:
        if interval_start >= end:
            break
        covered += max(min(interval_end, end) - max(interval_start, start), 0.0)
    return min(covered / span, 1.0)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Both sides are unit vectors, so the dot product is the cosine."""
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _nvidia_lib_dirs() -> List[str]:
    """Library directories of the pip nvidia wheels, if they are installed."""
    try:
        import nvidia  # noqa: PLC0415 - optional, and only present in some installs
    except ImportError:
        return []
    if not nvidia.__file__:
        return []
    return sorted(glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "*", "lib")))


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str, base: str) -> str:
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(base, path)
