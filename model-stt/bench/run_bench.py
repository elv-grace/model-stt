"""Speed benchmark for the model-stt configurations (openai-whisper vs. faster-whisper (ct2), large-v3 vs. large-v3-turbo).

Runs one or more (backend, model) combinations over the same files and reports
real-time factor, so the two runtimes and the two model sizes are directly
comparable. Model load is timed separately from inference: load happens once per
container start, inference happens per segment, and averaging them together
flatters short runs.

    python -m bench.run_bench --files test-files/bench-files/*.m4a
    python -m bench.run_bench --files ... --systems turbo-ct2 turbo-openai

Output: bench-output/<system>.jsonl (tags) and bench-output/summary.json.

Can also score model-asr and model-multilingual-stt .jsonl output with bench/score.py, which reads the
same format.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from loguru import logger

from config import config
from src.model import RuntimeConfig, WhisperSTT
from src.punctuate import PunctuationConfig
# DISABLED (translation): from src.translate import TranslatorConfig

# the combinations worth comparing; --systems selects a subset
SYSTEMS: Dict[str, RuntimeConfig] = {
    # path A, both runtimes: the speed comparison that motivates faster-whisper
    "turbo-openai": RuntimeConfig(backend="openai", model_name="large-v3-turbo"),
    "turbo-ct2": RuntimeConfig(backend="faster-whisper", model_name="large-v3-turbo"),
    # path A on the undistilled weights: what turbo's 4-layer decoder costs in
    # transcription accuracy, as opposed to translation (which it cannot do)
    "large-v3-openai": RuntimeConfig(backend="openai", model_name="large-v3"),
    "large-v3-ct2": RuntimeConfig(backend="faster-whisper", model_name="large-v3"),
    # DISABLED (translation): path B (whisper native translate, needs large-v3)
    # and path C (turbo + LLM). RuntimeConfig no longer has task/translator, so
    # these cannot be constructed; restore them together with those fields.
    # "translate-openai": RuntimeConfig(
    #     backend="openai", model_name="large-v3", task="translate", translator="whisper"
    # ),
    # "translate-ct2": RuntimeConfig(
    #     backend="faster-whisper", model_name="large-v3", task="translate", translator="whisper"
    # ),
    # "translate-llm": RuntimeConfig(
    #     backend="faster-whisper", model_name="large-v3-turbo",
    #     task="translate", translator="llm",
    # ),
}


@dataclass
class FileResult:
    file: str
    audio_seconds: float
    wall_seconds: float
    rtf: float  # audio_seconds / wall_seconds; higher is faster than real time
    n_tags: int


@dataclass
class SystemResult:
    system: str
    backend: str
    requested_model: str
    effective_model: str
    task: str
    load_seconds: float
    total_audio_seconds: float
    total_wall_seconds: float
    rtf: float
    peak_gpu_mib: Optional[int]
    files: List[FileResult]


def audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def gpu_memory_mib() -> Optional[int]:
    """GPU memory attributed to this process, across both backends.

    torch.cuda.max_memory_allocated only sees torch allocations, so it reports
    nothing for the CTranslate2 backend. nvidia-smi sees both.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    pid = str(os.getpid())
    for line in out.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0] == pid:
            return int(fields[1])
    return None


def run_system(
    name: str,
    cfg: RuntimeConfig,
    files: List[str],
    outdir: str,
) -> SystemResult:
    logger.info(f"=== {name}: {cfg.backend} / {cfg.model_name}")

    load_start = time.perf_counter()
    model = WhisperSTT(
        cfg,
        models=config["models"],
        weights_dir=config["storage"]["weights_dir"],
        sentence_gap_ms=config["postprocessing"]["sentence_gap"],
        max_caption_words=config["postprocessing"]["max_caption_words"],
        # Punctuation off: the bench measures the decoder, and FLEURS is
        # scored through EnglishTextNormalizer, which strips punctuation before
        # comparing.
        punctuation=PunctuationConfig(enabled=False),
    )
    load_seconds = time.perf_counter() - load_start

    results: List[FileResult] = []
    peak_gpu = None
    out_path = os.path.join(outdir, f"{name}.jsonl")

    with open(out_path, "w") as fout:
        for path in files:
            # benchmark fixtures are unrelated assets, not contiguous segments of
            # one; without this a previous file's language leaks into the next
            model.reset_context()
            duration = audio_duration(path)
            start = time.perf_counter()
            tags = model.tag(path)
            wall = time.perf_counter() - start

            for tag in tags:
                fout.write(json.dumps({"type": "tag", "data": asdict(tag)}) + "\n")

            used = gpu_memory_mib()
            if used is not None:
                peak_gpu = max(peak_gpu or 0, used)

            results.append(FileResult(
                file=path,
                audio_seconds=round(duration, 3),
                wall_seconds=round(wall, 3),
                rtf=round(duration / wall, 2) if wall else 0.0,
                n_tags=len(tags),
            ))
            logger.info(f"  {os.path.basename(path)}: {wall:.2f}s for {duration:.1f}s audio "
                        f"({results[-1].rtf:.1f}x realtime, {len(tags)} tags)")

    total_audio = sum(r.audio_seconds for r in results)
    total_wall = sum(r.wall_seconds for r in results)

    return SystemResult(
        system=name,
        backend=cfg.backend,
        requested_model=cfg.model_name,
        effective_model=model.effective_model_name,
        task="transcribe",
        load_seconds=round(load_seconds, 2),
        total_audio_seconds=round(total_audio, 2),
        total_wall_seconds=round(total_wall, 2),
        rtf=round(total_audio / total_wall, 2) if total_wall else 0.0,
        peak_gpu_mib=peak_gpu,
        files=results,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--systems', nargs='+', default=["turbo-openai", "turbo-ct2"],
                        choices=sorted(SYSTEMS))
    parser.add_argument('--outdir', default='bench-output')
    # DISABLED (translation): --llm-host / --llm-model configured path C
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summaries = []
    for name in args.systems:
        try:
            summaries.append(run_system(name, SYSTEMS[name], args.files, args.outdir))
        except Exception as e:
            # one misconfigured system should not lose the whole run
            logger.opt(exception=e).error(f"{name} failed")

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)

    print(f"\n{'system':<18} {'model':<16} {'load':>7} {'audio':>9} {'wall':>9} {'RTF':>8} {'GPU MiB':>9}")
    for s in summaries:
        print(f"{s.system:<18} {s.effective_model:<16} {s.load_seconds:>6.1f}s "
              f"{s.total_audio_seconds:>8.1f}s {s.total_wall_seconds:>8.1f}s "
              f"{s.rtf:>7.1f}x {str(s.peak_gpu_mib or '-'):>9}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
