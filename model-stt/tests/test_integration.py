"""End-to-end tests against real weights.

Skipped unless the weights are staged locally:

    python download_weights.py --models large-v3-turbo
    pytest -m weights tests/test_integration.py
"""
from __future__ import annotations

import glob
import os

import pytest

from config import config
from src.backends import resolve_weights
from src.model import SENTENCE_TRACK, WORD_TRACK, RuntimeConfig, WhisperSTT, has_audio_stream

pytestmark = [pytest.mark.weights, pytest.mark.gpu]

WEIGHTS_DIR = config["storage"]["weights_dir"]
MODELS = config["models"]

# any short speech file; model-asr's fixtures are 30s AAC segments
_HERE = os.path.dirname(__file__)
CANDIDATES = sorted(
    glob.glob(os.path.join(_HERE, "..", "test-files", "bench-files", "*.m4a"))
    + glob.glob(os.path.join(_HERE, "..", "test-files", "*.m4a"))
    + glob.glob(os.path.join(_HERE, "..", "..", "model-asr", "test-files", "*.m4a"))
)


@pytest.fixture(scope="module")
def media() -> str:
    if not CANDIDATES:
        pytest.skip("no test media found under test-files/")
    return CANDIDATES[0]


def requires_staged(model_name: str, backend: str) -> None:
    if not resolve_weights(model_name, backend, MODELS, WEIGHTS_DIR).staged:
        pytest.skip(f"{model_name} ({backend}) not staged in {WEIGHTS_DIR}")


@pytest.mark.parametrize("backend", ["openai", "faster-whisper"])
def test_transcribes_with_either_backend(backend, media):
    requires_staged("large-v3-turbo", backend)

    model = WhisperSTT(
        RuntimeConfig(backend=backend, model_name="large-v3-turbo"),
        models=MODELS,
        weights_dir=WEIGHTS_DIR,
    )
    tags = model.tag(media)

    assert tags, "expected speech to be transcribed"
    words = [t for t in tags if t.track == WORD_TRACK]
    sentences = [t for t in tags if t.track == SENTENCE_TRACK]
    assert words and sentences

    for tag in tags:
        assert tag.source_media == media
        assert tag.end_time >= tag.start_time
        assert tag.start_time >= 0
        assert tag.additional_info["language"]

    # sentence text must be recoverable from the word track
    joined = " ".join(t.tag for t in words)
    assert sentences[0].tag.split()[0] in joined


def test_both_backends_agree_on_transcript(media):
    """The two runtimes decode the same weights and should broadly agree.

    A loose check: this guards against a backend being misconfigured (wrong
    weights, wrong task, broken timestamps), not against small decoding
    differences, which are expected.
    """
    for backend in ("openai", "faster-whisper"):
        requires_staged("large-v3-turbo", backend)

    texts = {}
    for backend in ("openai", "faster-whisper"):
        model = WhisperSTT(
            RuntimeConfig(backend=backend, model_name="large-v3-turbo"),
            models=MODELS,
            weights_dir=WEIGHTS_DIR,
            )
        tags = [t for t in model.tag(media) if t.track == WORD_TRACK]
        texts[backend] = [t.tag.lower().strip(".,!?") for t in tags]

    a, b = texts["openai"], texts["faster-whisper"]
    overlap = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
    assert overlap > 0.6, f"backends disagree badly: {overlap:.2f} token overlap"


def test_timestamps_stay_within_the_file(media):
    requires_staged("large-v3-turbo", "faster-whisper")
    import subprocess

    duration_ms = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media],
        capture_output=True, text=True, check=True,
    ).stdout.strip()) * 1000

    model = WhisperSTT(
        RuntimeConfig(backend="faster-whisper", model_name="large-v3-turbo"),
        models=MODELS,
        weights_dir=WEIGHTS_DIR,
    )
    tags = model.tag(media)

    # timestamps are relative to the file handed to tag(), so nothing may exceed
    # its duration (a small tolerance covers whisper's boundary rounding)
    assert max(t.end_time for t in tags) <= duration_ms + 1000


def test_has_audio_stream_detects_video_without_audio(tmp_path):
    silent = tmp_path / "silent.mp4"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    assert has_audio_stream(str(silent)) is False
