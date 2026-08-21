"""Test fixtures.

Most of the logic worth testing -- track assignment, sentence grouping, the
translation-model substitution, hallucination filtering -- does not depend on
whisper weights, so a FakeBackend stands in for the real runtime and those tests
run anywhere. Tests that genuinely need weights or a GPU are marked.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends import DecodeOptions, Segment, Transcription, WhisperBackend, Word  # noqa: E402
from src.model import RuntimeConfig, WhisperSTT  # noqa: E402
from src.punctuate import PunctuationConfig  # noqa: E402

# mirrors the models: block in config.yml
MODELS = {
    "large-v3-turbo": {
        "openai": "large-v3-turbo.pt",
        "ct2": "large-v3-turbo",
        "ct2_repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "translate": False,
    },
    "large-v3": {
        "openai": "large-v3.pt",
        "ct2": "large-v3",
        "ct2_repo": "Systran/faster-whisper-large-v3",
        "translate": True,
    },
}


def word(text: str, start: float, end: float, probability: float = 0.9) -> Word:
    return Word(start=start, end=end, word=text, probability=probability)


def segment(
    words: List[Word],
    text: Optional[str] = None,
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.01,
    compression_ratio: float = 1.0,
    temperature: float = 0.0,
) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=text if text is not None else "".join(w.word for w in words),
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        temperature=temperature,
        words=words,
    )


class FakeBackend(WhisperBackend):
    name = "fake"

    def __init__(self, transcription: Transcription):
        self.transcription = transcription
        self.calls: List[DecodeOptions] = []

    def transcribe(self, fpath: str, opts: DecodeOptions) -> Transcription:
        self.calls.append(opts)
        return self.transcription


@pytest.fixture
def make_model(monkeypatch, tmp_path):
    """Build a WhisperSTT wired to a FakeBackend, bypassing weight loading."""

    def _make(transcription: Transcription, cfg: Optional[RuntimeConfig] = None, **kwargs):
        backend = FakeBackend(transcription)
        monkeypatch.setattr("src.model.build_backend", lambda **_: backend)
        # every media path is treated as having audio unless a test says otherwise
        monkeypatch.setattr("src.model.has_audio_stream", lambda _: True)
        # same reason the backend is faked: punctuation restoration is a 2.2 GB
        # download these tests must not need. A test that wants it passes its own
        # punctuator= (see FakePunctuator) or a punctuation= config.
        kwargs.setdefault("punctuation", PunctuationConfig(enabled=False))
        model = WhisperSTT(
            cfg or RuntimeConfig(),
            models=MODELS,
            weights_dir=str(tmp_path),
            **kwargs,
        )
        model.backend = backend
        return model

    return _make


@pytest.fixture
def hello_world() -> Transcription:
    return Transcription(
        language="en",
        segments=[
            segment([
                word(" Hello", 0.0, 0.4),
                word(" world.", 0.4, 0.9),
                word(" How", 1.0, 1.2),
                word(" are", 1.2, 1.4),
                word(" you?", 1.4, 1.8),
            ])
        ],
    )
