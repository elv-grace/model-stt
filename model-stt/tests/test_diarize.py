"""Diarization: timeline queries, cross-part identity, and failure behaviour.

Nothing here launches the binary or touches a GPU. What is worth testing is the
logic around speakrs, not speakrs: how a span is resolved to one speaker, how a
part's labels are carried onto the next part's, and what a failure costs -- which
is the speaker fields and nothing else, since the transcript is already decoded
by the time the diarizer runs.
"""
import subprocess

import pytest

from conftest import FakeDiarizer, diarization, segment, word
from src.backends import Transcription
from src.diarize import (
    DiarizationConfig,
    SpeakerDiarizer,
    SpeakerTurn,
    _covered_fraction,
    _overlapped_regions,
    _union,
    build_diarizer,
)


@pytest.fixture
def diarizer(tmp_path):
    """A SpeakerDiarizer whose binary exists but is never run.

    Only the label-matching half is exercised here; _run is the part that needs
    a GPU, and what it produces is covered by the integration tests.
    """
    binary = tmp_path / "stt-diarize"
    binary.touch(mode=0o755)
    return SpeakerDiarizer(
        DiarizationConfig(enabled=True, binary=str(binary), models_dir=None)
    )


def throws(exception):
    def _raise(*_args, **_kwargs):
        raise exception

    return _raise


# --------------------------------------------------------------------------
# resolving a span to one speaker
# --------------------------------------------------------------------------

def test_speaker_at_picks_the_dominant_overlap():
    # the word straddles a handover, and belongs to whoever holds more of it
    result = diarization([(0.0, 10.0, "SPEAKER_00"), (10.0, 20.0, "SPEAKER_01")])
    assert result.speaker_at(9.7, 10.6) == "SPEAKER_01"
    assert result.speaker_at(9.4, 10.3) == "SPEAKER_00"


def test_speaker_at_returns_none_outside_every_turn():
    # a word over music: whisper produced it, the diarizer places nobody on it
    result = diarization([(0.0, 5.0, "SPEAKER_00")])
    assert result.speaker_at(30.0, 31.0) is None


def test_speaker_at_resolves_a_collapsed_word():
    # whisper emits zero-duration words; a span of nothing still sits somewhere
    result = diarization([(0.0, 5.0, "SPEAKER_00")])
    assert result.speaker_at(3.0, 3.0) == "SPEAKER_00"
    assert result.speaker_at(9.0, 9.0) is None


def test_speaker_at_on_an_empty_timeline():
    assert diarization([]).speaker_at(1.0, 2.0) is None


def test_coverage_measures_speech_over_the_span():
    result = diarization([(0.0, 1.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")])
    assert result.coverage(0.0, 4.0) == pytest.approx(0.5)
    assert result.coverage(0.0, 1.0) == pytest.approx(1.0)
    assert result.coverage(1.0, 2.0) == pytest.approx(0.0)


def test_overlap_measures_only_simultaneous_speech():
    # the exclusive timeline says 00 then 01; the raw one says they talked over
    # each other for half a second
    result = diarization(
        [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")],
        raw=[(0.0, 2.5, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")],
    )
    assert result.overlap(0.0, 4.0) == pytest.approx(0.125)
    assert result.overlap(0.0, 1.0) == pytest.approx(0.0)


def test_union_merges_touching_and_nested_spans():
    turns = [
        SpeakerTurn(0.0, 2.0, "A"),
        SpeakerTurn(1.0, 1.5, "B"),   # nested
        SpeakerTurn(2.0, 3.0, "C"),   # touching
        SpeakerTurn(5.0, 6.0, "D"),
    ]
    assert _union(turns) == [(0.0, 3.0), (5.0, 6.0)]


def test_overlapped_regions_ignores_a_clean_handover():
    # one turn ending exactly where the next begins is not two people talking
    turns = [SpeakerTurn(0.0, 2.0, "A"), SpeakerTurn(2.0, 4.0, "B")]
    assert _overlapped_regions(turns) == []


def test_overlapped_regions_finds_simultaneous_speech():
    turns = [SpeakerTurn(0.0, 3.0, "A"), SpeakerTurn(2.0, 5.0, "B")]
    assert _overlapped_regions(turns) == [(2.0, 3.0)]


def test_covered_fraction_of_an_empty_span():
    assert _covered_fraction([(0.0, 10.0)], 5.0, 5.0) == 0.0


# --------------------------------------------------------------------------
# carrying speaker identity across the parts of one asset
# --------------------------------------------------------------------------

def test_same_voice_keeps_its_label_across_parts(diarizer):
    voice = [1.0, 0.0, 0.0]
    first = diarizer._globalize(
        [SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": voice}
    )
    # speakrs numbers each run on its own, so the next part calls the same
    # person something else -- the whole reason the registry exists
    second = diarizer._globalize(
        [SpeakerTurn(0.0, 10.0, "SPEAKER_03")], {"SPEAKER_03": voice}
    )
    assert first["SPEAKER_00"] == second["SPEAKER_03"]


def test_a_different_voice_gets_a_new_label(diarizer):
    diarizer._globalize([SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": [1.0, 0.0]})
    mapping = diarizer._globalize(
        [SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": [0.0, 1.0]}
    )
    assert mapping["SPEAKER_00"] == "SPEAKER_01"


def test_two_speakers_never_collapse_onto_one_label(diarizer):
    diarizer._globalize([SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": [1.0, 0.0]})
    # both of these resemble the known voice; only the better match may take it
    mapping = diarizer._globalize(
        [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(10.0, 20.0, "SPEAKER_01")],
        {"SPEAKER_00": [0.99, 0.14], "SPEAKER_01": [0.96, 0.28]},
    )
    assert mapping["SPEAKER_00"] != mapping["SPEAKER_01"]
    assert "SPEAKER_00" in mapping.values()


def test_labels_follow_speaking_time_not_turn_order(diarizer):
    # the first turn belongs to the minor speaker; numbering should still lead
    # with whoever actually carries the scene
    mapping = diarizer._globalize(
        [SpeakerTurn(0.0, 1.0, "SPEAKER_05"), SpeakerTurn(1.0, 60.0, "SPEAKER_09")],
        {"SPEAKER_05": [1.0, 0.0], "SPEAKER_09": [0.0, 1.0]},
    )
    assert mapping["SPEAKER_09"] == "SPEAKER_00"
    assert mapping["SPEAKER_05"] == "SPEAKER_01"


def test_reset_forgets_every_speaker(diarizer):
    voice = [1.0, 0.0]
    diarizer._globalize([SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": voice})
    diarizer.reset()
    # with the registry cleared the same voice is a stranger again, which is
    # what stops one asset's speakers collecting the next one's
    assert diarizer._centroids == {}
    mapping = diarizer._globalize(
        [SpeakerTurn(0.0, 10.0, "SPEAKER_07")], {"SPEAKER_07": voice}
    )
    assert mapping["SPEAKER_07"] == "SPEAKER_00"


def test_a_speaker_without_an_embedding_still_gets_a_label(diarizer):
    # nothing to match it by later, but it must not vanish from the timeline
    mapping = diarizer._globalize([SpeakerTurn(0.0, 10.0, "SPEAKER_00")], {})
    assert mapping == {"SPEAKER_00": "SPEAKER_00"}
    assert diarizer._centroids == {}


def test_registry_weights_by_speaking_time(diarizer):
    diarizer._globalize([SpeakerTurn(0.0, 100.0, "SPEAKER_00")], {"SPEAKER_00": [1.0, 0.0]})
    diarizer._globalize([SpeakerTurn(0.0, 1.0, "SPEAKER_00")], {"SPEAKER_00": [0.8, 0.6]})
    # a one-second sample must not move the centroid as far as a 100-second one
    assert diarizer._centroids["SPEAKER_00"][0] > 0.99


# --------------------------------------------------------------------------
# what a failure costs
# --------------------------------------------------------------------------

def test_disabled_config_builds_nothing():
    assert build_diarizer(DiarizationConfig(enabled=False), weights_dir=None) is None


def test_a_missing_binary_degrades_quietly(tmp_path):
    cfg = DiarizationConfig(enabled=True, binary=str(tmp_path / "absent"))
    assert build_diarizer(cfg, weights_dir=None) is None


def test_a_missing_binary_raises_when_required(tmp_path):
    cfg = DiarizationConfig(enabled=True, binary=str(tmp_path / "absent"), required=True)
    with pytest.raises(FileNotFoundError):
        build_diarizer(cfg, weights_dir=None)


def test_a_missing_model_bundle_is_reported_as_such(tmp_path):
    binary = tmp_path / "stt-diarize"
    binary.touch(mode=0o755)
    cfg = DiarizationConfig(
        enabled=True, binary=str(binary), models_dir=str(tmp_path / "absent"), required=True
    )
    with pytest.raises(FileNotFoundError, match="speakrs models"):
        build_diarizer(cfg, weights_dir=None)


def test_a_failed_run_leaves_the_file_unlabelled(diarizer, monkeypatch):
    monkeypatch.setattr(SpeakerDiarizer, "_run", throws(RuntimeError("boom")))
    assert diarizer.diarize("whatever.mp4") is None


def test_a_failed_run_raises_when_required(tmp_path, monkeypatch):
    binary = tmp_path / "stt-diarize"
    binary.touch(mode=0o755)
    diarizer = SpeakerDiarizer(
        DiarizationConfig(enabled=True, binary=str(binary), models_dir=None, required=True)
    )
    monkeypatch.setattr(SpeakerDiarizer, "_run", throws(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        diarizer.diarize("whatever.mp4")


def test_a_crashed_run_is_retried(tmp_path, monkeypatch):
    # speakrs' CUDA path aborts intermittently; the same audio survives a retry
    binary = tmp_path / "stt-diarize"
    binary.touch(mode=0o755)
    diarizer = SpeakerDiarizer(
        DiarizationConfig(enabled=True, binary=str(binary), models_dir=None, retries=1)
    )

    attempts = []

    def flaky(self, fpath):
        attempts.append(fpath)
        if len(attempts) == 1:
            raise RuntimeError("corrupted double-linked list")
        return {"duration": 1.0, "segments": [], "turns": [], "embeddings": {}}

    monkeypatch.setattr(SpeakerDiarizer, "_run", flaky)
    assert diarizer.diarize("whatever.mp4") is not None
    assert len(attempts) == 2


def test_a_timeout_is_not_retried(tmp_path, monkeypatch):
    # a retry cannot make the work fit a budget it has already overrun
    binary = tmp_path / "stt-diarize"
    binary.touch(mode=0o755)
    diarizer = SpeakerDiarizer(
        DiarizationConfig(enabled=True, binary=str(binary), models_dir=None, retries=3)
    )

    attempts = []

    def slow(self, fpath):
        attempts.append(fpath)
        raise subprocess.TimeoutExpired("stt-diarize", 1)

    monkeypatch.setattr(SpeakerDiarizer, "_run", slow)
    assert diarizer.diarize("whatever.mp4") is None
    assert len(attempts) == 1


def test_a_failed_diarization_costs_the_speakers_and_nothing_else(make_model):
    """The transcript is decoded before the diarizer runs, so it survives it.

    A diarizer that returns None must leave output identical to running with
    diarization off: same text, same timings, and no speaker fields at all --
    not null ones, which would claim the question was asked and unanswered.
    """
    transcription = Transcription(
        language="en",
        segments=[segment([word(" Hello", 0.0, 0.4), word(" world.", 0.4, 0.9)])],
    )
    failed = make_model(transcription, diarizer=FakeDiarizer(None))
    off = make_model(transcription)

    assert [(t.tag, t.start_time, t.end_time) for t in failed.tag("a.mp4")] == \
           [(t.tag, t.start_time, t.end_time) for t in off.tag("a.mp4")]
    assert all("speaker" not in t.additional_info for t in failed.tag("a.mp4"))


def test_reset_context_clears_the_speaker_registry(make_model):
    model = make_model(
        Transcription(language="en", segments=[]),
        diarizer=FakeDiarizer(diarization([(0.0, 1.0, "SPEAKER_00")])),
    )
    model.reset_context()
    assert model.diarizer.resets == 1
