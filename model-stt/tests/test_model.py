import pytest

from conftest import MODELS, segment, word
from src.backends import Transcription
from src.model import (
    CONTEXT_CHARS,
    SENTENCE_TRACK,
    TRANSLATION_TRACK,
    WORD_TRACK,
    RuntimeConfig,
    WhisperSTT,
)

FILE = "test-files/1.m4a"


def tracks(tags, track):
    return [t for t in tags if t.track == track]


def test_emits_word_and_sentence_tracks(make_model, hello_world):
    tags = make_model(hello_world).tag(FILE)

    words = tracks(tags, WORD_TRACK)
    sentences = tracks(tags, SENTENCE_TRACK)
    assert [t.tag for t in words] == ["Hello", "world.", "How", "are", "you?"]
    assert [t.tag for t in sentences] == ["Hello world.", "How are you?"]
    assert all(t.source_media == FILE for t in tags)
    assert all(t.additional_info["language"] == "en" for t in tags)


def test_timestamps_are_milliseconds_and_ordered(make_model, hello_world):
    tags = make_model(hello_world).tag(FILE)

    first = tracks(tags, WORD_TRACK)[0]
    assert (first.start_time, first.end_time) == (0, 400)
    assert all(t.end_time >= t.start_time for t in tags)
    assert [t.start_time for t in tags] == sorted(t.start_time for t in tags)


def test_word_level_off_still_produces_timed_sentences(make_model, hello_world):
    model = make_model(hello_world, RuntimeConfig(word_level=False))
    tags = model.tag(FILE)

    assert tracks(tags, WORD_TRACK) == []
    # word timings must still be requested: the sentence track needs them
    assert model.backend.calls[0].word_timestamps is True
    assert tracks(tags, SENTENCE_TRACK)[0].end_time == 900


def test_silent_segment_dropped_only_when_both_signals_agree(make_model):
    confident_over_noise = segment(
        [word(" real", 0.0, 0.5)], no_speech_prob=0.99, avg_logprob=-0.1
    )
    hallucinated = segment(
        [word(" Thanks for watching!", 1.0, 2.0)], no_speech_prob=0.99, avg_logprob=-2.0
    )
    transcription = Transcription(language="en", segments=[confident_over_noise, hallucinated])

    tags = make_model(transcription).tag(FILE)

    kept = [t.tag for t in tracks(tags, WORD_TRACK)]
    assert kept == ["real"]


def test_degenerate_end_of_audio_repetition_is_dropped(make_model):
    """faster-whisper emits a trailing repeat of the previous sentence collapsed
    into ~20ms with zero-duration words, and reports it as confident."""
    real = segment([word(" But", 79.2, 79.6), word(" the prince didn't answer.", 79.6, 82.14)])
    repeat = segment(
        [word(" The", 82.14, 82.14), word(" prince", 82.14, 82.14),
         word(" didn't answer.", 82.14, 82.16)],
        no_speech_prob=0.0, avg_logprob=-0.253,   # confident: the usual filters miss it
    )
    tags = make_model(Transcription(language="en", segments=[real, repeat])).tag(FILE)

    captions = [t.tag for t in tracks(tags, SENTENCE_TRACK)]
    assert captions == ["But the prince didn't answer."]
    # its zero-duration words must not survive either
    assert all(t.start_time < 82140 for t in tracks(tags, WORD_TRACK))


def test_short_but_plausible_segment_is_kept(make_model):
    # a genuine one-word utterance must not be swept up by the duration floor
    seg = segment([word(" No.", 1.0, 1.3)])
    tags = make_model(Transcription(language="en", segments=[seg])).tag(FILE)
    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["No."]


def test_empty_segments_produce_no_tags(make_model):
    blank = segment([word("   ", 0.0, 0.5)], text="   ")
    assert make_model(Transcription(language="en", segments=[blank])).tag(FILE) == []


def test_context_carries_into_the_next_file(make_model, hello_world):
    model = make_model(hello_world, RuntimeConfig(carry_context=True))

    model.tag(FILE)
    model.tag("test-files/2.m4a")

    assert model.backend.calls[0].initial_prompt is None
    assert model.backend.calls[1].initial_prompt.endswith("How are you?")


def test_carried_context_is_bounded_and_starts_at_a_boundary(make_model):
    # a long prompt suppressed whole files in measurement, and a raw slice opened
    # on a word fragment ("hen there was a war")
    long_speech = Transcription(language="en", segments=[
        segment([word(f" word{i}.", i * 0.5, i * 0.5 + 0.4) for i in range(60)])
    ])
    model = make_model(long_speech, RuntimeConfig(carry_context=True))
    model.tag(FILE)

    assert len(model._prev_tail) <= CONTEXT_CHARS
    assert not model._prev_tail.startswith(("ord", "rd"))
    assert model._prev_tail[0].isalnum()


def test_prompted_decode_that_returns_nothing_is_retried_without_the_prompt(
    make_model, hello_world, monkeypatch
):
    empty = Transcription(language="en", segments=[])

    class SuppressingBackend:
        """Returns nothing whenever a prompt is supplied, as whisper did at 220 chars."""
        name = "suppressing"

        def __init__(self):
            self.calls = []

        def transcribe(self, fpath, opts):
            self.calls.append(opts)
            return empty if opts.initial_prompt is not None else hello_world

    backend = SuppressingBackend()
    monkeypatch.setattr("src.model.build_backend", lambda **_: backend)
    monkeypatch.setattr("src.model.has_audio_stream", lambda _: True)
    model = WhisperSTT(
        RuntimeConfig(carry_context=True), models=MODELS,
        weights_dir="/tmp", translate_fallback="large-v3",
    )

    model.tag(FILE)          # no prompt yet, primes _prev_tail
    tags = model.tag("test-files/2.m4a")   # prompted -> suppressed -> retried

    assert backend.calls[1].initial_prompt is not None
    assert backend.calls[2].initial_prompt is None
    assert [t.tag for t in tracks(tags, WORD_TRACK)] == ["Hello", "world.", "How", "are", "you?"]


def test_language_change_discards_the_carried_prompt(monkeypatch, hello_world):
    """A Korean tail carried into a French file cost 88% of its words in
    measurement; a language change means the prompt does not belong to this file."""
    french = Transcription(
        language="fr", segments=[segment([word(" Bonjour", 0.0, 0.5)])]
    )

    class LanguageSwitchingBackend:
        name = "switching"

        def __init__(self):
            self.calls = []

        def transcribe(self, fpath, opts):
            self.calls.append(opts)
            return hello_world if fpath == FILE else french

    backend = LanguageSwitchingBackend()
    monkeypatch.setattr("src.model.build_backend", lambda **_: backend)
    monkeypatch.setattr("src.model.has_audio_stream", lambda _: True)
    model = WhisperSTT(
        RuntimeConfig(carry_context=True), models=MODELS,
        weights_dir="/tmp", translate_fallback="large-v3",
    )

    model.tag(FILE)                        # english, primes the tail
    tags = model.tag("test-files/fr.mp3")  # french -> prompt is from another language

    assert backend.calls[1].initial_prompt is not None
    assert backend.calls[2].initial_prompt is None   # re-decoded clean
    assert all(t.additional_info["language"] == "fr" for t in tags)


def test_reset_context_clears_carried_state(make_model, hello_world):
    model = make_model(hello_world, RuntimeConfig(carry_context=True))
    model.tag(FILE)
    assert model._prev_tail is not None

    model.reset_context()

    assert model._prev_tail is None
    model.tag("test-files/2.m4a")
    assert model.backend.calls[-1].initial_prompt is None


def test_context_is_not_carried_by_default(make_model, hello_world):
    # the enabled case is covered by test_context_carries_into_the_next_file
    model = make_model(hello_world)
    assert model.cfg.carry_context is False

    model.tag(FILE)
    model.tag("test-files/2.m4a")

    assert all(call.initial_prompt is None for call in model.backend.calls)


def test_file_without_audio_is_skipped(make_model, hello_world, monkeypatch):
    model = make_model(hello_world)
    monkeypatch.setattr("src.model.has_audio_stream", lambda _: False)

    assert model.tag("test-files/silent.mp4") == []


# --- path selection -------------------------------------------------------


def test_path_a_transcribes_with_turbo(make_model, hello_world):
    model = make_model(hello_world, RuntimeConfig(model_name="large-v3-turbo"))
    model.tag(FILE)

    assert model.effective_model_name == "large-v3-turbo"
    assert model.backend.calls[0].task == "transcribe"


def test_path_b_uses_whisper_translation_and_labels_output_english(make_model):
    french = Transcription(
        language="fr", segments=[segment([word(" Good", 0.0, 0.5), word(" morning.", 0.5, 1.0)])]
    )
    cfg = RuntimeConfig(model_name="large-v3", task="translate", translator="whisper")
    model = make_model(french, cfg)
    tags = model.tag(FILE)

    assert model.backend.calls[0].task == "translate"
    # whisper's translate task emits English regardless of the detected language
    assert all(t.additional_info["language"] == "en" for t in tags)
    assert tracks(tags, TRANSLATION_TRACK) == []


def test_turbo_translation_request_is_upgraded_not_served(make_model, hello_world, caplog):
    # turbo accepts task="translate" and returns untranslated source-language
    # text, so the request must be substituted rather than honoured
    cfg = RuntimeConfig(model_name="large-v3-turbo", task="translate", translator="whisper")
    model = make_model(hello_world, cfg)

    assert model.effective_model_name == "large-v3"


def test_translation_without_a_usable_fallback_raises(monkeypatch, tmp_path, hello_world):
    monkeypatch.setattr("src.model.build_backend", lambda **_: None)
    cfg = RuntimeConfig(model_name="large-v3-turbo", task="translate", translator="whisper")

    with pytest.raises(ValueError, match="translate_fallback"):
        WhisperSTT(cfg, models=MODELS, weights_dir=str(tmp_path), translate_fallback=None)


def test_path_c_transcribes_in_source_language_and_adds_translation_track(
    make_model, monkeypatch
):
    class StubTranslator:
        def __init__(self, *_, **__):
            pass

        def translate(self, sentences, source_language):
            assert source_language == "fr"
            return [s.__class__(start=s.start, end=s.end, text="Good morning.") for s in sentences]

    monkeypatch.setattr("src.model.LLMTranslator", StubTranslator)

    french = Transcription(
        language="fr", segments=[segment([word(" Bonjour", 0.0, 0.5), word(" matin.", 0.5, 1.0)])]
    )
    cfg = RuntimeConfig(model_name="large-v3-turbo", task="translate", translator="llm")
    model = make_model(french, cfg)
    tags = model.tag(FILE)

    # path C keeps turbo and does not use whisper's translate task
    assert model.effective_model_name == "large-v3-turbo"
    assert model.backend.calls[0].task == "transcribe"

    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["Bonjour matin."]
    assert tracks(tags, SENTENCE_TRACK)[0].additional_info["language"] == "fr"

    translated = tracks(tags, TRANSLATION_TRACK)
    assert [t.tag for t in translated] == ["Good morning."]
    assert translated[0].additional_info["language"] == "en"
    # the span comes from whisper's alignment, not from the translator
    assert (translated[0].start_time, translated[0].end_time) == (0, 1000)


# --- config validation ----------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        RuntimeConfig(task="summarize"),
        RuntimeConfig(translator="deepl"),
        RuntimeConfig(backend="onnx"),
        RuntimeConfig(model_name="tiny"),
    ],
)
def test_invalid_config_rejected(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.model.build_backend", lambda **_: None)
    with pytest.raises(ValueError):
        WhisperSTT(cfg, models=MODELS, weights_dir=str(tmp_path), translate_fallback="large-v3")
