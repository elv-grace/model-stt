import pytest

from conftest import MODELS, segment, word
from src.backends import Transcription
from src.model import (
    CONTEXT_CHARS,
    SENTENCE_TRACK,
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
        RuntimeConfig(carry_context=True), models=MODELS, weights_dir="/tmp",
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
        RuntimeConfig(carry_context=True), models=MODELS, weights_dir="/tmp",
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


# DISABLED (translation): tests for path B (whisper native translation), path C
# (turbo + LLM), and the turbo-cannot-translate auto-upgrade lived here. They are
# removed with the task/translator fields rather than commented, since they no
# longer compile against RuntimeConfig; git history has them.


# --- config validation ----------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        RuntimeConfig(backend="onnx"),
        RuntimeConfig(model_name="tiny"),
    ],
)
def test_invalid_config_rejected(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.model.build_backend", lambda **_: None)
    with pytest.raises(ValueError):
        WhisperSTT(cfg, models=MODELS, weights_dir=str(tmp_path))


def test_unique_short_fragment_is_kept_with_repaired_span(make_model):
    """Measured on real media: of 13 sub-100ms fragments, some were verbatim
    repeats but others ('Right?', 'Good.') were genuine short interjections whose
    alignment collapsed. Dropping those loses transcribed content."""
    real = segment([word(" Forbes had a good breakup.", 10.0, 12.0)])
    fragment = segment([word(" Right?", 12.5, 12.52)])   # 20ms, unrelated text
    tags = make_model(Transcription(language="en", segments=[real, fragment])).tag(FILE)

    captions = [t.tag for t in tracks(tags, SENTENCE_TRACK)]
    assert "Right?" in " ".join(captions)
    frag_words = [t for t in tracks(tags, WORD_TRACK) if t.tag == "Right?"]
    assert frag_words, "fragment text must survive"
    # its span is repaired to something physically plausible
    assert frag_words[0].end_time - frag_words[0].start_time >= 50


def test_near_repeat_fragment_is_still_dropped(make_model):
    """whisper's end-of-audio artifact often drops a leading word, so exact
    equality misses it."""
    real = segment([word(" But the prince didn't answer.", 79.0, 82.0)])
    repeat = segment([word(" The prince didn't answer.", 82.0, 82.02)])
    tags = make_model(Transcription(language="en", segments=[real, repeat])).tag(FILE)

    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["But the prince didn't answer."]


def test_repetitive_segment_dropped_only_after_failed_fallback(make_model):
    # fell back and still repetitive -> looping, drop
    looped = segment([word(" Thank you.", 1.0, 2.0)], compression_ratio=3.1, temperature=0.6)
    # repetitive text but a clean first-pass decode -> legitimate, keep
    clean = segment([word(" Thank you.", 3.0, 4.0)], compression_ratio=3.1, temperature=0.0)

    dropped = make_model(Transcription(language="en", segments=[looped])).tag(FILE)
    kept = make_model(Transcription(language="en", segments=[clean])).tag(FILE)

    assert tracks(dropped, SENTENCE_TRACK) == []
    assert [t.tag for t in tracks(kept, SENTENCE_TRACK)] == ["Thank you."]


def test_repetition_signals_reach_additional_info(make_model):
    seg = segment([word(" Hello.", 0.0, 1.0)], compression_ratio=1.4, temperature=0.2)
    tags = make_model(Transcription(language="en", segments=[seg])).tag(FILE)

    info = tracks(tags, WORD_TRACK)[0].additional_info
    assert info["compression_ratio"] == 1.4
    assert info["temperature"] == 0.2


def test_segment_straddling_a_vad_cut_is_pulled_back_together(make_model):
    """The real 'Good luck, guys.' case: restore_speech_timestamps resolved the
    first word to the chunk before a 207s excision and the rest to the chunk
    after, stretching one 3-word segment across the whole cut."""
    straddled = segment([
        word(" Good", 126.71, 126.95),
        word(" luck,", 333.54, 333.66),
        word(" guys.", 333.70, 333.92),
    ])
    tags = make_model(
        Transcription(language="en", segments=[straddled]), cfg=RuntimeConfig(vad_filter=True)
    ).tag(FILE)

    sentences = tracks(tags, SENTENCE_TRACK)
    assert [t.tag for t in sentences] == ["Good luck, guys."]
    # the run holding the most spoken time wins, so the phrase lands at ~333s;
    # decoding the same audio unfiltered puts it at 333.06-333.90
    assert 333_000 <= sentences[0].start_time <= 333_500
    assert sentences[0].end_time == 333_920


def test_straddle_repair_leaves_ordinary_pauses_alone(make_model):
    """A pause shorter than straddle_gap_max is real speech timing, not an
    excision, and must survive untouched."""
    seg = segment([word(" Wait", 1.0, 1.4), word(" for", 3.2, 3.4), word(" it.", 3.4, 3.8)])
    tags = make_model(Transcription(language="en", segments=[seg])).tag(FILE)

    assert [(t.start_time, t.end_time) for t in tracks(tags, WORD_TRACK)] == [
        (1000, 1400), (3200, 3400), (3400, 3800)
    ]


def test_repaired_straddle_does_not_overlap_its_neighbours(make_model):
    """Packing the misplaced run back against the anchor must stop at the
    neighbouring segments, which keep their own timings."""
    before = segment([word(" Who did that?", 124.5, 126.4)])
    straddled = segment([word(" Good", 126.7, 126.95), word(" luck.", 127.2, 127.4)])
    after = segment([word(" See you.", 127.5, 128.0)])
    cfg = RuntimeConfig(vad_filter=True, straddle_gap_max=0.2)  # force both runs split
    tags = make_model(
        Transcription(language="en", segments=[before, straddled, after]), cfg=cfg
    ).tag(FILE)

    words = tracks(tags, WORD_TRACK)
    assert all(a.end_time <= b.start_time for a, b in zip(words, words[1:]))


def test_vad_tuning_reaches_the_backend(make_model, hello_world):
    cfg = RuntimeConfig(vad_threshold=0.4, vad_min_silence_ms=1500, vad_speech_pad_ms=900)
    model = make_model(hello_world, cfg=cfg)
    model.tag(FILE)

    opts = model.backend.calls[0]
    assert (opts.vad_threshold, opts.vad_min_silence_ms, opts.vad_speech_pad_ms) == (
        0.4, 1500, 900
    )


def test_sentence_tags_carry_word_probability_aggregates(make_model):
    """So a consumer can weigh a caption without joining to the word track."""
    seg = segment([
        word(" Hey,", 0.0, 0.2, probability=0.3435),
        word(" Mr.", 0.4, 0.6, probability=0.5352),
        word(" Sheldon,", 0.7, 1.1, probability=0.4955),
        word(" you're", 1.2, 1.3, probability=0.9993),
        word(" not", 1.3, 1.5, probability=0.9985),
        word(" crazy.", 1.5, 1.9, probability=0.999),
    ])
    tags = make_model(Transcription(language="en", segments=[seg])).tag(FILE)

    info = tracks(tags, SENTENCE_TRACK)[0].additional_info
    assert info["min_word_probability"] == 0.3435
    assert info["mean_word_probability"] == 0.7285


def test_probability_aggregates_omitted_when_there_are_no_word_timings(make_model):
    """The segment-level fallback has nothing to aggregate; the keys are absent
    rather than null so 'not measured' is distinguishable from 'measured low'."""
    from src.backends import Segment

    seg = Segment(start=0.0, end=1.0, text=" Hello.", avg_logprob=-0.2,
                  no_speech_prob=0.01, words=[])
    tags = make_model(Transcription(language="en", segments=[seg])).tag(FILE)

    info = tracks(tags, SENTENCE_TRACK)[0].additional_info
    assert "min_word_probability" not in info
    assert info["language"] == "en"


def _spread(texts, gap, start=0.0, dur=0.6):
    """Segments laid out with `gap` seconds of silence between them."""
    segs, t = [], start
    for text in texts:
        segs.append(segment([word(f" {text}", t, t + dur)]))
        t += dur + gap
    return segs


def test_isolated_recurring_artifact_is_dropped(make_model):
    """Whisper's stock non-speech phrase: recurs, and always stands alone."""
    segs = _spread(["Thank you."] * 5, gap=40.0)
    tags = make_model(Transcription(language="en", segments=segs)).tag(FILE)

    assert tracks(tags, SENTENCE_TRACK) == []
    assert tracks(tags, WORD_TRACK) == []  # word track drops with it


def test_conversational_phrase_is_kept_however_often_it_recurs(make_model):
    """The median-isolation test is what protects real dialogue: 'Thank you.'
    said inside a conversation has neighbours, so it is never an artifact."""
    segs = _spread(["Thank you."] * 5, gap=0.5)
    tags = make_model(Transcription(language="en", segments=segs)).tag(FILE)

    assert len(tracks(tags, SENTENCE_TRACK)) == 5


def test_one_off_isolated_line_is_kept(make_model):
    """A short real line can legitimately sit alone between long musical
    stretches -- 'Whoa.', 'Hello.' -- so recurrence is required too."""
    segs = _spread(["Whoa.", "Hello.", "Again, Moss?", "One, two, three!"], gap=40.0)
    tags = make_model(Transcription(language="en", segments=segs)).tag(FILE)

    assert len(tracks(tags, SENTENCE_TRACK)) == 4


def test_only_the_isolated_occurrences_are_dropped(make_model):
    """A phrase that is an artifact in this file may still be spoken for real in
    one place; the instance with company survives."""
    segs = _spread(["Thank you."] * 4, gap=40.0)
    real = segment([word(" Thank you.", 400.0, 400.6)])
    neighbour = segment([word(" You're welcome.", 401.0, 402.0)])
    tags = make_model(
        Transcription(language="en", segments=segs + [real, neighbour])
    ).tag(FILE)

    assert [t.tag for t in tracks(tags, SENTENCE_TRACK)] == ["Thank you.", "You're welcome."]


def test_isolated_artifact_guard_can_be_disabled(make_model):
    cfg = RuntimeConfig(isolated_artifact_gap=0.0)
    segs = _spread(["Thank you."] * 5, gap=40.0)
    tags = make_model(Transcription(language="en", segments=segs), cfg=cfg).tag(FILE)

    assert len(tracks(tags, SENTENCE_TRACK)) == 5


def test_long_isolated_segment_is_never_an_artifact(make_model):
    """Only short texts are candidates; a full sentence standing alone is
    normal in sparse-dialogue content."""
    segs = _spread(["I am here to talk about the first organization."] * 5, gap=40.0)
    tags = make_model(Transcription(language="en", segments=segs)).tag(FILE)

    assert len(tracks(tags, SENTENCE_TRACK)) == 5


def test_shipped_defaults_decode_everything_unconditioned(make_model, hello_world):
    """VAD off and conditioning off are a pair, not two independent choices: with
    VAD off the decoder sees noise it was shielded from, and conditioning is what
    turns that into a repeat loop. Guard the coupling so neither drifts alone."""
    cfg = RuntimeConfig()
    assert (cfg.vad_filter, cfg.condition_on_previous_text) == (False, False)

    model = make_model(hello_world, cfg=cfg)
    model.tag(FILE)
    opts = model.backend.calls[0]
    assert opts.vad_filter is False
    assert opts.condition_on_previous_text is False


def test_straddle_repair_does_not_run_without_vad(make_model):
    """A straddle is restore_speech_timestamps mapping words back across a VAD
    excision. With VAD off there is no excision, every gap is real, and repairing
    one moves real speech: over 13.2h it fired 9 times, all false positives."""
    real_pause = segment([
        word(" Oh,", 32.24, 33.04),
        word(" my", 38.48, 38.84),
        word(" God.", 38.84, 39.24),
    ])
    tags = make_model(Transcription(language="en", segments=[real_pause])).tag(FILE)

    assert [(t.start_time, t.end_time) for t in tracks(tags, WORD_TRACK)] == [
        (32240, 33040), (38480, 38840), (38840, 39240)
    ]


def test_default_keeps_a_punctuated_sentence_whole_across_a_long_pause(make_model):
    """Both halves of the fix, on the shipped default. The repair is off (VAD is
    off) so the word timings stay as decoded, and to_sentences no longer ends a
    caption on a pause, so the sentence is not torn."""
    seg = segment([
        word(" Oh,", 32.24, 33.04),
        word(" my", 38.48, 38.84),
        word(" God.", 38.84, 39.24),
    ])
    sentences = tracks(make_model(Transcription(language="en", segments=[seg])).tag(FILE),
                       SENTENCE_TRACK)

    assert [t.tag for t in sentences] == ["Oh, my God."]
    assert (sentences[0].start_time, sentences[0].end_time) == (32240, 39240)


def test_default_does_not_split_good_luck_guys(make_model):
    """The same input as the straddle test, decoded WITHOUT vad: nothing repairs
    the timings, and nothing tears the sentence either."""
    seg = segment([
        word(" Good", 126.71, 126.95),
        word(" luck,", 333.54, 333.66),
        word(" guys.", 333.70, 333.92),
    ])
    sentences = tracks(make_model(Transcription(language="en", segments=[seg])).tag(FILE),
                       SENTENCE_TRACK)

    assert [t.tag for t in sentences] == ["Good luck, guys."]
    assert (sentences[0].start_time, sentences[0].end_time) == (126710, 333920)


def test_deterministic_fallback_is_on_by_default_and_reaches_the_backend(make_model, hello_world):
    """Reproducible decoding is the shipped default; the `stochastic` profile opts out."""
    default = make_model(hello_world)
    default.tag(FILE)
    assert default.backend.calls[0].deterministic_fallback is True

    opted_out = make_model(hello_world, cfg=RuntimeConfig(deterministic_fallback=False))
    opted_out.tag(FILE)
    assert opted_out.backend.calls[0].deterministic_fallback is False
