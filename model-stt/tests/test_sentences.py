import pytest

from conftest import segment, word
from src.sentences import terminates_sentence, to_sentences


def test_splits_on_terminal_punctuation():
    seg = segment([
        word(" Hello", 0.0, 0.4),
        word(" world.", 0.4, 0.9),
        word(" How", 1.0, 1.2),
        word(" are", 1.2, 1.4),
        word(" you?", 1.4, 1.8),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Hello world.", "How are you?"]
    assert sentences[0].start == 0.0 and sentences[0].end == 0.9
    assert sentences[1].start == 1.0 and sentences[1].end == 1.8


def test_long_silence_splits_without_punctuation():
    # a dropped full stop must not glue together two distant utterances
    seg = segment([
        word(" one", 0.0, 0.5),
        word(" two", 0.5, 1.0),
        word(" three", 40.0, 40.5),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["one two", "three"]


def test_gap_within_threshold_does_not_split():
    seg = segment([
        word(" one", 0.0, 0.5),
        word(" two", 3.0, 3.5),
    ])
    assert len(to_sentences([seg], max_gap_ms=5000, max_words=150)) == 1


def test_trailing_words_without_punctuation_are_kept():
    seg = segment([
        word(" unfinished", 0.0, 0.5),
        word(" thought", 0.5, 1.0),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["unfinished thought"]


def test_falls_back_to_segment_boundaries_without_word_timings():
    seg = segment([word(" x", 0.0, 1.0)], text=" a whole segment ")
    seg = seg.__class__(
        start=0.0, end=5.0, text=" a whole segment ",
        avg_logprob=-0.2, no_speech_prob=0.01, words=[],
    )
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["a whole segment"]
    assert sentences[0].start == 0.0 and sentences[0].end == 5.0


def test_empty_input():
    assert to_sentences([], max_gap_ms=5000, max_words=150) == []


def test_titles_do_not_end_a_sentence():
    # observed in bench output: "Mr." was emitted as its own tag, splitting
    # "Mr. Anthony Eden's speech on the wireless."
    seg = segment([
        word(" Mr.", 0.0, 0.3),
        word(" Anthony", 0.3, 0.7),
        word(" Eden's", 0.7, 1.1),
        word(" speech.", 1.1, 1.6),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Mr. Anthony Eden's speech."]


def test_initials_do_not_end_a_sentence():
    seg = segment([
        word(" J.", 0.0, 0.2),
        word(" R.", 0.2, 0.4),
        word(" Tolkien", 0.4, 0.9),
        word(" wrote.", 0.9, 1.3),
    ])
    assert [s.text for s in to_sentences([seg], max_gap_ms=5000, max_words=150)] == ["J. R. Tolkien wrote."]


@pytest.mark.parametrize("token,ends", [
    ("Mr.", False), ("Dr.", False), ("St.", False), ("etc.", False),
    ("Mme.", False), ("Sra.", False),        # non-English titles
    ("A.", False), ("M.", False), ("É.", False),   # Latin initials
    ("done.", True), ("No.", True), ("really?", True),
    ("stop!", True), ("word", False),
    # single CJK characters are whole words, not initials: str.isalpha() is
    # Unicode-aware, so treating them as initials would suppress real breaks
    ("好.", True), ("네.", True), ("好。", True), ("ア.", True),
])
def test_terminates_sentence(token, ends):
    # "No." stays a terminator: as speech it is far more often a one-word
    # sentence than an abbreviation of "number"
    assert terminates_sentence(token) is ends


def test_single_character_cjk_words_still_split():
    seg = segment([
        word(" 네.", 0.0, 0.4),
        word(" 잘", 0.5, 0.8),
        word(" 지내요?", 0.8, 1.2),
    ])
    assert [s.text for s in to_sentences([seg], max_gap_ms=5000, max_words=150)] == ["네.", "잘 지내요?"]


@pytest.mark.parametrize("terminator", [".", "?", "!", "。", "！"])
def test_recognises_cjk_and_ascii_terminators(terminator):
    # multi-letter words: a lone Latin letter before '.' is an initial, not a terminator
    seg = segment([
        word(" one", 0.0, 0.5),
        word(f" two{terminator}", 0.5, 1.0),
        word(" three", 1.2, 1.5),
    ])
    assert len(to_sentences([seg], max_gap_ms=5000, max_words=150)) == 2


def test_deterministic_fallback_rungs_cover_the_temperature_ladder():
    """One deterministic rung per temperature rung, so an index never falls off
    the end. Rung 0 is unreachable (temperature 0 is not the sampling branch) but
    keeps the indices aligned."""
    from src.backends import TEMPERATURE_FALLBACK, _DeterministicFallback

    assert len(_DeterministicFallback.RUNGS) == len(TEMPERATURE_FALLBACK)
    # each rung must actually differ from the one before, or a retry repeats itself
    assert all(a != b for a, b in zip(_DeterministicFallback.RUNGS,
                                      _DeterministicFallback.RUNGS[1:]))
    # and none may reintroduce sampling
    forbidden = {"sampling_temperature", "sampling_topk", "num_hypotheses"}
    assert all(not forbidden & set(r) for r in _DeterministicFallback.RUNGS)


def test_deterministic_fallback_proxy_swaps_only_sampled_calls():
    from src.backends import _DeterministicFallback

    class FakeCT2:
        def __init__(self): self.calls = []
        def generate(self, *a, **kw): self.calls.append(kw); return "ok"
        def other_method(self): return "forwarded"

    inner = FakeCT2()
    proxy = _DeterministicFallback(inner)
    proxy.temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # disabled: passes straight through, sampling intact
    proxy.generate(sampling_temperature=0.4, sampling_topk=0, num_hypotheses=5)
    assert inner.calls[-1]["sampling_temperature"] == 0.4

    # enabled: sampling stripped, rung applied
    proxy.enabled = True
    proxy.generate(sampling_temperature=0.4, sampling_topk=0, num_hypotheses=5,
                   repetition_penalty=1.0)
    kw = inner.calls[-1]
    assert "sampling_temperature" not in kw and "sampling_topk" not in kw
    assert kw["repetition_penalty"] == 1.15  # rung 2 overrides the caller's value

    # a greedy call is untouched even when enabled
    proxy.generate(beam_size=5, patience=1)
    assert inner.calls[-1] == {"beam_size": 5, "patience": 1}

    assert proxy.other_method() == "forwarded"


# --------------------------------------------------------------------------
# speaker boundaries, when diarization has labelled the words
# --------------------------------------------------------------------------

def test_speaker_change_ends_a_caption_mid_sentence():
    # an interruption: whisper heard one sentence, two people said it, and the
    # handover left a pause where the microphone changed hands
    seg = segment([
        word(" Are", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" you", 0.2, 0.4, speaker="SPEAKER_00"),
        word(" seriously", 0.9, 1.3, speaker="SPEAKER_01"),
        word(" asking?", 1.3, 1.7, speaker="SPEAKER_01"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Are you", "seriously asking?"]
    assert [s.speaker for s in sentences] == ["SPEAKER_00", "SPEAKER_01"]


def test_one_speaker_throughout_is_one_caption():
    seg = segment([
        word(" It", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" is", 0.2, 0.4, speaker="SPEAKER_00"),
        word(" fine.", 0.4, 0.8, speaker="SPEAKER_00"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["It is fine."]
    assert sentences[0].speaker == "SPEAKER_00"


def test_an_unlabelled_word_does_not_split_a_caption():
    # the diarizer placed nobody on the middle word; that is missing evidence,
    # not a boundary, and it takes the label of the run it sits in
    seg = segment([
        word(" one", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" two", 0.2, 0.4, speaker=None),
        word(" three", 0.4, 0.8, speaker="SPEAKER_00"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["one two three"]
    assert sentences[0].speaker == "SPEAKER_00"


def test_words_before_the_first_label_join_the_run_that_follows():
    seg = segment([
        word(" um", 0.0, 0.2, speaker=None),
        word(" hello", 0.2, 0.4, speaker="SPEAKER_00"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["um hello"]
    assert sentences[0].speaker == "SPEAKER_00"


def test_wholly_unlabelled_words_make_a_caption_with_no_speaker():
    seg = segment([word(" music", 0.0, 0.4), word(" lyrics.", 0.4, 0.8)])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert sentences[0].speaker is None


def test_length_backstop_still_applies_within_one_speaker():
    # a long unpunctuated monologue is cut by max_words exactly as before;
    # speaker splitting shortens runs, it does not exempt them
    words = [word(f" w{i}", i * 0.1, i * 0.1 + 0.05, speaker="SPEAKER_00") for i in range(40)]
    sentences = to_sentences([segment(words)], max_gap_ms=5000, max_words=10)

    assert len(sentences) > 1
    assert all(len(s.text.split()) <= 10 for s in sentences)
    assert all(s.speaker == "SPEAKER_00" for s in sentences)


def test_speaker_split_pieces_keep_their_own_timings():
    seg = segment([
        word(" my", 0.0, 0.5, speaker="SPEAKER_00"),
        word(" turn", 0.5, 1.0, speaker="SPEAKER_00"),
        word(" your", 5.0, 5.5, speaker="SPEAKER_01"),
        word(" turn", 5.5, 6.0, speaker="SPEAKER_01"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert (sentences[0].start, sentences[0].end) == (0.0, 1.0)
    assert (sentences[1].start, sentences[1].end) == (5.0, 6.0)


def test_a_one_word_speaker_flip_does_not_split_a_caption():
    """The dominant failure mode: one word inside an utterance changes label.

    Over the seven-title run this was 52% of every split diarization made, and
    it produced captions like "Thank" / "you." -- so a lone word is not enough
    to end a caption.
    """
    seg = segment([
        word(" Thank", 0.0, 0.3, speaker="SPEAKER_00"),
        word(" you", 0.3, 0.5, speaker="SPEAKER_05"),
        word(" very", 0.5, 0.7, speaker="SPEAKER_00"),
        word(" much.", 0.7, 1.0, speaker="SPEAKER_00"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Thank you very much."]
    # the flip must not name the caption either
    assert sentences[0].speaker == "SPEAKER_00"


def test_a_leading_one_word_flip_is_absorbed_by_what_follows():
    # "Or" / "you get into a final club?" -- the fragment leads, and the real
    # speaker is the one holding the rest of the sentence
    seg = segment([
        word(" Or", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" you", 0.2, 0.4, speaker="SPEAKER_01"),
        word(" get", 0.4, 0.6, speaker="SPEAKER_01"),
        word(" in?", 0.6, 0.9, speaker="SPEAKER_01"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Or you get in?"]
    assert sentences[0].speaker == "SPEAKER_01"


def test_consecutive_flips_collapse_into_one_caption():
    seg = segment([
        word(" one", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" two", 0.2, 0.4, speaker="SPEAKER_01"),
        word(" three", 0.4, 0.6, speaker="SPEAKER_02"),
        word(" four", 0.6, 0.8, speaker="SPEAKER_00"),
        word(" five", 0.8, 1.0, speaker="SPEAKER_00"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["one two three four five"]
    assert sentences[0].speaker == "SPEAKER_00"


def test_a_speaker_change_between_contiguous_words_does_not_split():
    """The dominant failure mode after word count: a boundary out of step.

    Nobody takes over mid-phrase with no pause at all, but a diarizer boundary
    a word or two off whisper's timings looks exactly like that -- 87% of splits
    over the seven-title run, producing "You didn't" / "stay long." So a change
    with no silence in it is not a turn, however much substance sits either
    side.
    """
    seg = segment([
        word(" You", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" didn't", 0.2, 0.4, speaker="SPEAKER_00"),
        word(" stay", 0.4, 0.6, speaker="SPEAKER_01"),
        word(" long.", 0.6, 0.9, speaker="SPEAKER_01"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["You didn't stay long."]


def test_the_pause_is_measured_between_words_not_captions():
    # the gap that counts is the silence at the boundary itself, so a long
    # pause elsewhere in the caption does not license a split
    seg = segment([
        word(" One", 0.0, 0.2, speaker="SPEAKER_00"),
        word(" moment", 3.0, 3.4, speaker="SPEAKER_00"),
        word(" please", 3.4, 3.7, speaker="SPEAKER_01"),
        word(" sir.", 3.7, 4.0, speaker="SPEAKER_01"),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["One moment please sir."]


def test_undiarized_words_group_exactly_as_before():
    # with diarization off every speaker is None, and this rule never fires
    seg = segment([
        word(" Hello", 0.0, 0.4),
        word(" world.", 0.4, 0.9),
        word(" How", 1.0, 1.2),
        word(" are", 1.2, 1.4),
        word(" you?", 1.4, 1.8),
    ])
    sentences = to_sentences([seg], max_gap_ms=5000, max_words=150)

    assert [s.text for s in sentences] == ["Hello world.", "How are you?"]
    assert all(s.speaker is None for s in sentences)
