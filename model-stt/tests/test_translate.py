import json

import pytest

from src.sentences import Sentence
from src.translate import LLMTranslator, TranslatorConfig, _extract_json

SENTENCES = [
    Sentence(start=0.0, end=1.0, text="Bonjour."),
    Sentence(start=1.0, end=2.0, text="Comment allez-vous ?"),
]


class StubClient:
    """Replays canned replies and records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"response": reply}


def translator(replies, **cfg_kwargs):
    cfg = TranslatorConfig(**cfg_kwargs)
    return LLMTranslator(cfg, client=StubClient(replies))


def reply(*texts, start_id=1):
    return json.dumps(
        {"translations": [{"id": start_id + i, "text": t} for i, t in enumerate(texts)]}
    )


def test_translates_and_preserves_spans():
    t = translator([reply("Hello.", "How are you?")])
    out = t.translate(SENTENCES, "fr")

    assert [s.text for s in out] == ["Hello.", "How are you?"]
    # spans come from whisper's alignment and are never round-tripped through the LLM
    assert [(s.start, s.end) for s in out] == [(0.0, 1.0), (1.0, 2.0)]


def test_batches_respect_batch_size():
    t = translator([reply("Hello."), reply("How are you?")], batch_size=1)
    out = t.translate(SENTENCES, "fr")

    assert len(t.client.prompts) == 2
    assert [s.text for s in out] == ["Hello.", "How are you?"]


def test_missing_entries_fall_back_to_per_sentence():
    # batch reply covers only the first sentence; the second is retried alone
    t = translator([reply("Hello."), reply("How are you?")])
    out = t.translate(SENTENCES, "fr")

    assert [s.text for s in out] == ["Hello.", "How are you?"]
    assert len(t.client.prompts) == 2


def test_malformed_batch_reply_does_not_lose_the_file():
    t = translator(["not json at all", reply("Hello."), reply("How are you?")])
    out = t.translate(SENTENCES, "fr")

    assert [s.text for s in out] == ["Hello.", "How are you?"]


def test_unrecoverable_sentence_is_dropped_not_fatal():
    t = translator(["garbage", "garbage", reply("How are you?")])
    out = t.translate(SENTENCES, "fr")

    # the first sentence is lost, the rest of the file survives
    assert [s.text for s in out] == ["How are you?"]


def test_out_of_range_ids_are_ignored():
    # a reply that renumbers entries beyond the batch must not be mapped onto
    # whatever sentence happens to sit at that index
    renumbered = json.dumps({
        "translations": [
            {"id": 1, "text": "Hello."},
            {"id": 99, "text": "Belongs to nothing."},
        ]
    })
    t = translator([renumbered, reply("How are you?")])
    out = t.translate(SENTENCES, "fr")

    assert [s.text for s in out] == ["Hello.", "How are you?"]


def test_empty_input_makes_no_requests():
    t = translator([])
    assert t.translate([], "fr") == []
    assert t.client.prompts == []


# --- JSON extraction ------------------------------------------------------


def test_extract_plain_json():
    assert _extract_json('{"translations": []}') == {"translations": []}


def test_extract_json_wrapped_in_prose():
    raw = 'Sure! Here you go:\n```json\n{"translations": [{"id": 1, "text": "Hi"}]}\n```'
    assert _extract_json(raw)["translations"][0]["text"] == "Hi"


def test_extract_spans_to_the_last_brace_not_the_first():
    # model-multilingual-stt matches to the FIRST '}', which truncates any reply
    # containing nested objects into invalid JSON and loses the whole file
    raw = '{"translations": [{"id": 1, "text": "Hi"}, {"id": 2, "text": "There"}]}'
    assert len(_extract_json(raw)["translations"]) == 2


def test_extract_raises_without_json():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("I'm sorry, I can't help with that.")
