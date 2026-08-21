"""Path C: translate the sentence track to English with an LLM.

Timestamps are never round-tripped through the model. Each source sentence
already carries its own [start, end] from whisper's word alignment, so we
translate the text and keep the span.

Batches are id-tagged so a reply can be validated per sentence; anything the
batch reply misses is retried individually, and a sentence that still fails is
dropped with a warning rather than taking the file down with it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from .sentences import Sentence

SYSTEM_PROMPT = (
    "You are a translation engine. Translate each numbered line into English. "
    "Preserve meaning and register; do not summarize, merge, split, or add commentary. "
    "If a line is already English, repeat it unchanged. "
    'Reply with JSON only, in the form {"translations":[{"id":1,"text":"..."}]}, '
    "with exactly one entry per input line."
)


@dataclass(frozen=True)
class TranslatorConfig:
    host: str = "http://ml-004.eluvio:11434"
    model: str = "llama3.3:70b"
    batch_size: int = 20
    timeout: int = 300


class LLMTranslator:
    def __init__(self, cfg: TranslatorConfig, client=None):
        self.cfg = cfg
        if client is None:
            # imported lazily so images without translation configured, and tests
            # that inject a stub client, need not install ollama
            import ollama

            client = ollama.Client(host=cfg.host, timeout=cfg.timeout)
        self.client = client

    def translate(self, sentences: List[Sentence], source_language: Optional[str]) -> List[Sentence]:
        """Translate sentence text to English, carrying each span through unchanged."""
        if not sentences:
            return []

        out: List[Sentence] = []
        for start in range(0, len(sentences), self.cfg.batch_size):
            batch = sentences[start : start + self.cfg.batch_size]
            translated = self._translate_batch(batch, source_language)
            for idx, sentence in enumerate(batch):
                text = translated.get(idx)
                if text is None:
                    text = self._translate_one(sentence, source_language)
                if text is None:
                    logger.warning(f"dropping untranslatable sentence at {sentence.start:.1f}s")
                    continue
                out.append(Sentence(start=sentence.start, end=sentence.end, text=text))
        return out

    def _translate_batch(
        self, batch: List[Sentence], source_language: Optional[str]
    ) -> Dict[int, str]:
        numbered = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(batch))
        src = f" from {source_language}" if source_language else ""
        prompt = f"Translate the following {len(batch)} lines{src} to English.\n\n{numbered}"

        try:
            raw = self._generate(prompt)
            payload = _extract_json(raw)
            entries = payload["translations"]
        except Exception as e:
            logger.warning(f"batch translation failed ({e}); falling back to per-sentence")
            return {}

        result: Dict[int, str] = {}
        for entry in entries:
            try:
                # ids are 1-based in the prompt
                idx = int(entry["id"]) - 1
                text = str(entry["text"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(batch) and text:
                result[idx] = text

        missing = len(batch) - len(result)
        if missing:
            logger.warning(f"batch reply covered {len(result)}/{len(batch)} sentences")
        return result

    def _translate_one(self, sentence: Sentence, source_language: Optional[str]) -> Optional[str]:
        src = f" from {source_language}" if source_language else ""
        prompt = (
            f"Translate this single line{src} to English. Reply with JSON only, "
            f'in the form {{"translations":[{{"id":1,"text":"..."}}]}}.\n\n1. {sentence.text}'
        )
        try:
            payload = _extract_json(self._generate(prompt))
            text = str(payload["translations"][0]["text"]).strip()
            return text or None
        except Exception as e:
            logger.warning(f"per-sentence translation failed: {e}")
            return None

    def _generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.cfg.model,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            stream=False,
            format="json",
            options={"seed": 1, "temperature": 0.0},
        )
        return response["response"]


def _extract_json(raw: str) -> Dict:
    """Parse a JSON object out of a model reply.

    Tries the whole string first (ollama's format="json" usually delivers clean
    JSON), then falls back to the outermost braces."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
    return json.loads(match.group(0))
