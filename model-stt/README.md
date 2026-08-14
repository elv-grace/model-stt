# model-stt

Whisper speech-to-text as an Eluvio tagger, implementing `common_ml.tagging.models.av.AVModel`.

## Paths

| Path | Config | What it does |
|---|---|---|
| **A** | `profile=default` — `large-v3-turbo`, `task=transcribe` | Fast multilingual ASR. ~8x faster than `large-v3`, minimal WER cost. |
| **B** | `profile=translate_whisper` — `large-v3`, `task=translate` | Whisper's native X→English speech translation. |
| **C** | `profile=translate_llm` — `large-v3-turbo` + `translator=llm` | Turbo transcription, then LLM translation of the sentence track. Emits source **and** English. |
| prod | `profile=production` — `faster-whisper`, `large-v3-turbo` | Path A on the CTranslate2 runtime. |

**`large-v3-turbo` cannot translate.** It was distilled without the translation task and returns
source-language text if asked to translate. A `task=translate` + `translator=whisper` request on turbo
is **automatically upgraded** to `translate_fallback` (`large-v3`) with a warning — silently-wrong
output is worse than a slower model. The model that actually ran is on `WhisperSTT.effective_model_name`
and in each sentence tag's `additional_info["model"]`.

## Output tracks

| Track | Contents |
|---|---|
| `""` | Word-level tags. Source language (A, C) or English (B). |
| `"auto_captions"` | Sentence-level tags, same language as the word track. |
| `"translation"` | English sentence-level tags. Path C only. |

Every tag carries `additional_info["language"]`, so a consumer never has to infer a track's language
from the config that produced it. Timestamps are milliseconds relative to the file passed to `tag()`.

Word timings are always computed — the sentence track needs them to place its boundaries.
`word_level: false` suppresses word *tags*, it does not disable word *timing*.

## Weights

Nothing is downloaded at container start. Populate the local cache once:

```bash
python download_weights.py                                   # all 4 combinations, 9.4 GB
python download_weights.py --models large-v3-turbo --backends ct2   # 1.62 GB
```

| | openai-whisper | faster-whisper (CT2) |
|---|---|---|
| `large-v3-turbo` | 1.62 GB | 1.62 GB |
| `large-v3` | 3.09 GB | 3.09 GB |

Cache layout is `<weights_dir>/openai/*.pt` and `<weights_dir>/faster-whisper/<model>/`, defaulting to
`~/.cache/model-stt/whisper`. `WEIGHTS_DIR` overrides it, which is how a baked image points at its
in-image copy without a config edit.

Two loading subtleties worth not rediscovering:

- **openai-whisper is loaded by *name*, not path.** `whisper.load_model()` only attaches DTW alignment
  heads when handed a registered model name; handed a file path it sets `alignment_heads=None` and word
  timestamps silently degrade. The staging directory travels separately as `download_root`, where
  `_download()` sha256-checks the pre-staged file and skips the network.
- **faster-whisper is loaded by explicit path or repo id, never a size name.** Size names resolve
  through a repo mapping that changes between library versions, so `large-v3-turbo` would not be
  reproducible across upgrades.

## Building

```bash
WEIGHTS=turbo-ct2 ./build.sh     # production, CT2 only    (~1.6 GB weights, no torch)
WEIGHTS=full-ct2  ./build.sh     # production + translation (~4.7 GB weights, no torch)
WEIGHTS=all       ./build.sh     # benchmark, both backends (~9.4 GB weights, with torch)
WEIGHTS=none      ./build.sh     # mount the cache at run time instead
```

`WEIGHTS` also selects the pip extras: anything but `all` installs the CT2 backend only and skips the
torch/nvidia wheel stack entirely, which is most of the image size.

**If an image serves translation, `large-v3` must be in its baked set** — otherwise the automatic
upgrade triggers a 3.09 GB download at load time, which fails outright offline. `full-ct2` covers this.

## Running

```bash
# baked weights
podman run --rm --network host --device nvidia.com/gpu=0 \
  --volume=$(pwd)/test-files:/elv/test-files:ro --volume=$(pwd)/tags:/elv/tags \
  model-whisper-stt --output-path /elv/tags/out.jsonl --params '{"profile":"production"}'

# WEIGHTS=none, cache mounted
podman run --rm --network host --device nvidia.com/gpu=0 \
  --volume=$HOME/.cache/model-stt:/root/.cache/model-stt \
  --volume=$(pwd)/test-files:/elv/test-files:ro --volume=$(pwd)/tags:/elv/tags \
  model-whisper-stt --output-path /elv/tags/out.jsonl
```

Input file paths arrive on stdin; `--params` takes any `RuntimeConfig` field, plus `profile` to pick a
base profile from `config.yml`.

## Testing

```bash
make test                              # 39 unit tests, no weights or GPU needed
pytest -m weights tests/               # end-to-end against real weights
```

Unit tests run against a `FakeBackend`, so track assignment, sentence grouping, the translation-model
substitution, hallucination filtering and JSON recovery are all covered without a GPU.

## Measured behaviour

Measured on 10 fixtures / 21.8 min of audio (en, fr, ko, zh, ta), `large-v3-turbo`, one L40S:

| | load | wall | RTF | peak GPU |
|---|---|---|---|---|
| openai-whisper | 9.1s | 83.2s | 15.7x | 5772 MiB |
| faster-whisper | 1.8s | 36.9s | **35.4x** | 7894 MiB |

faster-whisper is ~2.3x faster end to end and ~5x faster to load, for ~37% more GPU memory. Tag
counts agree closely between the two (120/120, 87/87, 532/532, 62/62).

All five systems over the same 10 fixtures (21.8 min, 5 languages):

| System | Path | Model | Load | Wall | RTF | peak GPU |
|---|---|---|---|---|---|---|
| `turbo-ct2` | A | large-v3-turbo | 3.0s | 38.5s | **34.0x** | 2556 MiB |
| `turbo-openai` | A | large-v3-turbo | 11.4s | 80.6s | 16.2x | 5774 MiB |
| `translate-ct2` | B | large-v3 | 1.2s | 177.6s | **7.4x** | 9590 MiB |
| `translate-llm` | C | turbo + llama3.3:70b | 0.4s | 347.5s | 3.8x | 12368 MiB |
| `translate-openai` | B | large-v3 | 14.0s | 413.6s | 3.2x | 10248 MiB |

Translation costs roughly **4.5x** the throughput of transcription (37.1x → 8.2x): `large-v3` is not
turbo, and there is no turbo option for path B. Path C is slower still (6.1x, LLM round trip
included) but is the only path that emits **both** the source-language transcript and the English
translation — path B replaces the transcript with English. Path C also keeps the GPU footprint of
turbo, moving the cost to the ollama host.

Path C preserved every sentence in testing (3/3, 3/3, 17/17 translated, no drops) with spans
identical between the source and translation tracks, since timestamps never reach the LLM.

### Quality (FLEURS)

Scored with `bench/fetch_fleurs.py` + `bench/score.py` on the FLEURS test split (39 fr / 37 ko /
40 zh utterances). FLEURS is n-way parallel, so the English transcript for a sentence id doubles as
the X→English translation reference.

**Transcription — `large-v3-turbo`, both backends:**

| | fr | ko | zh |
|---|---|---|---|
| openai-whisper | WER 6.46% / CER 2.64% | WER 13.53% / CER 3.77% | CER 9.44% |
| faster-whisper | WER 6.46% / CER 2.64% | WER 13.53% / CER 3.77% | CER 9.44% |

Not a copy-paste error: **116/116 utterances are byte-identical after normalization.** On short,
clean audio at temperature 0 the two runtimes converge on the same tokens and differ only in timing
metadata — so faster-whisper's ~2.3x speed win is free here. They *do* diverge on longer, messier
material (216 vs 206 tags on a 162s song), so this is not a universal guarantee.

WER is reported as n/a for Chinese: the script has no whitespace word boundary, and FLEURS ships its
Chinese references space-separated per character. Scoring naively gives ~100% WER and ~50% CER on a
near-perfect transcription; `score.py` strips whitespace for these scripts and reports CER only.

**Translation to English:**

Scored on the file intersection all three produced (fr 39, ko 33, zh 33):

| | fr | ko | zh |
|---|---|---|---|
| B — whisper `large-v3`, openai | chrF2 59.68 / BLEU 31.16 | 29.42 / 9.79 | 42.46 / 14.05 |
| B — whisper `large-v3`, CT2 | chrF2 59.75 / BLEU 31.31 | 29.31 / 9.80 | 43.36 / 14.84 |
| **C — turbo + llama3.3:70b** | **66.16 / 42.05** | **58.55 / 34.63** | **59.17 / 32.12** |

Path C wins on every language, scored on the identical file intersection. The margin scales with
distance from English: +11 BLEU on French, +17 on Chinese, +26 on Korean. Whisper's translate task
was trained on X→English data dominated by European languages, and it shows.

Path C is also the only path with full coverage. Whisper's native translation silently produced **no
output at all** for 2–7 of 40 files (fr 39/39, ko 34–35/37, zh 33–35/40); path C covered 39/39,
37/37, 40/40. A translation path that drops 17% of Chinese files is a correctness problem, not a
quality one.

The two backends are statistically indistinguishable on translation quality (differences are within
noise), though they agree less exactly than on transcription — 26–35 of n utterances identical
rather than all of them.

Note openai-whisper warns `"Word-level timestamps on translations may not be reliable"` whenever
`task=translate` runs with word timestamps. That affects path B's word track only; the sentence
track takes its boundaries from punctuation.

### Against the existing taggers

**English ASR**, all four systems on the same 37 FLEURS utterances (the intersection every system
produced output for):

| System | WER | CER |
|---|---|---|
| model-asr (NeMo CTC + kenlm, **CPU**) | 7.32% | 4.69% |
| model-multilingual-stt (NeMo `blend_eu`) | 5.65% | 2.76% |
| **model-stt turbo** (either backend) | **4.36%** | **2.13%** |

Whisper turbo cuts WER ~40% against model-asr and ~23% against model-multilingual-stt. model-asr had
to run on CPU (see below), which does not affect its WER but makes any timing from it meaningless.

**French**, same 39 utterances:

| | French ASR | French → English |
|---|---|---|
| model-multilingual-stt (NeMo `blend_eu` + llama3.3:70b) | WER 9.33% / CER 4.21% | chrF2 66.25 / BLEU 40.62 |
| **model-stt path C** (turbo + llama3.3:70b) | **WER 6.46% / CER 2.64%** | chrF2 66.16 / BLEU **42.05** |
| model-stt path B (whisper native translate) | — | chrF2 59.68 / BLEU 30.79 |

Translation scores are statistically tied, which is the expected result: both feed the same
llama3.3:70b, so the LLM dominates and the ASR front-end is the only real variable. Whisper turbo
wins that comparison — ~30% lower WER, far broader language coverage (NeMo `blend_eu` is
en/de/es/fr/it/ru/hr/pl/by/ua, with no CJK at all), and higher throughput. The architecture of
model-multilingual-stt was sound; its implementation was what failed.

Two blockers found while getting the baselines to run, both worth knowing independently of this work:

- **model-asr collapses on long-form audio.** It feeds the whole file through the CTC model as one
  tensor with no chunking, so it is only usable on the ~30s segments it was built for. On an 85s
  audiobook chapter it returned 5 tags (`'Little. Until. My. Plan. Unt.'`) against whisper's 188;
  re-cut into 30s chunks it recovered to 52 tags, but only the first chunk was coherent. Both
  failures are independent of the GPU and container issues below.
- **model-asr cannot run on the current GPU fleet.** Its frozen `asr:legacy-live` image ships
  PyTorch compiled for `sm_37 sm_50 sm_60 sm_70`; an L40S is `sm_89`, so CUDA init fails with
  `no kernel image is available for execution on the device`. It also cannot ingest WAV or MP3 —
  `src/audio.py` hardcodes `f='mov,mp4,m4a,3gp,3g2,mj2'`, so anything outside the faststart-encoded MP4 family fails
  with `moov atom not found` — and it pulls a 2.24 GB punctuation model at runtime unless
  `~/.cache` is mounted, to recover punctuation whisper emits natively.
- **model-multilingual-stt needed three fixes beyond the common-ml API port** to run on NeMo 3.x:
  its checkpoint's `AggregateTokenizer` asserts on a missing `lang` and `transcribe()` has no
  language argument (so the language must travel in a manifest), and `channel_selector="average"`
  is rejected by the new lhotse dataloader.

### large-v3 vs large-v3-turbo for transcription

On FLEURS the two are close: `large-v3` is better on English (WER 4.08% vs 4.33%, CER 1.67% vs
2.08%) and Chinese (CER 8.65% vs 9.44%), turbo is marginally better on French, and turbo's apparent
3.3pp Korean win shrinks to 1.1pp once a single `large-v3` repetition hallucination is excluded.
`large-v3` costs 2.6x the wall time (13.1x vs 34.0x RTF on CT2) and 1.7x the memory.

On real media (`test-files/`, no references — see `bench-output/turbo-vs-large-v3.md` to read the
transcripts) they diverge much more, in both directions, and `large-v3` misbehaves in ways turbo
does not:

- **They disagree on the language of the news fixture** — turbo detects Tamil, `large-v3` detects
  Malayalam, and `large-v3` then switches to Gurmukhi script partway through and repeats the same
  fragment at 151s and 181s. Both cover only ~a third of the file; neither output is usable, but
  only `large-v3` is unstable within a single file.
- **`large-v3` sometimes drops punctuation and casing** (`6.m4a`: "don't be ridiculous come along"),
  which collapses the sentence track from 16 segments to 3, since boundaries are read off
  punctuation. turbo punctuated the same audio correctly.
- `large-v3` covers slightly more of the long audiobook chapter (257.7s vs 233.5s of 308s) but
  repeats more sentences, and gets a verifiable line wrong that turbo gets right ("boa constrictor
  **and** the act of" vs "**in** the act of").

So turbo remains the right default for transcription. `large-v3` is required only for path B
translation, which turbo cannot do at all.

Three further findings worth not rediscovering:

- **Carried context (`carry_context`) is off by default because it can destroy output.** Two
  independent failure modes, both reproducing identically on *either* backend. (1) *Length*: a
  ≥200-char `initial_prompt` made whisper return **zero** segments for a file that yields 87 tags
  without one; ≤100 chars was safe across both fixtures and backends. (2) *Contamination*: a Korean
  tail carried into a French song produced looping `'다음 영상에서 만나요'` hallucinations and lost
  **88% of the words** (204 → 24). Language detection still reported `fr` — the prompt overrode the
  audio. Both are now guarded (length cap, sentence-aligned tail, re-decode on empty output or on a
  language change), and `reset_context()` exists for batch jobs walking unrelated assets.
- **Decoding files independently is not the loss it sounds like.** Chunked vs whole-file transcript
  length ranged 75%–166% (median ~112%) across four files: whole-file decoding has its own
  suppression failure mode, so chunking sometimes *gains* content. Do not add cross-file buffering
  without measuring the specific asset type first.
- **faster-whisper's word alignment crashes on sung audio.** `find_alignment()` raises `IndexError`
  on full 30s music chunks; the backend now catches it and re-decodes without word timestamps,
  degrading to segment-level timings rather than losing the file. openai-whisper produces word
  timings on the same chunks — so on music-heavy content CT2 trades word-level granularity for speed.

`large-v3-turbo` is also visibly weak on low-resource languages: the Tamil news fixture is the
slowest and lowest-agreement file in the set. Check `large-v3` before shipping turbo for those.

## Benchmarking

```bash
python -m bench.run_bench --files test-files/*.m4a --systems turbo-openai turbo-ct2
python -m bench.score --hyp bench-output/turbo-ct2.jsonl --ref refs/ --task asr
```

`run_bench` reports real-time factor with model load timed separately, plus per-process GPU memory
(via `nvidia-smi`, since `torch.cuda.max_memory_allocated` sees nothing for the CT2 backend).

`score` reads the common-ml `.jsonl` message format, so it scores `model-asr` and
`model-multilingual-stt` output identically. Both sides are normalized with whisper's own
`EnglishTextNormalizer` — without it, WER against `model-asr` mostly measures the fact that whisper
emits punctuation and `model-asr` does not.

`model-asr` and `model-multilingual-stt` run in containers with incompatible dependency stacks and are
not driven from `run_bench`; run them separately and score their output with `bench/score.py`.
