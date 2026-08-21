# model-stt

Multilingual speech-to-text as an Eluvio tagger, implementing
`common_ml.tagging.models.av.AVModel`.

**Transcription only**: faster-whisper (CTranslate2) running `large-v3-turbo`, which detects the
spoken language per file and transcribes in that language. Translation is out of scope — see
[restoring translation](#restoring-translation).

The three configurations that were evaluated, and why this one:

| | Choice | Why |
|---|---|---|
| Model | `large-v3-turbo` | Within 0.25pp WER of `large-v3` on English, ahead on fr/ko, 2.6x faster, and more stable on real media (`large-v3` switched script mid-file and dropped punctuation). |
| Backend | faster-whisper | Byte-identical text to openai-whisper on 116 clean FLEURS utterances, ~2.3x faster, 2.3x less GPU (2556 vs 5774 MiB), and the only one with a VAD or a hookable fallback — so the `clean-audio` and deterministic-retry paths exist only there. (Keeping torch out of the image was a fourth reason until [punctuation restoration](#punctuation-restoration) brought it back; the others stand.) |
| Translation | none shipped | Out of scope. When it was measured, transcribe-then-LLM beat whisper's native translation decisively (BLEU 42 vs 31 fr, 34.6 vs 9.8 ko, 32.1 vs 14.8 zh). |

## Output tracks

| Track | Contents |
|---|---|
| `""` | Word-level tags, one per word. |
| `"auto_captions"` | Sentence-level tags. |

A caption ends at terminal punctuation (`. ? ! 。 ？ ！`, with abbreviation and initial exceptions so
`"Mr. Anthony Eden"` stays one tag) — the same primary rule as model-asr's `_merge_to_sentences`.
Whisper's own segments are decoder windows, not sentences, so the sentence track is rebuilt from word
timings rather than used as-is.

That rule is only as good as the punctuation it reads, and whisper's is not reliable enough to build
on — so the punctuation is **re-decided by a separate model** over the whole file's words before the
track is cut. Every caption is punctuated and every caption starts with a capital, which is what makes
the boundaries safe to hang speaker labels off downstream. See
[punctuation restoration](#punctuation-restoration).

**A pause never ends a caption.** An earlier version also split on any silence over
`postprocessing.sentence_gap`, which tore punctuated sentences apart whenever the pause inside them
was genuine — whisper merged an `"Oh,"` at 32.24s with a `"my God."` at 38.48s into one segment and
the silence rule emitted them as two tags. Speakers pause mid-sentence, and with VAD off the
timestamps either side of a pause are true, so there is nothing to repair. Equivalently: rather than
splitting on a pause and then merging back every fragment that does not end in terminal punctuation,
the split is never made.

Two backstops cover the cases where punctuation cannot be trusted:

| | Applies to | Behaviour |
|---|---|---|
| `postprocessing.max_caption_words` (150) | Whisper's unpunctuated run-on decode (below) | Split at the widest internal pause, repeatedly, until every piece fits |
| `postprocessing.sentence_gap` (5000ms) | A trailing run reaching the end of the input without ever terminating — possibly a genuine unfinished thought rather than a failure | The original dropped-full-stop rule: whisper sometimes omits a full stop, and without one the accumulator runs on into the *next* utterance, so a long silence ends the caption instead |

`max_caption_words` is only the *trigger*; cuts always land on pauses, so this stays pause-based
segmentation with the threshold found adaptively. A fixed threshold cannot work — the worst observed
caption's widest internal pause was 3.4s, under any sane fixed value, so it would never have split.
Such a cut can land mid-sentence, which is unavoidable once the text has no sentence boundaries left
to respect. 150 is chosen because no single spoken sentence runs that long in any language, so
anything above it is a decode failure by construction; it is deliberately *not* tuned to the fixture
set, where it would sit near 100. It is a safety valve — if it fires often, something upstream is wrong.

Restricting `sentence_gap` to the trailing run gives up the mid-file case it was written for: a
dropped full stop between two utterances 39s apart now yields one caption spanning 39s. That is
deliberate — without punctuation it is indistinguishable from a real sentence containing a long
pause, which is the `"Oh, my God."` case above, and trusting punctuation means trusting it both ways.

**The unpunctuated run-on decode is a model behaviour, not a pipeline fault.** This README already
noted it for `large-v3` (*"sometimes drops punctuation and casing... collapses the sentence track from
16 segments to 3"*); it happens on turbo too. `NBAallstar.mp4` once produced a single **1595-word**
lowercase caption this way, and `NFL.mp4` still does — *"earlier in the game the way this defense is
playing Stafford gonna go deep to..."*, a 62s unpunctuated run. Every plausible cause has been
eliminated by measuring the word tags in both instances:

| signal | value in the run-on region | rules out |
|---|---|---|
| `temperature` | **0.0** on every word | the temperature fallback (that file has 82 words above 0 elsewhere) |
| `condition_on_previous_text` | `False` in the run that produced it | prompt feedback |
| `compression_ratio` | 1.61–1.76 | a repetition loop |
| `no_speech_prob` / `avg_logprob` | 0.0 / −0.09 to −0.31 | low confidence — the decoder is *sure* |

So it is a first-pass greedy decode that whisper is confident about and simply does not punctuate.
Nothing downstream can detect it from the decoder's own signals, which is exactly why the backstop is
structural rather than confidence-based.

> **Residual case, not handled.** Captions are bounded by word count, not span, so a *short*
> unpunctuated run can still stretch a long way: `equalizer.mp4` yields an 82-word caption spanning
> **303s** over sung lyrics. Bounding by span instead would re-tear `"Good luck, guys."` (3 words,
> 207s), which is the tradeoff this design deliberately takes — punctuation is trusted over duration.

Every tag carries `additional_info["language"]`, so a consumer never has to infer a track's language
from the config that produced it. Timestamps are milliseconds relative to the file passed to `tag()`.

Word tags also carry the decoder's own signals (`probability`, `avg_logprob`, `no_speech_prob`,
`compression_ratio`, `temperature`); sentence tags carry `min_word_probability` and
`mean_word_probability`, aggregated from the words the sentence was built from, so a caption can be
weighed without joining back to the word track. The keys are **absent**, not null, when whisper
returned no word timings, so "not measured" stays distinguishable from "measured and low". Read them
as *lexical* confidence — a low minimum marks a word the model guessed at, typically a proper noun.
They are **not** a hallucination score; see [recall and hallucination](#recall-and-hallucination-what-vad-costs).

Word timings are always computed — the sentence track needs them to place its boundaries.
`word_level: false` suppresses word *tags*, it does not disable word *timing*.

## Restoring translation

Everything is commented out rather than deleted. Search for `DISABLED (translation)` in
[src/model.py](src/model.py), plus:

- [config.yml](config.yml) — the `large-v3` model entry, `translate_fallback`, the
  `translate_whisper` profile, and the `llm` section
- [setup.py](setup.py) — move `ollama` from the `translate` extra back into `install_requires`
- [run.py](run.py) — the `TranslatorConfig` import and the two constructor arguments
- [src/translate.py](src/translate.py) — intact, just unused
- `tests/test_model.py` — the path B/C tests were removed rather than commented (they no longer
  compile against `RuntimeConfig`); they are in git history

Path C also reintroduces a dependency the transcription-only image does not have: a reachable ollama
host. When it was unreachable during testing, translation produced **no output and no error** —
every sentence was dropped with only a log warning. Restoring path C should come with a loud failure
when the whole file fails to translate.

## Weights

Nothing is downloaded at container start. Populate the local cache once:

```bash
python download_weights.py                              # what the image needs, 1.62 GB
python download_weights.py --backends openai ct2        # adds the bench checkpoints
```

Two loading subtleties:

- **openai-whisper is loaded by *name*, not path.** `whisper.load_model()` only attaches DTW alignment
  heads when handed a registered model name (passing a file path sets `alignment_heads=None` and degrades word
  timestamps). The staging directory travels separately as `download_root` (`_download()` sha256-checks the pre-staged file and skips the network).
- **faster-whisper is loaded by explicit path or repo id, never a size name.** Size names resolve
  through a repo mapping that changes between library versions, so `large-v3-turbo` would not be
  reproducible across upgrades.

## Building

```bash
WEIGHTS=turbo-ct2 ./build.sh     # the shipped image (~1.6 GB decoder + 2.2 GB punctuation model)
WEIGHTS=none      ./build.sh     # mount the cache at run time instead
```

`WEIGHTS` also selects the pip extras: anything but `all` installs the CT2 backend only. The `full-ct2` and `all` presets stage `large-v3` and/or the openai checkpoints for `bench/`; neither is needed by this image, and
both require the corresponding `config.yml` entries to be uncommented first.

## Running

```bash
# baked weights
podman run --rm --network host --device nvidia.com/gpu=3 \
  --volume=$(pwd)/test-files:/elv/test-files:ro \
  --volume=$(pwd)/tags:/elv/tags \
  model-whisper-stt --output-path /elv/tags/out.jsonl

# WEIGHTS=none, cache mounted
podman run --rm --network host --device nvidia.com/gpu=3 \
  --volume=$HOME/.cache/model-stt:/root/.cache/model-stt \
  --volume=$(pwd)/test-files:/elv/test-files:ro \
  --volume=$(pwd)/tags:/elv/tags \
  model-whisper-stt --output-path /elv/tags/out.jsonl
```

Input file paths arrive on stdin; `--params` takes any `RuntimeConfig` field, plus `profile` to pick a
base profile from `config.yml`.

Two profiles ship:

| Profile | VAD | `condition_on_previous_text` | For |
|---|---|---|---|
| `default` | **off** | off | Everything by default. Decode all the audio and let the decoder's own guards filter. Output is reproducible — see [deterministic retries](#deterministic-retries-on-by-default). |
| `clean-audio` | on, `threshold 0.25` / `speech_pad 2000ms` | on | Content with *distinguishable silence* — scripted film, studio recordings — where suppressing stock-phrase artifacts is worth more than recall. |
| `stochastic` | inherits | inherits | Whisper's original sampled retry ladder. Not reproducible; 5–32% slower. Kept as the upstream behaviour. |
| `reproducible` | inherits | inherits | Empty alias, kept so existing callers keep working. The default is already reproducible. |

The two fields in each profile are a pair, not independent knobs. With VAD off the decoder sees noise
it used to be shielded from, and a *conditioned* decode locks onto a phrase and repeats it; turning
conditioning off is what prevents that. Do not flip one without the other — there is a unit test
guarding the coupling.

VAD-off is the default because VAD fails *silently*: audio it discards never reaches the decoder, so
over-filtering leaves no low-confidence segment and no repetition signal behind, only missing time.
See [recall and hallucination](#recall-and-hallucination-what-vad-costs).

## Testing

```bash
make pytest                # 113 unit tests, no weights or GPU needed
pytest -m weights tests/   # end-to-end against real weights
make test                  # container end-to-end (buildscripts/testers/test-model.sh)
```

`make test` is the standard tagger-model container test from `buildscripts`. It transcribes every
file **directly in** `test-files/` (subdirectories are excluded by its `-maxdepth 1`) and does
`rm -rf test-output/` first — that is now the 13 feature films, so budget ~18 minutes.

Unit tests run against a `FakeBackend`, so track assignment, sentence grouping, hallucination and
degenerate-segment filtering, and the carried-context guards are all covered without a GPU.

## Measured behaviour

Measured on 10 fixtures / 21.8 min of audio (English en, French fr, Korean ko, Chinese zh, Tamil ta) on one L40S. Paths B and C are no longer shipped.

faster-whisper is ~2.3x faster end to end and ~5x faster to load than openai-whisper, and their tag
counts agree closely (120/120, 87/87, 532/532, 62/62).

| System | Path | Model | Load | Wall | RTF | peak GPU |
|---|---|---|---|---|---|---|
| `turbo-ct2` | A | large-v3-turbo | 3.0s | 38.5s | **34.0x** | 2556 MiB |
| `turbo-openai` | A | large-v3-turbo | 11.4s | 80.6s | 16.2x | 5774 MiB |
| `translate-ct2` | B | large-v3 | 1.2s | 177.6s | **7.4x** | 9590 MiB |
| `translate-llm` | C | turbo + llama3.3:70b | 0.4s | 347.5s | 3.8x | 12368 MiB |
| `translate-openai` | B | large-v3 | 14.0s | 413.6s | 3.2x | 10248 MiB |

Translation costs roughly **4.5x** the throughput of transcription (37.1x → 8.2x): 
`large-v3` is not turbo, and there is no turbo option for path B. Path C is slower still (6.1x, LLM round trip included) but is the only path that emits **both** the source-language transcript and the English translation — path B replaces the transcript with English. 
Path C also keeps the GPU footprint of turbo, moving the cost to the ollama host.

Path C preserved every sentence in testing (3/3, 3/3, 17/17 translated, no drops) with spans
identical between the source and translation tracks, since timestamps never reach the LLM.

### Punctuation restoration

Whisper has no punctuation stage. `.`, `,` and capital letters are ordinary tokens its text decoder
predicts, and *which convention* it predicts them in is inferred per 30-second window from whatever
training data the audio resembles. Over the 11,648 captions of the fixture set that fails three ways:

| Failure | Example | |
|---|---|---|
| lyric mode | equalizer 7849.06s | `"pray to God, I just opened enough eyes later on Gave you the supplies and the tools to hopefully use that'll make you strong Enough to lift yourself up…"` — 135 words, **one caption** |
| style lapse | spiderman-across 8133.42s | `"is but nicknamed mr keenan do the most is i was living down bad in my folks crib now i'm laughing to the bank and the joke is…"` — 134 words, entirely lowercase |
| run-on | NFL 731.78s | `"You have to deal with the run game as surprised early to see him give up that long one to Nakua Hooker takes a rest here on a second and ten…"` — 142 words, cased but never terminated |

Because a caption ends at punctuation, a window with none becomes a single caption holding many
utterances. 81 captions (0.7%) ended with no terminal mark at all, holding 1,909 words (2.1%); 187
(1.6%) opened on a lowercase letter. Those percentages are small and the impact is not — a caption is
the unit a speaker gets assigned to, so one 142-word caption spanning four speakers is a worse failure
than its share of captions suggests.

[src/punctuate.py](src/punctuate.py) re-decides punctuation with the same token-classification model
model-asr uses (`oliverguhr/fullstop-punctuation-multilang-large`), run over the file's **flat word
stream** with no segment boundaries in view. Measured across the seven fixtures, on identical decodes:

| | captions | words | unpunctuated | words in them | lowercase-start |
|---|---|---|---|---|---|
| whisper's own punctuation | 11,648 | 89,715 | 81 | 1,909 | 187 |
| restored | 13,565 | 89,717 | **0** | **0** | **0** |

Every caption punctuated and capitalised, on every file, with the word count unchanged — this only
moves characters attached to words, never which word sits where, so every timing is untouched. The
16.5% rise in captions is the run-ons being cut into the sentences they always contained.

**It never changes case downward, and it never overrules whisper on a sentence *end*.** Two deliberate
asymmetries, both of which cost accuracy if reversed:

- The model labels punctuation only, so `"Antetokounmpo"` and `"Wesby's Village"` survive a round trip
  that a truecasing model would have to re-derive. Capitals are *added* at sentence starts and nowhere
  else. The visible cost is that lyric-mode captions keep their line-initial capitals mid-sentence.
- Where whisper ended a sentence, it keeps whisper's mark. The label set is `0 . , ? - :` — no `!`, no
  CJK — so overwriting would flatten `"Get out of here!"` to `"Get out of here."` and `"好。"` to `"好."`.

#### Why it may not remove a full stop

Whisper decides punctuation per *segment* and closes a segment at a pause, so a mid-sentence breath
becomes a full stop. The obvious repair is to let the model delete one. `demote_terminal` does exactly
that and is **off**, because measuring it showed the cure is worse than the disease: on
spiderman-into it removed 21 boundaries, overwhelmingly real ones.

```
'Melissa.' + 'Richard.'      -> 'Melissa, Richard.'     two speakers merged
'Spider-Man!' + 'Hands up!'  -> one caption
'Dr.' + 'Olivia Octavius.'   -> 'Dr Olivia Octavius.'   abbreviation destroyed
```

It is not a tuning problem. Ranked by confidence, the **worst merges are the most confident**:
`"What?"|"Is"` at 0.9998, `"Mrs."|"Grady"` at 0.9997, `"Dr."|"Olivia"` at 0.9994. Raising the bar keeps
those and discards only the marginal ones; lowering it merges more. And the case it was built for is
not even a candidate — over hollywood 3330–3341s, where whisper splits `"but it's the pursuit."` from
`"It's meaningful."`, the model agrees with whisper's full stop and proposes no demotion at all. There
is no benefit to weigh against the harm. The model is trained on written prose, where two consecutive
three-word exclamations are rare, so it reports no boundary exactly where fast dialogue puts one.

#### What it does not fix: `hallucination_silence_threshold`

Punctuation restoration removes caption *quality* from the missing-time tradeoff, but not the missing
time. faster-whisper's `hallucination_silence_threshold` fires on `is_segment_anomaly` — unusually
long, short or improbable words — and when such a segment is bracketed by quiet it moves the decoder's
**seek pointer past the audio entirely**. Nothing about the skip is visible downstream: no
low-confidence segment, no repetition, just a gap. Sung audio and period radio advertising score as
anomalous, which is why once-upon-a-time-in-hollywood loses car-radio commercials and songs that
model-asr transcribes correctly.

Two things follow, and they answer different halves of the question:

- **Punctuation absorbs the re-segmentation damage.** Moving seek re-segments everything after it, and
  that used to change caption quality. It no longer can: with restoration on, unpunctuated and
  lowercase-start captions are **0 at both `2.0` and `null`**. What is left is a pure content
  question — which words exist — not a boundary question.
- **It does not recover the words.** Across the seven fixtures, `null` is +117 words (+0.1%) and 25%
  faster, but that total hides per-file swings of ±226.

Swept on words recovered (deterministic decoding, so these are reproducible, not one draw):

| | 1.0 | 2.0 | 4.0 | 8.0 | off |
|---|---|---|---|---|---|
| NBAallstar (sports) | 29,179 | **29,353** | 29,136 | 29,190 | 29,128 |
| spiderman-into (animation) | 807 | **804**¹ | 780 | 759 | 759 |
| hollywood (live-action) | 13,642 | 13,809 | 13,979 | **14,084** | 14,032 |
| equalizer (live-action) | 8,035 | 8,103 | 8,213 | **8,219** | 8,097 |

¹ 1.0 returns 3 more words but 16s more uncovered time; 2.0 is the better point.

**There is no globally optimal value, and the split is not by genre** — the animated short prefers 2.0
exactly like live sports does, while the two long live-action films want 4.0–8.0. 2.0 stays the default
because it is best for two of the four and worst for none. Raise it to 4.0–8.0 for long live-action
film with a lot of score and in-world radio: +275 words and 182s less uncovered time on hollywood,
+116 words and 222s less on equalizer. The whole range spans ~1.5%, so this is a tuning knob, not a
correctness fix.

#### Cost, and slimming the punctuation model

Inference is free in practice — 0.77s for a-few-good-men's 17,189 words on GPU, against a 66.6s decode. The
cost is in the image: torch, transformers and the nvidia wheels torch pulls that CTranslate2 does not
need add **~4.25 GB**. cuBLAS and cuDNN are *not* part of that — CTranslate2 declares no CUDA
dependencies and dlopens `libcublas.so.12`/`libcudnn.so.9` off `LD_LIBRARY_PATH`, and torch pins the
very same wheels (`nvidia-cublas-cu12==12.8.4.1`, `nvidia-cudnn-cu12==9.10.2.21`), so CT2 runs on
torch's copies. Verified by running a CT2 decode and the punctuation model on one GPU in one process.

ONNX Runtime was measured as the way to drop torch again, and **only its CPU provider is viable**:

| | added to image | per file | agreement with torch |
|---|---|---|---|
| torch fp16, GPU (shipped) | ~4.25 GB | 0.5s | reference |
| ONNX fp32, CPU | ~30 MB | 29.3s | 99.983% |
| ONNX int8, CPU | ~30 MB, and 0.56 GB of weights instead of 2.2 GB | ~15s | 96.2% — real decision changes |

`onnxruntime-gpu` 1.29 requires **CUDA 13** (`libcublasLt.so.13`) against CTranslate2's CUDA 12, so a
GPU ONNX path means two CUDA majors in one image — larger than torch. Take the CPU provider if image size matters more than ~17% wall time; int8 needs a caption-level evaluation first, which has not been done.

If the model is missing or fails to load, the tagger logs a warning and falls back to whisper's own
punctuation rather than failing the file. Set `postprocessing.punctuation.required: true` to invert
that, or `enabled: false` to skip it entirely.

### Quality (FLEURS)

Scored with `bench/fetch_fleurs.py` + `bench/score.py` on the FLEURS test split (39 fr / 37 ko / 40 zh utterances). 
FLEURS is n-way parallel, so the English transcript for a sentence id doubles as the X→English translation reference.

**Transcription — `large-v3-turbo`, both backends:**

| | fr | ko | zh |
|---|---|---|---|
| openai-whisper | WER 6.46% / CER 2.64% | WER 13.53% / CER 3.77% | CER 9.44% |
| faster-whisper | WER 6.46% / CER 2.64% | WER 13.53% / CER 3.77% | CER 9.44% |

Not a copy-paste error: **116/116 utterances are byte-identical after normalization.** On short clean audio at temperature 0, the two runtimes converge on the same tokens and differ only in timing
metadata — so faster-whisper's ~2.3x speed win is free here. They *do* diverge on longer, messier
material (216 vs 206 tags on a 162s song), so this is **not** a universal guarantee.

**Measured with VAD off on both**, which is what the shipped configuration is. The equivalence does not survive noisy audio.

- `NBAallstar 1200-1800s` is ~340s of All-Star player introductions —
  transcribed accurately, down to the spellings of Antetokounmpo and Gilgeous-Alexander — followed by
  ~260s of genuine non-speech.
- Single runs cannot support the comparison. These were measured under the sampled ladder, which is
  not reproducible (see [deterministic retries](#deterministic-retries-on-by-default)), and
  repeated-segment counts vary by 2–3x run to run.

Re-measured on that slice, 3 reps under the shipped ladder, reported as min–max:

| | segments | words | repeated | segments in the *speech* half |
|---|---|---|---|---|
| openai-whisper (no VAD exists) | 39–52 | 316–431 | 11–24 | 25–28 |
| faster-whisper, vad off | 37–55 | 326–354 | 9–31 | 24–28 |
| faster-whisper, vad on `0.5/400` | **0** | 0 | 0 | **0** |
| faster-whisper, vad on `0.25/2000` | 1 | 2 | 0 | 1 |

**The two backends are indistinguishable** — every range overlaps.

At Silero's own defaults VAD emitted *zero segments* for 600
seconds containing 340s of clean announcer speech, and one segment at the tuned setting. It is total suppression and is the finding
that moved VAD out of the default (see
[recall and hallucination](#recall-and-hallucination-what-vad-costs)).

The VAD rows are reproducible; the two unfiltered rows are not, hence the ranges. Pinning
`temperature=[0.0]` makes all four reproducible but measures a different system — repetition on
`vad off` jumps to 107, because the temperature fallback *is* the repetition guard.

The backend choice rests on speed, GPU footprint, and VAD remaining *available* for the `clean-audio`
profile. openai-whisper has no VAD to turn on — it detects silence from the decoder's own
`no_speech_prob`, exactly the signal that fails when the model is confident it heard speech in music
or crowd noise. whisper's own source concedes the gap (`timing.py`: *"a better segmentation algorithm
based on VAD should be able to replace this"*). With VAD on, CT2 scores en 4.21% / fr 6.82% WER
against openai's 4.33% / 6.46%.

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

Path C wins on every language, scored on the identical file intersection. 
The margin scales with distance from English: +11 BLEU on French, +17 on Chinese, +26 on Korean. 
Whisper's translate task was trained on X→English data dominated by European languages.

Path C is also the only path with full coverage. Whisper's native translation silently produced **no
output at all** for 2–7 of 40 files (fr 39/39, ko 34–35/37, zh 33–35/40); path C covered 39/39, 37/37, 40/40. 
A translation path that drops 17% of Chinese files is a correctness problem, not a quality one.

The two backends are statistically indistinguishable on translation quality (differences are within
noise), though they agree less exactly than on transcription — 26–35 of n utterances identical rather than all of them.

Note openai-whisper warns `"Word-level timestamps on translations may not be reliable"` whenever
`task=translate` runs with word timestamps. That affects path B's word track only; 
the sentence track takes its boundaries from punctuation.

### Recall and hallucination: what VAD costs

Measured on one L40S over the original 8-file fixture set, which now lives in `test-files/movies/`
and `test-files/sports/` — `test-files/` itself holds the 13 feature films used by
[the model comparison](#choosing-between-model-stt-and-model-asr). `NBAsummerleague33min.mp4` has no
audio stream and is skipped. That set runs in **10.0 minutes**, 2556 MiB peak, producing 101,588 tags.

The central finding is that **VAD-strictness and hallucination are not one axis you can tune.**
Speech-over-loud-crowd (wants permissive VAD) and crowd-without-speech (wants aggressive VAD) are the
same thing to Silero, and in `NBAallstar.mp4` they sit *minutes apart in the same file*. So no global
threshold solves both:

| | `NBAallstar` 50s–1884s | `spiderverse` four dropped stretches |
|---|---|---|
| VAD `thr 0.5 / pad 400` | 30.5 min dropped | 0 of 4 recovered |
| VAD `thr 0.25 / pad 2000` | **30.5 min still dropped** | 4 of 4 recovered |
| VAD off | recovered | recovered |

That NBAallstar stretch is 30 minutes of arena PA announcements — decoded with VAD off
it yields 130 captions, 67 of them six words or longer. Silero scores it as non-speech at every
threshold tried. `model-asr` also emits into that window, but its output there is unusable
(`'Ye.'`, `'My kee.'`, `'For your liver.'`), so it is not recovering the content either.

**VAD off, over the whole fixture set:** +6.6% words (77,068 → 81,778), +460 substantive
(≥6 word) captions, at a cost of **32** extra artifact-shaped captions — short texts sitting alone in
a caption desert. Roughly 14:1 in favour of decoding everything. FLEURS is unchanged (en 4.33% /
2.08%, fr 6.46% / 2.64%, ko 13.53% / 3.77%, zh CER 9.44%), and it is **faster** — 102x against 95x,
because running Silero over a whole file costs more than decoding the parts it would discard.

**`condition_on_previous_text: false` is the other half and is not optional.** With VAD off the
decoder locks onto a phrase and repeats it. That loop is prompt feedback, not a silence artifact:

| decode of `NBAallstar` 1680–1740s | output |
|---|---|
| conditioned | `'Thank you, Michael.'`, `'Thank you.'` |
| unconditioned | `'Thank you, Michael.'`, `'The USA First World Tourna…'`, `'Voting will be over at the…'`, `'Now Mr. Chitton, stand up…'` |

Turning conditioning off does not merely suppress the repeats, **it recovers the speech underneath
them**. This is why there is no post-hoc repetition filter anywhere in this pipeline — see below.

#### The isolated-artifact guard

Whisper emits stock phrases over non-speech (`"Thank you."`) because they are frequent in its
subtitle training data. `_drop_isolated_artifacts` removes them, using the one signal that separates
them from real speech: **company**.

> A short text (≤3 words) that **recurs** in a file (≥4 times) **and is characteristically isolated**
> (median gap to nearest neighbour ≥10s) is that file's artifact. Only its isolated instances drop.

Over `NBAallstar`, `"Thank you."` captions sat a median **18.9s** from their nearest neighbour (36 of
55 more than 10s away), against **0.2s** for substantive captions (1588 of 1661 within 2s). Two orders
of magnitude, so the threshold is not a knife-edge. On the current fixture set it fires 7 times and
drops 81 segments of 11,850 captions (0.7%); FLEURS moves `+0.00pp` on all seven metrics, since
10-second utterances never reach `min_count=4`.

Both conditions are required, and it is deliberately **not** a list of known whisper phrases:

- **Isolation alone** deleted 16 real short lines that legitimately stand alone between long musical
  stretches — `'Whoa.'`, `'Hello.'`, `'Again, Moss?'`, `'One, two, three!'`, `'You want in?'`,
  `'George, you wake?'` — each confirmed genuine by re-decoding its audio in isolation.
- **A hardcoded phrase list** worked, but could only be assembled by reading this fixture set, and
  would not transfer to another language where whisper has different stock phrases.
- The self-calibrating rule was validated **leave-one-file-out**: the learned set stayed stable
  whichever file was held out, and no validated-real caption was dropped in any held-out file.

**Per-file calibration does real work.** The learned phrases are not a fixed
set — they differ by file, including phrases that look far too common to touch. On `equalizer.mp4`
the guard learned `'i dont know'` and `'im sorry'` and dropped 17 captions of 1439 (1.2%);
re-decoding every one of them with fresh context, **14 of 17 returned completely different text**
(`"I don't know."` → `"I'm going to go ahead and cook it."`), and 2 of the remaining 3 re-decoded into
an `"I'm sorry."` ×6 loop rather than clean speech. Meanwhile `spiderman-across-the-spiderverse`
*kept* all 14 of its `'i dont know'` captions, because there they sit in conversation and fail the
median-isolation test. A global list would have had to choose one behaviour for both files.

Disable with `isolated_artifact_gap: 0`. Note what it does **not** do: it catches one specific
failure — the recurring stock phrase over non-speech — and cannot catch a confident one-off
invention such as a wrong proper noun.

**It is well-targeted but low-recall.** Swept over the 13 films (`gap` dominates; `min_count` barely
matters):

| gap / max_words / min_count | dropped | WER |
|---|---|---|
| off | 0 | 21.97% |
| 20 / 3 / 4 | 98 | 21.85% |
| **10 / 3 / 4** (shipped) | 177 | 21.74% |
| 5 / 5 / 3 | 236 | **21.67%** |

Scored by subtitle *cue overlap*: 81% of all decoded segments sit under a cue, against only 18% of the
ones the guard drops — a 4.5× enrichment toward uncaptioned time, so the rule is picking the right
things. But it removes just 22–28% of the artifact burden (674 artifact-shaped captions in uncaptioned
time with the guard off, 525 shipped, 486 at `5/5/3`), and what survives is `"no"`, `"yeah"`, `"okay"`,
`"come on"` — indistinguishable from real speech by any text rule, because they *are* real speech
hundreds of times per film.

So `5/5/3` is a real but narrow gain: it removes 189 artifacts against 145, at 47 real captions lost
against 32, and on the original seven fixtures it also drops genuine `"let's go"`, `"oh my god"` and a
player's name. Lowering the risk further needs an audio-side signal — is there speech energy here —
not a better threshold.

#### Four things that do not work, with the evidence

- **Per-segment confidence does not detect hallucination.** Whisper is *confidently* wrong over
  noise. In the recovered NBAallstar window, `"Thank you."` words carry `no_speech_prob` **0.000**
  (zero percent above the 0.6 threshold) and `avg_logprob` −0.69, against 0.000 / −0.45 for real
  speech in the same file. `no_speech_threshold` cannot see these at all.
- **Word probability does not either.** A "3+ consecutive words below p<0.6" rule flagged **2%** of a
  100%-hallucinated sample against **9%** of clean Hollywood dialogue — anti-correlated. Read
  `min_word_probability` as marking a *guessed word* (usually a proper noun), not a fabricated one.
  In `"Hey, Mr. Sheldon, you're not crazy."` the name scored 0.50 between neighbours at 0.999, and no
  two decodes of that audio produced the same name.
- **Dropping runs of identical segments deletes real speech.** This is the obvious fix for looping
  and it is wrong: the runs sit *on top of* real audio the conditioned decode failed to transcribe
  (see the 1680–1740s table above). Fix the cause, do not delete the symptom. `src/model.py` carries a
  comment recording this so it is not re-attempted.
- **Whisper's signals cannot detect singing, so lyrics cannot be suppressed from them.** Over 38 sung
  and 276 dialogue segments on equalizer, `avg_logprob` is **higher** on the singing (−0.209 against
  −0.440), `no_speech_prob` is 0.000 for both, and sung audio never triggers the temperature fallback
  (0/38 against 30/276). Sweeping `avg_logprob` for a threshold gives TPR−FPR = **0.00**. A text-side
  rule does not work either: missing terminal punctuation is 100% sensitive but flags every style
  lapse and run-on too — the exact captions punctuation restoration exists to repair. Not worth
  building regardless: **model-asr transcribes lyrics as well** (19 and 25 captions in the same two
  windows where model-stt has 23 and 38), so this is not a difference between the systems.

#### Straddled segments

faster-whisper decodes the VAD-*concatenated* audio and maps timestamps back afterwards, resolving
each word to a speech chunk **by its midpoint** (`restore_speech_timestamps`). A word whose alignment
lands a frame either side of a chunk boundary resolves to the wrong chunk — and since a segment's
span is taken from its first and last word, one misplaced word stretches the whole segment across the
excision. `"Good luck, guys."` was emitted spanning **126.71s → 333.92s**; the phrase is really at
333.06–333.90, and the audio at 126.7s is `"What the fuck? Who did that?"`.

Worse, `to_sentences` then splits on the 207s gap, so the caption is not merely mistimed — it is torn
into `"Good"` and `"luck, guys."`.

For when VAD is on, `_repair_straddles` splits a segment's words into runs separated by more than `straddle_gap_max`,
keeps the run holding the most *spoken time*, and packs the others back against it preserving each
word's duration, clamped so they cannot overlap neighbouring segments. The example above repairs to
333.30–333.94. Threshold justification: across **4096** genuine intra-segment word gaps measured with
VAD off (where no excision is possible) over film, animation and sports, exactly **one** exceeded 2.0s.

It fired 50 times across 13.2h **with VAD on**, and is skipped entirely with VAD off. That gate is
not an optimisation. With VAD off the timestamps are already in source-media time, so every
intra-segment gap is real, and repairing one moves real speech: over the same 13.2h decoded without
VAD it fired 9 times, all false positives. On the clearest, whisper merged an `"Oh,"` at 32.24s with
a `"my God."` at 38.48s into one segment; the repair anchored on `"Oh,"` and emitted the phrase at
32.24–33.80, placing `"my God."` five seconds from where it was said. Another collapsed a segment to
zero duration.

It bounds an unbounded error rather than guaranteeing correctness — the anchor is a majority
heuristic, and it cannot repair *text* that whisper merged lossily, only the span.

#### Deterministic retries (on by default)

Whisper retries a decode that trips `compression_ratio_threshold` or `logprob_threshold` at
temperature > 0, which *samples* instead of decoding greedily — and that sampling is unseeded. So the
shipped decoder used to give different captions every run:

| | 3 runs |
|---|---|
| full temperature ladder (old default) | **varies** — 182 / 185 / 193 segments |
| `temperature=[0.0]` | **bit-identical** |

That isolates the cause. `condition_on_previous_text` is not it (off by default, still varies), nor is
float16 reduction order (it would vary at temperature 0, and does not). `ctranslate2.set_random_seed()`
has no effect, and `Whisper.generate()` has no per-call seed. `temperature=[0.0]` is not the answer
either — the fallback is a load-bearing repetition guard, and removing it took repeated segments on
one 600s slice from ~9–31 to **107**.

The fix keeps the reject-and-retry loop and makes the *retries* deterministic: escalating beam size,
patience, `repetition_penalty` and `no_repeat_ngram_size` does the same job as sampling without the
randomness. `_DeterministicFallback` in [src/backends.py](src/backends.py) proxies the CTranslate2
model and swaps each sampled rung for a deterministic one. Use `{"profile": "stochastic"}` to get the
old behaviour back.

Over 13 features / 28.24h, determinism is close to free:

| | corpus WER | wall |
|---|---|---|
| `default` (deterministic) | 21.74% | **1,059.8s** |
| `stochastic` (sampled) | 21.43% / 21.67% / 21.70% across three draws | 1,090–1,398s |

It costs ~0.1–0.3pp WER and is 5–32% faster (beam search escapes a failing decode in fewer rungs). The
argument for it is that **two sampled runs differ from each other by 5.40% WER, while the deterministic
output differs from a sampled draw by 5.78%** — determinism adds no more divergence than running the
sampled decoder twice, and it is stable.

One caveat: on deliberately fallback-heavy audio the failure mode changes shape — deterministic
retries emit more short duplicate fragments, sampled ones fewer but longer and more varied. Duplicates
are the tractable kind, since the [isolated-artifact guard](#the-isolated-artifact-guard) catches
recurring short phrases and cannot catch varied long ones. Do not disable both together.

The proxy is coupled to faster-whisper's call into CTranslate2 and should be re-checked on upgrade. If
the keyword names change it stops intervening rather than misbehaving, since it only fires when
`sampling_temperature` is present.

### Choosing between model-stt and model-asr

Two corpora, both systems scored by `bench/score.py` with the same normalizer and references.
`test-files/*.mp4` is 13 feature films / 28.24h against their subtitle tracks, copied from [Springfield! Springfield!](https://www.springfieldspringfield.co.uk/movie_script.php) and [subslikescript](https://subslikescript.com/movie/);
`test-files/hand-val-ref-set/` is 10 excerpts / 29.4 min against **verbatim transcripts written by
ear**, covering clean dialogue, overlap, noise and music.

| | | WER | MER | CER | sub | del | ins |
|---|---|---|---|---|---|---|---|
| 13 films | model-stt | **21.74%** | **19.78%** | **17.54%** | **5.72%** | **6.11%** | 9.91% |
| | model-asr | 22.38% | 20.78% | 17.63% | 6.63% | 8.06% | **7.69%** |
| 10 clips | model-stt | **11.20%** | **10.89%** | **7.56%** | **4.51%** | **3.86%** | 2.82% |
| | model-asr | 13.44% | 13.23% | 8.82% | 6.38% | 5.45% | **1.61%** |

**model-stt substitutes and deletes less; model-asr inserts less.** That is the whole difference. It
wins 9 of 13 films and 8 of 10 clips, and its margin *widens* from 0.64pp to 2.24pp when the reference
is verbatim.

**Why subtitles understate it.** WER divides by reference length, so deletions are capped at the
reference but insertions are not. Subtitles condense and omit, so a *more complete* transcript books
correct words as errors — which is why MER, whose denominator is the alignment length, is reported
beside WER. Against verbatim references model-stt's insertion rate falls 9.91% → 2.82% and WER roughly
halves: most "insertions" were real speech the subtitler dropped.

**This is not a small correction.** On equalizer the subtitle score says model-stt is 6.88pp *worse*;
its four verbatim clips say it is 2.81pp *better* (19.07% against 21.88%). The film-level penalty was
end-credits music the subtitle omits and model-stt transcribes. Read the film numbers for ranking and
the clip numbers for absolute quality.

| | load | wall | ×realtime | peak GPU | peak RSS |
|---|---|---|---|---|---|
| model-stt (28.24h) | **7.2s** | **1,059.8s** | **95.9×** | 3,952 MiB | 11,384 MiB |
| model-asr (28.24h) | 43.5s | 1,309.9s | 77.6× | 3,996 MiB¹ | **8,963 MiB** |

¹ torch `reserved`; the model-stt figure is `nvidia-smi` process total and includes the CUDA context.
Treat GPU as a tie. The 2.4 GB host-memory gap is torch plus the punctuation model.

**Recommendation: model-stt**, more accurate on every aggregate metric on both corpora and faster,
against +2.4 GB of RAM. Two things to know. Both systems collapse on music-dominant audio (bd1 136% /
95%, eq4 52% / 51%) — that is a shared limit, not a reason to prefer one. And model-stt leaves ~486–525
short recurring artifact captions in uncaptioned time; they barely move WER but each is a phantom
speaker turn downstream, which is the real regression risk.

model-asr must be run on its own hardware: its container pins torch 1.9, which has no kernels for an
L40S (`sm_89`), so every file fails with `no kernel image is available`. Score its `.jsonl` here with
`--compare`.

On FLEURS, where all three systems overlap on 37 English utterances, the ordering is the same:
model-stt turbo 4.36% WER / 2.13% CER, model-multilingual-stt 5.65% / 2.76%, model-asr 7.32% / 4.69%.
model-multilingual-stt covers en/de/es/fr/it/ru/hr/pl/by/ua with no CJK at all.

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
# backends and model sizes, on short clean fixtures
python -m bench.run_bench --files test-files/bench-files/*.m4a --systems turbo-openai turbo-ct2
python -m bench.score --hyp bench-output/turbo-ct2.jsonl --ref refs/ --task asr

# the shipped profile on full-length media, against .txt references beside each .mp4
python -m bench.run_movies --system default
python -m bench.run_movies --media-dir test-files/hand-val-ref-set --outdir test-output-hand-val-ref-set
python -m bench.run_movies --score-only --compare model-asr=/path/to/model-asr.jsonl
```

`run_bench` measures the *decoder* — real-time factor with load timed separately, plus per-process GPU
memory via `nvidia-smi` (`torch.cuda.max_memory_allocated` sees nothing for CT2). Punctuation is off
there: FLEURS is scored through a normalizer that strips punctuation, so restoring it would cost time
and move no score.

`run_movies` measures the *shipped profile* and adds MER, the substitution/deletion/insertion
breakdown, peak RSS, and `--param FIELD=VALUE` for one-off config overrides (recorded in the summary).
`--compare NAME=PATH` scores another tagger's output through the same normalizer and references —
which is the only way the comparison means anything, since running each system through its own
benchmark measures the scorers as much as the models.

`score` reads the common-ml `.jsonl` format, so it scores `model-asr` and `model-multilingual-stt`
identically. A `.txt` reference holding SRT cue timings is parsed as SRT — cue numbers and timecodes
dropped, markup and speaker labels stripped, lyrics kept — and anything else is read as a plain
transcript. Both sides go through whisper's `EnglishTextNormalizer`; without it, WER against
`model-asr` mostly measures the fact that whisper emits punctuation and `model-asr` does not.

That normalizer is also what makes the two systems comparable, since it canonicalises everything they
render differently (`nineteen eighty five` ↔ `1985`, `thirty one` ↔ `31`, `gonna` ↔ `going to`).
Residual bias from the handful of forms it does *not* unify — `ok`/`okay`, `vs`, `&` — measured at
**~0.02pp WER**, because the two systems disagree on only ~35 tokens in 143k words.

`model-asr` and `model-multilingual-stt` run in containers with incompatible dependency stacks; run
them separately and score the output here.
