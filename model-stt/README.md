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
| Backend | faster-whisper | Byte-identical text to openai-whisper on 116 clean FLEURS utterances, ~2.3x faster, no torch (~5.5 GB of wheels out of the image), and it is the only one that *has* a VAD to offer — kept as the `clean-audio` profile, though it is [off by default](#recall-and-hallucination-what-vad-costs). |
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
WEIGHTS=turbo-ct2 ./build.sh     # the shipped image (~1.6 GB weights, no torch)
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
| `default` | **off** | off | Everything by default. Decode all the audio and let the decoder's own guards filter. |
| `clean-audio` | on, `threshold 0.25` / `speech_pad 2000ms` | on | Content with *distinguishable silence* — scripted film, studio recordings — where suppressing stock-phrase artifacts is worth more than recall. |
| `reproducible` | inherits | inherits | Byte-identical output across runs. Sets `deterministic_fallback` only, so compose it with either profile above. See [the reproducible profile](#the-reproducible-profile). |

The two fields in each profile are a pair, not independent knobs. With VAD off the decoder sees noise
it used to be shielded from, and a *conditioned* decode locks onto a phrase and repeats it; turning
conditioning off is what prevents that. Do not flip one without the other — there is a unit test
guarding the coupling.

VAD-off is the default because VAD fails *silently*: audio it discards never reaches the decoder, so
over-filtering leaves no low-confidence segment and no repetition signal behind, only missing time.
See [recall and hallucination](#recall-and-hallucination-what-vad-costs).

## Testing

```bash
make pytest                # 87 unit tests, no weights or GPU needed
pytest -m weights tests/   # end-to-end against real weights
make test                  # container end-to-end (buildscripts/testers/test-model.sh)
```

`make test` is the standard tagger-model container test from `buildscripts`. It transcribes every
file **directly in** `test-files/` (subdirectories are excluded by its `-maxdepth 1`) and does
`rm -rf test-output/` first.

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

**Measured with VAD off on both**, which is what the shipped configuration now is again. The
equivalence does not survive noisy audio.

An earlier version of this section reported a single run over "600s of basketball crowd noise"
(openai 24 segments / 19 repeated, CT2 unfiltered 38 / 31, CT2 with VAD 12 / 0) and concluded that
faster-whisper hallucinates more unfiltered. **Both the fixture and the method were wrong**, so the
rows below replace it:

- The fixture is not crowd noise. `NBAallstar 1200-1800s` is ~340s of All-Star player introductions —
  transcribed accurately, down to the spellings of Antetokounmpo and Gilgeous-Alexander — followed by
  ~260s of genuine non-speech.
- Single runs cannot support the comparison. Output is not reproducible under the shipped temperature
  ladder (see [reproducibility](#output-is-not-reproducible)), and repeated-segment counts vary by
  2–3x run to run.

Re-measured on that slice, 3 reps under the shipped ladder, reported as min–max:

| | segments | words | repeated | segments in the *speech* half |
|---|---|---|---|---|
| openai-whisper (no VAD exists) | 39–52 | 316–431 | 11–24 | 25–28 |
| faster-whisper, vad off | 37–55 | 326–354 | 9–31 | 24–28 |
| faster-whisper, vad on `0.5/400` | **0** | 0 | 0 | **0** |
| faster-whisper, vad on `0.25/2000` | 1 | 2 | 0 | 1 |

**The two backends are indistinguishable** — every range overlaps. Neither "CT2 hallucinates more"
nor "less" survives replication; both were single-run noise.

**What does survive is the VAD row.** At Silero's own defaults VAD emitted *zero segments* for 600
seconds containing 340s of clean announcer speech, and one segment at the tuned setting. The old
`vad on → 0 repeated` looked like clean suppression; it is total suppression. That is the finding
that moved VAD out of the default (see
[recall and hallucination](#recall-and-hallucination-what-vad-costs)).

The VAD rows are reproducible; the two unfiltered rows are not, hence the ranges. Pinning
`temperature=[0.0]` makes all four reproducible but measures a different system — repetition on
`vad off` jumps to 107, because the temperature fallback *is* the repetition guard.

The backend choice therefore no longer rests on VAD at all. It rests on speed, on the absence of a
torch dependency, and on VAD remaining *available* for the `clean-audio` profile. openai-whisper has
no VAD to turn on — it detects silence from the decoder's own `no_speech_prob`, exactly the signal
that fails when the model is confident it heard speech in music or crowd noise. whisper's own source
concedes the gap (`timing.py`: *"a better segmentation algorithm based on VAD should be able to
replace this"*). With VAD on, CT2 scores en 4.21% / fr 6.82% WER against openai's 4.33% / 6.46%.

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

Measured on the `test-files/` fixtures on one L40S. The current set is 8 files; `NBAsummerleague33min.mp4`
has no audio stream and is skipped. The full set runs in **10.0 minutes**, 2556 MiB peak, producing
101,588 tags into `test-output/out.jsonl`.

The central finding is that **VAD-strictness and hallucination are not one axis you can tune.**
Speech-over-loud-crowd (wants permissive VAD) and crowd-without-speech (wants aggressive VAD) are the
same thing to Silero, and in `NBAallstar.mp4` they sit *minutes apart in the same file*. So no global
threshold solves both:

| | `NBAallstar` 50s–1884s | `spiderverse` four dropped stretches |
|---|---|---|
| VAD `thr 0.5 / pad 400` | 30.5 min dropped | 0 of 4 recovered |
| VAD `thr 0.25 / pad 2000` | **30.5 min still dropped** | 4 of 4 recovered |
| VAD off | recovered | recovered |

That NBAallstar stretch is 30 minutes of arena PA announcements, not silence — decoded with VAD off
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

**Per-file calibration is the point, and it does real work.** The learned phrases are not a fixed
set — they differ by file, including phrases that look far too common to touch. On `equalizer.mp4`
the guard learned `'i dont know'` and `'im sorry'` and dropped 17 captions of 1439 (1.2%);
re-decoding every one of them with fresh context, **14 of 17 returned completely different text**
(`"I don't know."` → `"I'm going to go ahead and cook it."`), and 2 of the remaining 3 re-decoded into
an `"I'm sorry."` ×6 loop rather than clean speech. Meanwhile `spiderman-across-the-spiderverse`
*kept* all 14 of its `'i dont know'` captions, because there they sit in conversation and fail the
median-isolation test. A global list would have had to choose one behaviour for both files.

Disable with `isolated_artifact_gap: 0`. Note what it does **not** do: it catches one specific
failure — the recurring stock phrase over non-speech — and cannot catch a confident one-off
invention such as a wrong proper noun. Nothing in this pipeline can; see below.

#### Three things that do not work, with the evidence

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

#### Straddled segments

faster-whisper decodes the VAD-*concatenated* audio and maps timestamps back afterwards, resolving
each word to a speech chunk **by its midpoint** (`restore_speech_timestamps`). A word whose alignment
lands a frame either side of a chunk boundary resolves to the wrong chunk — and since a segment's
span is taken from its first and last word, one misplaced word stretches the whole segment across the
excision. `"Good luck, guys."` was emitted spanning **126.71s → 333.92s**; the phrase is really at
333.06–333.90, and the audio at 126.7s is `"What the fuck? Who did that?"`.

Worse, `to_sentences` then splits on the 207s gap, so the caption is not merely mistimed — it is torn
into `"Good"` and `"luck, guys."`.

`_repair_straddles` splits a segment's words into runs separated by more than `straddle_gap_max`,
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

#### Output is not reproducible

Same config, same process, four consecutive runs of the same file under the shipped default:

```
run0: words=806   run1: words=814   run2: words=806   run3: words=811     (all 4 texts differ)
```

The cause is **whisper's temperature fallback, and only that**. A decode that trips
`compression_ratio_threshold` or `logprob_threshold` is retried at temperature > 0, which *samples*
rather than decoding greedily, and that sampling is unseeded. Isolated by varying one thing at a time:

| | 3 runs |
|---|---|
| full temperature ladder, `cond=False` (shipped) | **varies** — 182 / 185 / 193 segments |
| `temperature=[0.0]`, `cond=False` | **bit-identical** |
| `temperature=[0.0]`, `cond=True` | **bit-identical** |

Two things this rules out, both of which earlier versions of this file asserted:
`condition_on_previous_text` is **not** the cause (it is off by default and the output still varies;
turning it off removes the propagation path, not the sampling), and float16 GPU reduction order is
**not** involved (it would still vary at temperature 0, and it does not).
`ctranslate2.set_random_seed()` does not help, because faster-whisper's sampling path does not draw
from the seeded CT2 RNG.

**There is a noise floor of roughly 1–2% on word counts**, and much larger on small counts — repeated
segments over a 600s slice varied 2–3x. Small run-to-run deltas mean nothing. FLEURS numbers *are*
stable, because clean short speech never trips the fallback.

`temperature=[0.0]` buys bit-reproducibility but is not a usable answer: the fallback is a
load-bearing repetition guard, and disabling it took repeated segments on one 600s slice from ~9–31
up to **107**.

#### The `reproducible` profile

Seeding is not an option — `ctranslate2.set_random_seed()` has no effect on the sampling (verified
with forced sampling, and with the seed set before model construction), and `Whisper.generate()` in
CT2 4.8.1 has no per-call seed. Neither do the deterministic knobs substitute for the retry:
`no_repeat_ngram_size` had *literally zero* effect, because it suppresses repeats within one 30s
generation while the loop repeats across windows.

What works is keeping the reject-and-retry loop and making the *retries* deterministic. Sampling is
only how faster-whisper makes a retry come out differently; escalating beam size, patience,
`repetition_penalty` and `no_repeat_ngram_size` does the same job without randomness.
`_DeterministicFallback` in [src/backends.py](src/backends.py) proxies the CTranslate2 model and
swaps each sampled rung for a deterministic one.

```bash
--params '{"profile": "reproducible"}'
--params '{"profile": "clean-audio", "deterministic_fallback": true}'   # composable
```

Over the whole fixture set it is **not a quality tradeoff**:

| | words | artifact-shaped captions | run-ons | wall | reproducible |
|---|---|---|---|---|---|
| `default` (sampled retries) | 90,254 | 57 | 19 | 601.8s | no |
| `reproducible` (deterministic retries) | 90,509 | 57 | 17 | **529.6s** | **yes** |

It is also *faster*, which is not what beam-search retries suggest: beam search escapes a failing
decode in fewer rungs than sampling does, so there are fewer retries overall.

Two honest caveats. That table is **one** draw of a nondeterministic system against a fixed one, and
`default` varies ~2x in artifact count run to run — the real claim is that `reproducible` lands inside
`default`'s own observed range on every metric, per file as well as in total. And on a deliberately
fallback-heavy 600s slice the two do separate: 238 words / 48 repeated against 332–453 / 6–20. Even
there the *speech* half is untouched (28 segments either way) and the lost words are all hallucinated
ones, but the failure mode changes shape — deterministic retries emit more short duplicate fragments
where sampled ones emit fewer, longer, more varied inventions. The duplicates are the more tractable
kind, since the [isolated-artifact guard](#the-isolated-artifact-guard) catches recurring short
phrases and cannot catch varied long ones.

It is off by default because the fixture set is not broad enough to justify flipping a decoder
default on. Turn it on when reproducibility has value on its own — benchmarking, diffing two runs,
cache keys, QA sign-off.

The proxy reaches into faster-whisper's call into CTranslate2, so it is coupled to that call's
keyword arguments and should be re-checked on upgrade. If the names change it stops intervening
rather than misbehaving, since it only fires when a `sampling_temperature` keyword is present.

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
python -m bench.run_bench --files test-files/bench-files/*.m4a --systems turbo-openai turbo-ct2
python -m bench.score --hyp bench-output/turbo-ct2.jsonl --ref refs/ --task asr
```

`run_bench` reports real-time factor with model load timed separately, plus per-process GPU memory
(via `nvidia-smi`, since `torch.cuda.max_memory_allocated` sees nothing for the CT2 backend). Need to uncomment paths B and C first to replicate experiments (for translation task (not in scope of tagger container)).

`score` reads the common-ml `.jsonl` message format, so it scores `model-asr` and
`model-multilingual-stt` output identically. Both sides are normalized with whisper's own
`EnglishTextNormalizer` — without it, WER against `model-asr` mostly measures the fact that whisper
emits punctuation and `model-asr` does not.

`model-asr` and `model-multilingual-stt` run in containers with incompatible dependency stacks and are
not driven from `run_bench`; run them separately and score their output with `bench/score.py`.
