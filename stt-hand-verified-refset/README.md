# Hand-verified STT reference set — build kit

A complete scaffold for producing a small, trustworthy verbatim reference set and
using it to calibrate your 28 hours of subtitle-based WER. Everything here is
built; the one thing only you can do is the listening-and-typing (transcription
must come from the audio, by ear).

## What's in here

```
excerpts.json                 single source of truth: the 10 excerpts, spans, categories
excerpt_plan.md               the plan, rationale, category coverage, the Equalizer test
protocol/
  transcription_protocol.md   the rulebook (blind-first workflow, verbatim conventions)
worksheets/                   one fill-in worksheet per excerpt + INDEX.md (generated)
references/                   <- YOU fill: hand-verified verbatim transcripts (the truth)
subtitle_refs/                <- generated from your SRTs by extract_subtitle_ref.py
hypotheses/<model>/           <- YOU fill: each model's transcription per excerpt
scoring/
  normalize.py                shared normalization (protocol and scorer agree here)
  score.py                    WER(S/D/I)/MER/CER, subtitle-WER, calibration + bootstrap CIs
  extract_subtitle_ref.py     pull subtitle text for an excerpt span from an SRT
  make_worksheets.py          regenerate worksheets from excerpts.json
  test_scorer.py              unit tests (10, all passing)
```

## The workflow, end to end

1. **Clip the audio.** For each excerpt in `excerpts.json`, verify the timecodes
   against your media and clip that span (e.g. with ffmpeg). Trim to sentence
   boundaries.
2. **Transcribe by ear** into the worksheet (`worksheets/<id>.md`), blind pass
   first with the subtitle and script CLOSED, then a QC pass with them open for
   spelling only. Move the finished verbatim text to `references/<id>.txt`, one
   utterance per line. Rules: `protocol/transcription_protocol.md`.
3. **Make the subtitle references** for the same spans:
   ```
   python scoring/extract_subtitle_ref.py --srt /path/Equalizer.srt \
       --start 01:42:58 --end 01:45:14 --out subtitle_refs/EQ4_homemart_music.txt
   ```
4. **Run each STT model** on the clipped audio and save outputs to
   `hypotheses/<model>/<id>.txt`.
5. **Score:**
   ```
   python scoring/score.py --bootstrap 2000
   ```
   Reads `results/per_excerpt.csv`, `results/per_category.csv`, `results/results.json`
   and prints a summary. Add `--drop-fillers` to see WER with fillers removed too.

You can run step 5 at any point — missing files are skipped, so scores appear as
you complete excerpts.

## Reading the output

- **true_wer / MER / CER** — against your verbatim reference. This is the number
  you can trust absolutely.
- **sub_wer** — the same model scored against the subtitle for the same span.
- **calibration = sub_wer − true_wer** (in pp). Positive ⇒ subtitle-based WER
  *overstates* error (the usual case: subtitles drop fillers/false starts the
  model correctly transcribes). This is your correction factor: apply the
  per-category constant to your 28 hours of subtitle-scored WER to recover an
  estimate of true WER.
- **95% CIs** — bootstrap over utterances (for WER) and a conservative
  independent-resampling CI for the calibration constant. Wide CIs mean "don't
  over-read the point estimate," especially in thin categories.

## The Equalizer test (why EQ1 and EQ4 are both here)

EQ1 (clean) and EQ4 (dense music-under-speech) are the *same film, cast, and
mics*. Comparing a model's true WER and substitution rate across EQ1 → EQ4
isolates whether its Equalizer weakness is condition-specific (music) or
film-specific (mix/accents/vocab), and whether it survives verbatim scoring at
all. See `excerpt_plan.md` for the full argument.

## Verified

`python scoring/test_scorer.py` → 10/10 passing (aligner S/D/I, WER can exceed 1
while MER is capped, CER, normalizer keeps fillers / strips annotations / spells
hypothesis-side digits / splits hyphens, end-to-end score_pair). The driver,
bootstrap, and subtitle extractor were smoke-tested on synthetic data.
