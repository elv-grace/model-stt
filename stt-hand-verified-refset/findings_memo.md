# Hand-verified reference set — findings

*Benchmark: two STT models (`model-asr`, `model-stt`) on 10 excerpts (~28 min) across four failure-mode categories, scored against hand-verified verbatim references and against the downloaded subtitles for the same spans. Calibration constant = subtitle_WER − true_WER (positive ⇒ subtitles overstate true error). CIs are 95% bootstrap over utterances (2000 resamples).*

## Bottom line

Subtitle-based WER is a trustworthy stand-in for true WER **only on clean, well-recorded dialogue**, where it *overstates* true error by a solid ~9–12 pp (CI excludes zero). In every other condition — fast/overlapping, music, noise — the hand-verified data does **not** support a reliable subtitle correction at this sample size: the calibration CIs all span zero, and in dense-music scenes the bias even trends negative (subtitles *understate*). So a single global "subtitles overstate by X" correction applied across the 28 hours would be wrong, and wrong in different directions by condition. Correct the clean stratum; treat the rest as "subtitle WER ≈ true WER, ± a lot" until you have more hand-verified minutes.

On the specific Equalizer question: the hand-verified data **confirms** dense music-under-speech is a real, large, condition-specific weakness (WER roughly triples within the same film, EQ1→EQ4), but **tempers** the earlier subtitle-based read that one model "genuinely misrecognises more" there — on verbatim audio the two models are essentially tied on the music scene and the errors are dominated by *deletions* (dropped words), not substitutions.

## 1. Calibration constants by category

| category | model | true WER | 95% CI | subtitle WER | calibration (pp) | 95% CI (pp) |
|---|---|--:|:--:|--:|--:|:--:|
| clean | model-asr | 9.9% | [7.1, 12.9] | 19.0% | **+9.2** | [+3.7, +14.2] |
| clean | model-stt | 7.2% | [4.8, 10.0] | 18.8% | **+11.6** | [+6.3, +17.0] |
| fast_overlap | model-asr | 11.5% | [9.1, 14.1] | 12.6% | +1.2 | [−2.4, +4.6] |
| fast_overlap | model-stt | 6.8% | [4.9, 8.8] | 8.1% | +1.3 | [−1.6, +4.4] |
| music *(incl. BD1)* | model-asr | 28.1% | [21.4, 36.2] | 25.1% | −2.9 | [−12.8, +5.8] |
| music *(incl. BD1)* | model-stt | 28.1% | [17.9, 43.2] | 21.1% | −7.0 | [−22.3, +5.3] |
| **music *(excl. BD1)*** | model-asr | 21.3% | [16.3, 26.8] | 20.1% | −1.1 | [−8.8, +6.0] |
| **music *(excl. BD1)*** | model-stt | 17.9% | [13.4, 23.0] | 15.3% | −2.6 | [−9.3, +4.1] |
| noise *(EQ3 only)* | model-asr | 16.9% | [10.7, 25.4] | 19.1% | +2.2 | [−11.6, +18.5] |
| noise *(EQ3 only)* | model-stt | 19.0% | [7.4, 38.5] | 23.7% | +4.7 | [−19.5, +28.0] |

Only the **clean** row has a calibration CI clear of zero. That's the one number you can bank: on clean dialogue the subtitle penalises the model for filler and false starts it actually transcribed correctly, inflating WER by ~9–12 pp.

## 2. The calibration constant changes sign by condition

Per-excerpt (Panel A of the figure) the pattern is consistent and interpretable:

- **Clean dialogue overstates** (GD1 +11, AFGM1 +16 pp): the subtitle drops "um"s, false starts, and repeats the model got right, so scoring against it invents errors.
- **Fast/overlap is near-neutral** (SN1 +1, EQ2 +4, WH1 +1 pp): subtitle and verbatim nearly agree on rapid clean speech.
- **Dense/degraded scenes understate** (EQ1 −4, BD2 −3, EQ4 −9 pp; BD1 far off): here the verbatim reference captured overlapped/mumbled words — and correctly *excluded* song lyrics — that the subtitle glossed, so the model's real misses count more against the truth than against the subtitle.

Note EQ1 (tagged `clean`) behaves like a degraded scene (−4 pp), unlike GD1/AFGM1 — the Equalizer diner has accent and room-tone challenges. It's an outlier inside `clean`; the pooled clean constant stays firmly positive because GD1 and AFGM1 carry far more words, but I'd report EQ1 alongside the other Equalizer scenes rather than with the pristine clean anchors.

## 3. The Equalizer verdict (Panel B)

Within the same film, cast, and mics, true WER **roughly triples** from the clean diner (EQ1) to the dense-music finale (EQ4):

| | model-asr | model-stt |
|---|--:|--:|
| EQ1 diner (clean) | 19.9% | 16.4% |
| EQ4 PA-music (dense) | 52.9% | 51.96% |

Two things follow. First, the weakness is **condition-specific, not film-specific** — same recording chain, only the music differs, and WER triples — and it survives verbatim scoring, so it is not an artifact of subtitle bias or perturbed alignment (the original honest caveat). Second, and this **revises the earlier read**: on EQ4 the two models are effectively tied (52.9 vs 52.0), and both fail by *deletion* (model-stt on EQ4: 36 deletions vs 14 substitutions; model-asr similar). Under dense music both models drop speech wholesale rather than one misrecognising more than the other. The earlier subtitle-based inference that one model genuinely misrecognises more on this file doesn't reproduce on the hand-verified audio — the gap is small and the mechanism is omission.

Caveat: EQ4 is ~102 reference words, so treat this as strongly suggestive on now-trustworthy ground, not decisive. A second Equalizer music excerpt would firm it up.

## 4. BD1: both models transcribe song lyrics as speech

BD1 (the "Harlem Shuffle" coffee-run) is music-dominant with sparse dialogue. The verbatim reference correctly marks the song as `[music]` (non-speech); **both models transcribed the sung lyrics as spoken dialogue** ("You move it to the left… Harlem Shuffle…"), producing true WER of 101% (model-asr) and 138% (model-stt) — the errors are almost all insertions. Because the *subtitle* also prints the lyrics on screen, the models' lyric text matches the subtitle, so subtitle WER looks far lower (~53%) — hence the huge negative calibration. This is a real capability gap (no speech/lyric discrimination), but as a single tiny-reference excerpt it destabilises the music aggregate, so it's reported on its own and excluded from the "music (excl. BD1)" row above.

## 5. Model comparison (secondary)

On hand-verified WER, **model-stt is the stronger model** on clean (7.2 vs 9.9%), fast/overlap (6.8 vs 11.5%), and music-excl-BD1 (17.9 vs 21.3%). The two converge exactly where it's hardest: they tie on the EQ4 music scene, and model-asr is marginally better on the single noise excerpt (EQ3, 16.9 vs 19.0%). model-stt also over-inserts more badly on BD1 (138 vs 101% — 64 inserted lyric words).

## 6. Caveats

- **Single-pass transcription.** One transcriber, no second-pass adjudication; careful human transcription still carries ~1–2% WER of its own. For any headline claim (esp. EQ4), a second independent pass on that excerpt is worth it.
- **Small n per category.** clean/fast_overlap are well-powered (>1,200 ref words each); music leans on ~870 words and noise on just 142 (EQ3 alone) — the noise CI is nearly uninformative.
- **Timecodes were from one rip;** confirmed against the media at clip time.
- **The calibration CI is conservative** (independent resampling of the two references), so "CI excludes zero" is a strong signal but "CI spans zero" may partly reflect that conservatism, not only sample size.

## 7. Suggested next steps

1. Add a **second noise excerpt** (Jumanji or Woman King action beat) and a **second Equalizer music excerpt** — the two thinnest, highest-leverage gaps.
2. **Second-pass** the EQ4 reference before publishing the Equalizer conclusion.
3. Apply the **clean-stratum correction only** to the 28-hour subtitle numbers; annotate the other strata as uncorrected with the CIs above.
4. Report BD1 as a standalone "lyrics-as-speech" finding, not inside the music average.
