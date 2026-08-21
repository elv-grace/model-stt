# Hand-verified reference set — excerpt plan & rationale

## Why build this at all (the recommendation, endorsed)

A subtitle is a *display artifact*: it compresses, paraphrases, drops fillers, and
cleans up false starts to fit reading speed and screen width. A verbatim reference
is *what was actually said*. Scoring STT against subtitles therefore conflates two
things — the model's real errors and the subtitler's editorial omissions — and you
can't tell how much of your 28 hours of WER is which.

A small, hand-verified verbatim set fixes this in two ways:

1. **Absolute WER you can trust** on the excerpts, uncontaminated by subtitle bias.
2. **A calibration constant** per failure-mode category: `subtitle_WER − true_WER`.
   Once you know "subtitle-based WER overstates clean dialogue by X pp and
   music-under-speech by Y pp," every cheap subtitle-scored number across all 28
   hours becomes interpretable instead of merely comparable.

This is worth doing, and it is the specific thing needed to settle the open
question on The Equalizer (below).

## Sources are AIDS, not the reference

The springfield scripts and subslikescript transcripts you connected are useful —
but only as spelling/lookup aids during a QC pass. Scripts are the *intended*
dialogue (actors improvise, drop, and add lines); fan transcripts carry their own
omission bias. Transcribe from the AUDIO; consult the aids only to fix proper
nouns and to log divergences. Full discipline in `protocol/transcription_protocol.md`.

## The excerpts (≈28 min, inside the 15–30 min target)

Timecodes are cue-anchored from each film's downloaded SRT. **Verify against your
own media before clipping** — rips differ by seconds to minutes.

| # | id | film | category | span | ~dur | why |
|---|---|---|---|---|---|---|
| 1 | GD1_cafe_baseline | Groundhog Day | clean | 00:43:28–00:45:54 | 2:26 | Cleanest dry two-hander; low-noise anchor |
| 2 | AFGM1_jessep_cross | A Few Good Men | clean (fast) | 02:04:56–02:08:00 | 3:04 | Clean audio but rapid + courtroom/military jargon |
| 3 | SN1_bar_breakup | The Social Network | fast_overlap | 00:00:16–00:05:14 | 4:58 | Densest, fastest overlapping Sorkin dialogue |
| 4 | WH1_rushing_dragging | Whiplash | music (aggressive) | 00:26:19–00:29:54 | 3:35 | Tempo corrections over band + profane tirade |
| 5 | BD1_coffee_run_music | Baby Driver | music (dominant) | 00:06:39–00:08:13 | 1:34 | Music-dominant, sparse speech; lyric-vs-speech test |
| 6 | BD2_diner_baby | Baby Driver | music (over dialogue) | 00:17:24–00:19:39 | 2:15 | Dialogue over continuous "B-A-B-Y" |
| 7 | EQ1_diner_clean | The Equalizer | clean | 00:15:45–00:18:24 | 2:39 | **Within-film clean control** for Equalizer |
| 8 | EQ2_mobmeeting_overlap | The Equalizer | fast_overlap (+noise) | 00:42:19–00:44:57 | 2:38 | Mob crosstalk + TV baseball bed |
| 9 | EQ3_store_robbery_noise | The Equalizer | noise | 00:57:11–00:59:45 | 2:34 | Store ambience + shouted stick-up |
| 10 | EQ4_homemart_music | The Equalizer | music (+noise) | 01:42:58–01:45:14 | 2:16 | **Signature dense music-under-speech** finale |

The Equalizer gets ≈10 min across four short excerpts (not one continuous
passage), per the brief, so its estimate averages over several scenes.

## Category coverage

| category | excerpts |
|---|---|
| clean dialogue | GD1, AFGM1, EQ1 |
| fast / overlapping | SN1, EQ2, (AFGM1 secondary) |
| dialogue + music | WH1, BD1, BD2, EQ4 |
| environmental / action noise | EQ3, (EQ2 secondary) |

All four target categories are covered, with the music axis sampled at three
intensities (over-dialogue → dominant → under-action) since that's the axis the
Equalizer question turns on.

## The Equalizer question this set is designed to answer

The earlier analysis found that on The Equalizer the gap between the two models
is *not* purely over-generation: MER halves the gap but doesn't close it, and the
substitution rate is higher for the weaker model — and substitutions (unlike
insertions) don't inflate with hypothesis length. The honest caveat was that at
~24.86% insertions the alignment is heavily perturbed, so S/D/I attribution is
noisy and the read ("dense music-under-speech is a genuine weak spot, not a
scoring artifact") is suggestive rather than airtight. That is exactly the file
where a hand-verified reference is worth the effort.

This plan is built to settle it with a **controlled within-film contrast**:
EQ1 (same film, cast, and mics, but *clean*) vs EQ4 (*dense music-under-speech*),
plus EQ2/EQ3 spanning the other conditions. If the weaker model's true WER and
substitution rate rise sharply from EQ1 → EQ4 while staying close on EQ1, the
weakness is condition-specific (music), not film-specific (mix/accents/vocabulary)
— and it's real, not an artifact of subtitle scoring or perturbed alignment. If
the two models are already far apart on the clean EQ1, the story is different.
Either way you get a defensible answer instead of a suggestive one.

## Caveats

- **Timecodes are approximate** scene boundaries from one particular rip; confirm
  against your media and trim to sentence boundaries.
- **Noise is the thinnest category** (EQ3 primary, EQ2 secondary), by design —
  environmental/action noise mostly lives in The Equalizer here. Add a second
  noise excerpt (e.g. a Jumanji or Woman King action beat) if you want a
  cross-film noise estimate.
- **Small n per category** means per-category CIs are wide; the scorer reports
  bootstrap CIs so you don't over-read a point estimate. The calibration constant
  is most trustworthy where you have the most words (Equalizer).
