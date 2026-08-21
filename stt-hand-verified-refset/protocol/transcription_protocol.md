# Transcription protocol — hand-verified STT reference set

This is the rulebook for producing the verbatim references. Follow it exactly, or
the reference and the scorer will disagree and the calibration constant will be
noise. It should take one careful pass plus one QC pass per excerpt (budget
roughly 6–10× the audio duration for a first-time transcriber).

## The one rule everything else serves

**Transcribe what was *said*, from the audio — not what was *shown*, from the
subtitle or the script.** A subtitle is a display artifact: it compresses,
paraphrases, drops filler, and cleans up false starts to fit reading speed and
screen space. A script is the *intended* dialogue: actors improvise, flub, drop
and add lines. If you edit either one into your reference, you inherit its
omissions and your "true" WER just re-measures the subtitler's or screenwriter's
choices. The whole point of this set is to escape that.

## Blind-first workflow (this is what prevents anchoring)

Do each excerpt in two passes:

**Pass 1 — blind.** Keep the subtitle and the script CLOSED. Play the audio in
short loops (3–8 seconds) and type exactly what you hear into the worksheet's
"what I heard" area. Include everything: false starts, repeated words, fillers,
grammatical mistakes, stammers.

**Pass 2 — QC with aids open.** Now open the springfield script and the
subslikescript transcript for that title (they're in your two connected folders)
and the subtitle. Use them ONLY to:
- fix spelling of proper nouns, place names, and jargon (see each excerpt's watchlist);
- catch words you genuinely couldn't make out on the first pass — then go back to
  the audio and confirm by ear before writing them;
- log places where the audio clearly differs from the script/subtitle in the
  worksheet's divergence table (these divergences are the evidence that the set
  is doing its job).

If an aid says one thing and your ears say another, **your ears win.** Never paste
a line from an aid because it "sounds about right."

## Verbatim conventions

These map one-to-one onto `scoring/normalize.py`. The scorer lowercases, removes
punctuation, splits hyphens, and (by default) KEEPS fillers, so what matters for
your score is the *words and their order*, not your capitalization or commas.

- **Fillers: keep them.** Write `um`, `uh`, `er`, `hmm`, `mm-hmm`, `huh`, `oh`,
  `ah`. They are real tokens and are scored by default. (The scorer can also be
  run with `--drop-fillers` to see WER both ways — that difference is itself a
  useful number.)
- **False starts and repeats: keep them verbatim.** "I— I don't—  I don't know"
  is written `I I don't I don't know`. Use a double hyphen `--` in the human-
  readable worksheet to show the cut-off if you like; the scorer treats hyphens
  as spaces, so `don't--` becomes `don't`.
- **Contractions: write what's said.** `don't`, `gonna`, `wanna`, `gotta`,
  `'em`, `dunno` if that's what you hear. Do NOT expand `gonna`→`going to`.
- **Numbers: write as WORDS, the way they're spoken.** `$9,800` said aloud as
  "ninety-eight hundred" → write `ninety-eight hundred`; said as "nine thousand
  eight hundred" → write that. This removes digit-vs-word ambiguity; the scorer
  spells out digits only on the model's side to match you.
- **Spelled-out letters:** "P.A. system" spoken "pee-ay system" → write `P A
  system` (letters as separate tokens). "B-A-B-Y" sung/spelled → `B A B Y`.
- **Non-speech: mark it, don't transcribe it.** Use bracket tags on their own or
  inline: `[music]`, `[laughter]`, `[gunshot]`, `[phone ringing]`,
  `[applause]`, `[overlapping]`. The scorer STRIPS anything in `[...]`, `(...)`,
  or `{...}` before scoring, so tags never count as words — they're there for
  human auditing of what the non-speech context was.
- **Unintelligible audio:** if after repeated listens you truly cannot tell,
  write `(unintelligible)`. It is stripped by the scorer (counts as neither a hit
  nor an error), so keep it rare — every `(unintelligible)` is a word you're
  removing from the measurement. If you can make a confident guess, write the
  guess instead.
- **Song lyrics that are part of the soundtrack (not a character singing to
  camera) are NOT speech.** Mark the stretch `[music: lyrics]` and don't
  transcribe the lyrics. (This matters most for Baby Driver, whose SRT prints
  song lyrics as cues.) If a character is *singing dialogue* meaningfully (e.g.
  Baby spelling "B-A-B-Y" to Debora), transcribe that.
- **Overlapping speech:** transcribe both speakers in the order their words
  land; if truly simultaneous and one is unrecoverable, transcribe the dominant
  speaker and tag `[overlapping]`.

## Segmentation (one utterance per line)

Put **one utterance per line** in the reference file — roughly one sentence or one
speaker-turn. The scorer aligns the whole excerpt as a stream (so line breaks
don't change your WER), but it uses the line boundaries as the resampling unit for
bootstrap confidence intervals. More lines = tighter CIs, so don't dump the whole
excerpt on one line. Don't put speaker names in the reference file; note them in
the worksheet instead.

## What "done" looks like for one excerpt

1. `references/<excerpt_id>.txt` — your verbatim transcript, one utterance per
   line, fillers and false starts intact, non-speech in brackets, numbers as
   words. This is the only file the scorer reads as truth.
2. The worksheet saved with its QC checklist ticked and any divergences logged.
3. `subtitle_refs/<excerpt_id>.txt` — produced mechanically with
   `scoring/extract_subtitle_ref.py` from your downloaded SRT. **Never hand-edit
   this**; it must stay a faithful copy of the subtitle for the same span.

Then run `python scoring/score.py` and read the calibration column.
