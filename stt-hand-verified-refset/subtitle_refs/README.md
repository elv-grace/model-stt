# subtitle_refs/  — subtitle text for each excerpt span (the "cheap" reference)

One file per excerpt, named `<excerpt_id>.txt`. Generate MECHANICALLY from the
SRT you downloaded, for the SAME span as the verbatim reference:

    python ../scoring/extract_subtitle_ref.py --srt /path/to/Film.srt \
        --start HH:MM:SS --end HH:MM:SS --out EQ4_homemart_music.txt

Never hand-edit these. They must stay a faithful copy of the subtitle so the
calibration constant (subtitle_WER - true_WER) measures the subtitle's bias, not
yours.
