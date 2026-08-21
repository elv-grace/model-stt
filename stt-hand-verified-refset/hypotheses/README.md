# hypotheses/  — model outputs, one subfolder per model

    hypotheses/<model_name>/<excerpt_id>.txt

e.g. hypotheses/whisper-large-v3/EQ4_homemart_music.txt

Each file is that model's transcription of the SAME excerpt span. The cleanest
way to get span-limited output is to run the model on the CLIPPED audio for each
excerpt (clip with ffmpeg using the verified timecodes). If instead you slice a
full-film transcript by timestamp, make sure the slice covers exactly the
excerpt window. The scorer reads every subfolder here automatically.
