import os
import pylangacq
from textgrid import TextGrid, IntervalTier, Interval

def clean_tokens(tokens):
    cleaned = []
    for tok in tokens:
        if tok.word in {"POSTCLITIC", "PRECLITIC"}:
            continue
        cleaned.append(tok.word)
    return " ".join(cleaned).strip()


def process_filereader(corpus, out_dir_textgrid, wav_dir):
    os.makedirs(out_dir_textgrid, exist_ok=True)

    # Get list of .cha files the reader has loaded
    cha_files = list(corpus._file_paths.keys())

    for cha_path in cha_files:
        stem = os.path.splitext(os.path.basename(cha_path))[0]
        wav_path = os.path.join(wav_dir, stem + ".wav")

        if not os.path.exists(wav_path):
            print(f"[WARNING] No matching WAV for {stem}, skipping.")
            continue

        print(f"Processing {cha_path} → {stem}.TextGrid")

        utterances = corpus.utterances(files=[cha_path])

        tg = TextGrid(minTime=0.0, maxTime=None)
        tier = IntervalTier(name="utterances")

        for utt in utterances:
            if utt.time_marks is None or utt.time_marks[0] is None or utt.time_marks[1] is None:
                continue

            start, end = utt.time_marks
            start = float(start) / 1000 if start > 1000 else float(start)
            end   = float(end)   / 1000 if end   > 1000 else float(end)

            text = clean_tokens(utt.tokens)
            if not text:
                continue

            label = f"{utt.participant}: {text}"
            tier.addInterval(Interval(start, end, label))

        if tier.intervals:
            tg.maxTime = max(iv.maxTime for iv in tier.intervals)
        else:
            tg.maxTime = 0.0

        tg.append(tier)

        out_tg = os.path.join(out_dir_textgrid, f"{stem}.TextGrid")
        with open(out_tg, "w", encoding="utf-8") as f:
            tg.write(f)

        print(f"Saved TextGrid: {out_tg}")


if __name__ == "__main__":

    cha_dir = "src/data/train/transcription/cd"
    wav_dir = "src/data/train/Full_wave_enhanced_audio/cd"
    out_dir_textgrid = "src/training/phoneme_posteriorgram/phoneme_targets"

    corpus = pylangacq.read_chat(cha_dir)  # returns a FileReader object for the whole directory
    for subreader in corpus:
        print(type(subreader), subreader._files)
    # process_filereader(corpus, out_dir_textgrid, wav_dir)
