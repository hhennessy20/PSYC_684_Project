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


    for subreader in corpus:
        
        cha_path = subreader.file_paths()[0]
        stem = os.path.splitext(os.path.basename(cha_path))[0]

        print(f"Processing {cha_path} → {stem}.TextGrid")

        utterances = subreader.utterances()

        tg = TextGrid(minTime=0.0, maxTime=None)
        tier = IntervalTier(name="utterances")

        for utt in utterances:
            if utt.time_marks is None or utt.time_marks[0] is None or utt.time_marks[1] is None:
                continue

            start, end = utt.time_marks
            start = float(start) / 1000 if start > 1000 else float(start)
            end   = float(end)   / 1000 if end   > 1000 else float(end)
            
            if tier.intervals:
                prev_end = tier.intervals[-1].maxTime
                if start < prev_end:
                    start = prev_end + 0.001  # avoid overlap by nudging forward

            if end <= start:
                end = start + 0.001

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
    
    process_filereader(corpus, out_dir_textgrid, wav_dir)
