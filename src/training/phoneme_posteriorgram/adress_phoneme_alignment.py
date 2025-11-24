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

        # --- Create empty TextGrid ---
        tg = TextGrid(minTime=0.0, maxTime=None)

        # --- Prepare dynamic tiers for each speaker ---
        tiers = {}   # {participant_name: IntervalTier}

        for utt in utterances:

            # Skip utterances without timestamps
            if not utt.time_marks or utt.time_marks[0] is None or utt.time_marks[1] is None:
                continue

            start, end = utt.time_marks

            # Convert ms → seconds if needed
            start = float(start) 
            end   = float(end) 

            if end <= start:
                end = start + 0.001  # prevent zero-length intervals

            # Clean transcript tokens
            text = clean_tokens(utt.tokens)
            if not text:
                continue

            speaker = utt.participant  # "INV", "PAR", etc.

            # Create tier if this speaker has not appeared yet
            if speaker not in tiers:
                tiers[speaker] = IntervalTier(name=speaker)

            tier = tiers[speaker]

            # Avoid overlap in a speaker tier (MFA requires no overlaps)
            if tier.intervals:
                prev_end = tier.intervals[-1].maxTime
                if start < prev_end:
                    start = prev_end + 0.001

            # Add interval with ONLY the actual text
            tier.addInterval(Interval(start, end, text))

        # Finish building the TextGrid
        all_intervals = []
        for t in tiers.values():
            tg.append(t)
            all_intervals.extend(t.intervals)

        tg.maxTime = max((iv.maxTime for iv in all_intervals), default=0.0)

        # Save result
        out_tg = os.path.join(out_dir_textgrid, f"{stem}.TextGrid")
        with open(out_tg, "w", encoding="utf-8") as f:
            tg.write(f)

        print(f"Saved TextGrid: {out_tg}")


if __name__ == "__main__":
    
    #
    #   Run this once in order to preprocess your transcripts for 
    #   forced alignment input
    #

    cha_dir = "src/data/train/transcription/cd"
    wav_dir = "src/data/train/Full_wave_enhanced_audio/cd"
    out_dir_textgrid = "src/data/train/Full_wave_enhanced_audio/cd"

    corpus = pylangacq.read_chat(cha_dir)  # returns a FileReader object for the whole directory
    
    process_filereader(corpus, out_dir_textgrid, wav_dir)
