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


def process_filereader(corpus, out_dir_textgrid, removeInvestigator=True):
    os.makedirs(out_dir_textgrid, exist_ok=True)
    
    
    for subreader in corpus:
        cha_path = subreader.file_paths()[0]
        stem = os.path.splitext(os.path.basename(cha_path))[0]
        speaker_id = stem[2:4]

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
            start = float(start) /1000
            end   = float(end) /1000

            if end <= start:
                end = start + 0.001  # prevent zero-length intervals

            # Clean transcript tokens
            text = clean_tokens(utt.tokens)
            if not text:
                continue
            
            speaker = utt.participant  # "INV", "PAR", etc.
            
            if removeInvestigator and speaker=='INV': continue

            
            speaker += str(speaker_id)
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
        out_tg = os.path.join(out_dir_textgrid, f"{stem}_patient.TextGrid")
        with open(out_tg, "w", encoding="utf-8") as f:
            tg.write(f)

        print(f"Saved TextGrid: {out_tg}")


if __name__ == "__main__":
    
    #
    #   Run this once in order to preprocess your transcripts for 
    #   forced alignment input
    #

    cha_dir = "src/data/train/transcription/cd"
    cha_dircc = "src/data/train/transcription/cc"
    
    out_dir_textgrid = "src/data/train/patient_audio_diarized/cd"
    out_dir_textgridcc = "src/data/train/patient_audio_diarized/cc"

    corpus = pylangacq.read_chat(cha_dir)  # returns a FileReader object for the whole directory
    corpuscc = pylangacq.read_chat(cha_dircc)  # returns a FileReader object for the whole directory
    
    process_filereader(corpus, out_dir_textgrid)
    process_filereader(corpuscc, out_dir_textgridcc)
