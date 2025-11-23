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

if __name__ == "__main__":
    
    # Extracts utterances
    # corpus = pylangacq.read_chat("src\\data\\train\\transcription\\cd\\S079.cha")    #single file for testing
    corpus = pylangacq.read_chat("src\\data\\train\\transcription\\cd")    # whole directory
    
    ## NEED REST OF CODE TO WORK WITH A pylangacq filereader object
    wav_path = "src\\data\\train\\Full_wave_enhanced_audio\\cd\\S079.wav"

    utterances = corpus.utterances()  # yields dicts with speaker, tier, timestamp, etc.
    
    
    out_textgrid = "src\\training\\phoneme_posteriorgram\\phoneme_targets\\S079.TextGrid"
    # Create TextGrid
    tg = TextGrid(minTime=0.0, maxTime=None)

    # One tier for the whole conversation (or split by speaker if you prefer)
    tier = IntervalTier(name="utterances")

    for i, utt in enumerate(utterances):
        # Utterance object fields:
        #   utt.participant -> speaker code (e.g., "INV", "PAR")
        #   utt.tokens -> list of Token objects
        #   utt.time_marks -> (start, end)
        
        # Skip utterances with no timestamps
        if utt.time_marks is None or utt.time_marks[0] is None or utt.time_marks[1] is None:
            continue

        start, end = utt.time_marks
        start = float(start) / 1000 if start > 1000 else float(start)   # ADReSS timestamps sometimes in ms
        end   = float(end)   / 1000 if end   > 1000 else float(end)

        # Build text from tokens
        # text = " ".join(tok.word for tok in utt.tokens).strip()
        text = clean_tokens(utt.tokens)

        # Empty text? Skip
        if not text:
            continue

        # Label includes speaker + text
        label = f"{utt.participant}: {text}"

        tier.addInterval(Interval(start, end, label))

    # Finalize TextGrid
    # Infer maxTime from last interval
    if len(tier.intervals) > 0:
        tg.maxTime = max(iv.maxTime for iv in tier.intervals)
    else:
        tg.maxTime = 0.0

    tg.append(tier)

    # Save TextGrid
    with open(out_textgrid, "w", encoding="utf-8") as f:
        tg.write(f)

    print(f"Saved TextGrid to {out_textgrid}")