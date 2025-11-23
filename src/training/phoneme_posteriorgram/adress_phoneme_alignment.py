import pylangacq

if __name__ == "__main__":
    corpus = pylangacq.read_chat("src\data\train\transcription\cd\S079.cha")

    utterances = corpus.utterances()  # yields dicts with speaker, tier, timestamp, etc.

    print(utterances)