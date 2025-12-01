from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm
import pylangacq
import time
from src import config


@dataclass
class ChaSegment:
    speaker: str        # e.g. "PAR", "INV"
    start_sec: float
    end_sec: float
    text: str


def clean_tokens(tokens) -> str:
    """
    Turn pylangacq token objects into a clean text string.
    Skip special tokens like POSTCLITIC/PRECLITIC.
    """
    words = []
    for tok in tokens:
        w = getattr(tok, "word", None)
        if not w:
            continue
        if w in {"POSTCLITIC", "PRECLITIC"}:
            continue
        words.append(w)
    return " ".join(words).strip()


def segments_from_subreader(subreader) -> Tuple[str, List[ChaSegment]]:
    """
    Given a pylangacq FileReader for a single .cha file,
    return (file_id, list_of_segments).

    file_id is the stem of the .cha file (e.g., "S001").
    """
    cha_paths = subreader.file_paths()
    if not cha_paths:
        raise ValueError("subreader has no file_paths()")

    cha_path = Path(cha_paths[0])
    file_id = cha_path.stem

    utterances = subreader.utterances()
    segments: List[ChaSegment] = []

    for utt in utterances:
        # time_marks is typically (start, end) or (None, None)
        tm = getattr(utt, "time_marks", None)
        if not tm or tm[0] is None or tm[1] is None:
            continue

        start, end = tm

        # Convert to float seconds, with ms → sec safety
        start = float(start)
        end = float(end)

        # If the times are ridiculously large, assume they are in milliseconds
        if end > 1000.0:  # > 1000 seconds (~16 minutes) is unlikely
            start /= 1000.0
            end /= 1000.0

        if end <= start:
            end = start + 0.001  # avoid zero/negative duration

        text = clean_tokens(getattr(utt, "tokens", []))
        if not text:
            continue

        speaker = getattr(utt, "participant", "").strip()  # "PAR", "INV", etc.
        if not speaker:
            continue

        segments.append(
            ChaSegment(
                speaker=speaker,
                start_sec=start,
                end_sec=end,
                text=text,
            )
        )

    # Sort segments by start time just in case
    segments.sort(key=lambda s: s.start_sec)
    return file_id, segments


def load_all_segments_from_transcripts() -> Dict[str, List[ChaSegment]]:
    """
    Read all .cha transcripts from ADReSS cc and cd transcription folders,
    and build a dict: {file_id: [ChaSegment, ...]}.
    """
    all_segments: Dict[str, List[ChaSegment]] = {}

    for trans_dir in [config.ADRESS_TRANSCRIPT_CC_DIR, config.ADRESS_TRANSCRIPT_CD_DIR]:
        if not trans_dir.exists():
            print(f"[DIAR] Transcript dir does not exist: {trans_dir}")
            continue

        print(f"[DIAR] Reading transcripts from: {trans_dir}")
        corpus = pylangacq.read_chat(str(trans_dir))
        time.sleep(1)
        # corpus is iterable: each subreader is a per-file reader
        for subreader in corpus:
            file_id, segs = segments_from_subreader(subreader)
            if not segs:
                print(f"[DIAR] No valid segments in transcript for {file_id}, skipping.")
                continue

            if file_id in all_segments:
                # Merge if duplicate ID appears (shouldn't happen in ADReSS)
                all_segments[file_id].extend(segs)
                all_segments[file_id].sort(key=lambda s: s.start_sec)
            else:
                all_segments[file_id] = segs

    print(f"[DIAR] Loaded segments for {len(all_segments)} files from transcripts.")
    #print(all_segments[file_id][:5])  # Print first 5 segments of last file_id processed
    return all_segments


def write_rttm(file_id: str, segments: List[ChaSegment], out_path: Path) -> None:
    """
    Write an RTTM file for all segments (PAR, INV, etc.)
    in NIST-style SPEAKER lines.

    Format:
      SPEAKER <file_id> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            dur = seg.end_sec - seg.start_sec
            if dur <= 0:
                continue
            f.write(
                f"SPEAKER {file_id} 1 {seg.start_sec:.3f} {dur:.3f} <NA> <NA> {seg.speaker} <NA> <NA>\n"
            )
            #print(f"SPEAKER {file_id} 1 {seg.start_sec:.3f} {dur:.3f} <NA> <NA> {seg.speaker} <NA> <NA>\n")
    print(f"[DIAR] Wrote RTTM: {out_path}")


def extract_patient_audio(
    audio_path: Path,
    segments: List[ChaSegment],
    out_path: Path,
    gap_sec: float = 0.1,
) -> None:
    """
    Concatenate all PAR segments from `audio_path` into a single
    patient-only wav file at `out_path`.

    - Keeps original sample rate
    - Converts to mono if needed
    - Inserts small silence gaps between segments
    """
    if not audio_path.exists():
        print(f"[DIAR] Audio file not found: {audio_path}")
        return

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # to mono

    if sr != config.TARGET_SR:
        print(
            f"[DIAR] WARNING: {audio_path.name} sample rate {sr} != TARGET_SR {config.TARGET_SR}"
        )

    par_segments = [seg for seg in segments if seg.speaker == "PAR"]
    if not par_segments:
        print(f"[DIAR] No PAR segments for {audio_path.name}, skipping patient audio.")
        return

    par_segments = sorted(par_segments, key=lambda s: s.start_sec)

    chunks = []
    gap = np.zeros(int(gap_sec * sr), dtype=audio.dtype)

    for seg in par_segments:
        start_sample = int(seg.start_sec * sr)
        end_sample = int(seg.end_sec * sr)
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(0, min(end_sample, len(audio)))
        if end_sample <= start_sample:
            continue

        segment_audio = audio[start_sample:end_sample]
        if chunks:
            chunks.append(gap)
        chunks.append(segment_audio)

    if not chunks:
        print(f"[DIAR] No valid audio samples from PAR segments in {audio_path.name}.")
        return

    concatenated = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), concatenated, sr)
    print(f"[DIAR] Wrote patient audio: {out_path}")


def find_audio_for_id(file_id: str) -> Path:
    """
    Find the audio file in TRAIN_AUDIO_DIR matching the transcript ID.
    We look for file_id.wav or file_id.WAV.
    """
    cand1 = config.TRAIN_AUDIO_DIR / f"{file_id}.wav"
    cand2 = config.TRAIN_AUDIO_DIR / f"{file_id}.WAV"
    print(f"[DIAR] Looking for audio for ID={file_id} at {cand1} or {cand2}")
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2

    # If not found, return a non-existing path but log later
    return cand1


def main():
    # Ensure output dirs exist
    config.PATIENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load all segments from cc + cd transcripts
    segments_index = load_all_segments_from_transcripts()
    if not segments_index:
        print("[DIAR] No segments loaded from transcripts. Check paths in config.py.")
        return

    # 2) For each file_id, generate RTTM + patient-only wav
    file_ids = sorted(segments_index.keys())
    print(f"[DIAR] Generating RTTM and patient audio for {len(file_ids)} files.")

    for file_id in tqdm(file_ids, desc="Transcript-based diarization"):
        segments = segments_index[file_id]
        print(segments[:3])  # Print first 3 segments for this file_id
        # RTTM path
        rttm_path = config.DIARIZATION_DIR / f"{file_id}.rttm"
        if not rttm_path.exists():
            write_rttm(file_id, segments, rttm_path)

        # Audio path in TRAIN_AUDIO_DIR
        audio_path = find_audio_for_id(file_id)
        if not audio_path.exists():
            print(f"[DIAR] No matching audio found in TRAIN_AUDIO_DIR for ID={file_id}")
            continue

        patient_out = config.PATIENT_AUDIO_DIR / f"{file_id}_patient.wav"
        if not patient_out.exists():
            print(f"[DIAR] Extracting patient audio for ID={file_id}")
            extract_patient_audio(audio_path, segments, patient_out)


if __name__ == "__main__":
    main()
