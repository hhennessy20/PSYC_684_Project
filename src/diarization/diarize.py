from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline

import config
from .role_heuristics import assign_roles_by_duration


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    role: str = ""


def load_diarization_pipeline() -> Pipeline:
    """
    Requires you to have configured HuggingFace auth if needed.
    Example (bash):
      export HF_TOKEN=...
    Then pyannote will use it internally if required.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    return pipeline


def diarize_file(pipeline: Pipeline, wav_path: Path) -> List[Segment]:
    diarization = pipeline(str(wav_path))

    segments: List[Dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    segments = assign_roles_by_duration(segments)
    return [Segment(**s) for s in segments]


def save_rttm(segments: List[Segment], file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for seg in segments:
            dur = seg.end - seg.start
            f.write(
                f"{file_id} 1 {seg.start:.3f} {dur:.3f} {seg.role} {seg.speaker}\n"
            )


def extract_patient_audio(
    wav_path: Path,
    segments: List[Segment],
    out_path: Path,
) -> None:
    """
    Concatenate all PATIENT segments into one wav file.
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != config.TARGET_SR:
        print(
            f"[WARN] {wav_path.name}: sample rate {sr} != TARGET_SR {config.TARGET_SR}"
        )

    patient_chunks = []
    for seg in segments:
        if seg.role != "PATIENT":
            continue
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        patient_chunks.append(audio[start_sample:end_sample])

    if not patient_chunks:
        print(f"[WARN] No PATIENT segments for {wav_path.name}, skipping.")
        return

    gap = np.zeros(int(0.1 * sr))  # 100ms silence between segments
    concatenated = patient_chunks[0]
    for chunk in patient_chunks[1:]:
        concatenated = np.concatenate([concatenated, gap, chunk])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), concatenated, sr)


def process_directory(audio_dir: Path) -> None:
    pipeline = load_diarization_pipeline()

    wav_paths = list(audio_dir.glob("*.wav"))
    for wav_path in tqdm(wav_paths, desc=f"Diarizing {audio_dir.name}"):
        file_id = wav_path.stem
        out_patient = config.PATIENT_AUDIO_DIR / f"{file_id}_patient.wav"
        out_rttm = config.DIARIZATION_DIR / f"{file_id}.rttm"

        if out_patient.exists() and out_rttm.exists():
            continue

        segments = diarize_file(pipeline, wav_path)
        save_rttm(segments, file_id, out_rttm)
        extract_patient_audio(wav_path, segments, out_patient)


if __name__ == "__main__":
    config.PATIENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)

    process_directory(config.TRAIN_AUDIO_DIR)
    process_directory(config.TEST_AUDIO_DIR)
