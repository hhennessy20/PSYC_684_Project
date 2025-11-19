from pathlib import Path
from tqdm import tqdm
import pandas as pd
import opensmile

from src import config


def create_smile_extractor() -> opensmile.Smile:
    """
    Create a Smile extractor for eGeMAPSv02 Functionals.
    This gives one aggregated feature row per file.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile


def process_patient_audio(patient_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    smile = create_smile_extractor()

    patient_files = sorted(patient_dir.glob("*_patient.wav"))
    for wav in tqdm(patient_files, desc="opensmile eGeMAPS (Python)"):
        file_id = wav.stem.replace("_patient", "")
        out_csv = out_dir / f"{file_id}_eGeMAPS.csv"
        if out_csv.exists():
            continue

        print(f"[opensmile] {wav.name} -> {out_csv.name}")
        # This returns a DataFrame with a single row of functionals
        df = smile.process_file(str(wav))

        # Optional: add ID as a column
        df["ID"] = file_id

        # Save to CSV; index=False so we don’t carry index as a column
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    process_patient_audio(config.TRAIN_AUDIO_DIR, config.FEATURES_RAW_DIR)
