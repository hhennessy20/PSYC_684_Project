from pathlib import Path
import shutil
from typing import List, Tuple

import pandas as pd
import sys

import src.config


ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config  # now safe


def list_wavs_with_label(dir_path: Path, label: int) -> List[Tuple[Path, str, int]]:
    """
    Returns list of (wav_path, id_str, label)
    id_str = stem of wav file (e.g., "S001")
    """
    entries = []
    # Handle both .wav and .WAV just in case
    for pattern in ("*.wav", "*.WAV"):
        for wav in sorted(dir_path.glob(pattern)):
            file_id = wav.stem
            entries.append((wav, file_id, label))
    return entries


def prepare_train_audio_and_labels() -> None:
    """
    1) Copy ADReSS wavs from cc/cd into data/train_audio/
    2) Create data/labels_train.csv with ID,label
       cc -> 0 (control), cd -> 1 (dementia)
    """
    train_audio_dir = config.TRAIN_AUDIO_DIR
    train_audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PREP] Using ADRESS_CC_DIR = {config.ADRESS_CC_DIR}")
    print(f"[PREP] Using ADRESS_CD_DIR = {config.ADRESS_CD_DIR}")
    print(f"[PREP] CC dir exists: {config.ADRESS_CC_DIR.exists()}")
    print(f"[PREP] CD dir exists: {config.ADRESS_CD_DIR.exists()}")

    cc_list = list_wavs_with_label(config.ADRESS_CC_DIR, label=0)
    cd_list = list_wavs_with_label(config.ADRESS_CD_DIR, label=1)

    print(f"[PREP] Found {len(cc_list)} control (cc) recordings.")
    print(f"[PREP] Found {len(cd_list)} dementia (cd) recordings.")

    all_rows = []

    for wav_path, file_id, label in cc_list + cd_list:
        dst = train_audio_dir / f"{file_id}.wav"

        if not dst.exists():
            print(f"[PREP] Copying {wav_path} -> {dst}")
            shutil.copy2(wav_path, dst)

        all_rows.append({"ID": file_id, "label": label})

    if not all_rows:
        print(
            "[PREP] ERROR: No WAV files found in cc/cd folders. "
            "Check that data/train_Data/Full_wave_enhanced_audio/cc and cd contain audio."
        )
        return

    labels_df = pd.DataFrame(all_rows)
    labels_df = labels_df.sort_values("ID").reset_index(drop=True)

    labels_csv_path = config.LABELS_CSV
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_csv_path, index=False)

    print(f"[PREP] Wrote labels to {labels_csv_path}")
    print(f"[PREP] Total recordings: {len(all_rows)}")


if __name__ == "__main__":
    prepare_train_audio_and_labels()
