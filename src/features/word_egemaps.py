# src/features/word_egemaps.py

"""
Extract eGeMAPS features for EVERY word interval from MFA TextGrids
for both cc (controls) and cd (patients).

Input:
    - TextGrids in:
        data/MFA_output_cc_diarized/*.TextGrid
        data/MFA_output_cd_diarized/*.TextGrid
    - alignment_analysis.csv in each of those dirs
    - patient audio wavs in:
        data/patient_audio/<file_id>.wav  (e.g., S077_patient.wav)

Output:
    - data/features/word_egemaps_diarized.csv

Columns:
    group   (cc or cd)
    file    (e.g., S077_patient)
    speaker (from alignment_analysis)
    word
    start, end, duration
    <eGeMAPS features...>
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from textgrid import TextGrid
import opensmile

from src.config import (
    PATIENT_AUDIO_DIR,
    MFA_CC_DIR,
    MFA_CD_DIR,
    MFA_CC_ALIGN_CSV,
    MFA_CD_ALIGN_CSV,
    WORD_EGEMAPS_CSV,
)

# ---------------------------------------------------------------------
# Global resources
# ---------------------------------------------------------------------

# Load alignment_analysis for cc and cd, if present
def _load_alignment_df() -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    if MFA_CC_ALIGN_CSV.exists():
        df_cc = pd.read_csv(MFA_CC_ALIGN_CSV)
        df_cc["group"] = "cc"
        dfs.append(df_cc)

    if MFA_CD_ALIGN_CSV.exists():
        df_cd = pd.read_csv(MFA_CD_ALIGN_CSV)
        df_cd["group"] = "cd"
        dfs.append(df_cd)

    if not dfs:
        raise FileNotFoundError(
            f"No alignment_analysis.csv found in {MFA_CC_ALIGN_CSV} or {MFA_CD_ALIGN_CSV}"
        )

    return pd.concat(dfs, ignore_index=True)


ALIGN_DF = _load_alignment_df()

# eGeMAPS v02 functionals via opensmile
SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_speaker_for_file(file_id: str) -> Optional[str]:
    """
    Look up speaker ID for a given file (e.g., 'S077_patient') from
    the merged alignment_analysis (cc + cd). We ignore begin/end; just grab the speaker.
    """
    df = ALIGN_DF[ALIGN_DF["file"] == file_id]
    if df.empty:
        return None
    # All rows for a file should have same speaker; take the first.
    return str(df["speaker"].iloc[0])


def get_group_for_file(file_id: str) -> Optional[str]:
    """
    Return 'cc' or 'cd' for this file_id based on alignment_analysis.
    """
    df = ALIGN_DF[ALIGN_DF["file"] == file_id]
    if df.empty:
        return None
    return str(df["group"].iloc[0])


def load_audio_mono(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load wav file as mono float audio and sampling rate.
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def load_words_from_textgrid(tg_path: Path) -> List[Dict]:
    """
    Parse the 'words' tier from a TextGrid and return a list of dicts:
        {'word': 'something', 'start': 1.234, 'end': 1.456, 'duration': 0.222}

    We take ALL word intervals (including pauses if present as labels),
    skipping only degenerate times.
    """
    tg = TextGrid.fromFile(str(tg_path))

    words_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in ["words", "word"]:
            words_tier = tier
            break

    if words_tier is None:
        raise ValueError(f"No words tier found in {tg_path}")

    words: List[Dict] = []
    for interval in words_tier.intervals:
        text = interval.mark.strip()
        start = float(interval.minTime)
        end = float(interval.maxTime)

        # Treat empty as pause token so we don't drop silent gaps
        if text == "":
            text = "PAUSE"

        # Skip degenerate intervals
        if end <= start:
            continue

        words.append(
            {
                "word": text,
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )

    return words


def extract_egemaps_for_segment(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> Optional[Dict[str, float]]:
    """
    Slice a segment [start_sec, end_sec] from full audio and compute eGeMAPS.

    Returns:
        dict of {feature_name: value} or None if segment is invalid.
    """
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)

    if start_idx >= end_idx:
        return None

    seg = audio[start_idx:end_idx]

    try:
        feats_df = SMILE.process_signal(seg, sr)
    except Exception as e:
        print(
            f"[WARN] openSMILE failed on segment {start_sec:.3f}-{end_sec:.3f}: {e}"
        )
        return None

    if feats_df.empty:
        return None

    return feats_df.iloc[0].to_dict()


# ---------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------

def _iter_textgrid_files():
    """
    Yield (group, tg_path) pairs for all TextGrids in cc and cd MFA dirs.
    """
    for tg_path in sorted(MFA_CC_DIR.glob("*.TextGrid")):
        yield "cc", tg_path

    for tg_path in sorted(MFA_CD_DIR.glob("*.TextGrid")):
        yield "cd", tg_path


def extract_word_egemaps_for_file(
    file_id: str, group_hint: Optional[str] = None
) -> pd.DataFrame:
    """
    For a single file (e.g., 'S077_patient'), extract eGeMAPS for ALL word
    intervals from its TextGrid words tier.

    group_hint:
        Optional 'cc'/'cd' from which directory the TextGrid was read.
        We'll cross-check with ALIGN_DF but favor the hint if given.
    """
    wav_path = PATIENT_AUDIO_DIR / f"{file_id}.wav"

    if not wav_path.exists():
        raise FileNotFoundError(f"Wav not found for {file_id}: {wav_path}")

    audio, sr = load_audio_mono(wav_path)
    speaker = get_speaker_for_file(file_id)
    group_from_align = get_group_for_file(file_id)

    group = group_hint or group_from_align

    # Prefer directory based on group hint (if present), else search both.
    if group == "cc":
        tg_dir = MFA_CC_DIR
    elif group == "cd":
        tg_dir = MFA_CD_DIR
    else:
        # Fallback: check each
        if (MFA_CC_DIR / f"{file_id}.TextGrid").exists():
            tg_dir = MFA_CC_DIR
        elif (MFA_CD_DIR / f"{file_id}.TextGrid").exists():
            tg_dir = MFA_CD_DIR
        else:
            raise FileNotFoundError(
                f"No TextGrid found for {file_id} in {MFA_CC_DIR} or {MFA_CD_DIR}"
            )

    tg_path = tg_dir / f"{file_id}.TextGrid"
    words = load_words_from_textgrid(tg_path)

    rows = []
    for w in words:
        feats = extract_egemaps_for_segment(
            audio, sr, w["start"], w["end"]
        )
        if feats is None:
            continue

        row = {
            "group": group,  # cc or cd
            "file": file_id,
            "speaker": speaker,
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "duration": w["duration"],
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract_all_word_egemaps() -> pd.DataFrame:
    """
    Iterate over ALL TextGrids in MFA cc and cd dirs and build the full
    word-level eGeMAPS table, then write to WORD_EGEMAPS_CSV.
    """
    WORD_EGEMAPS_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    tg_files = list(_iter_textgrid_files())
    print(f"[INFO] Found {len(tg_files)} TextGrid files (cc + cd)")

    for group, tg_path in tg_files:
        file_id = tg_path.stem  # e.g., 'S077_patient'
        try:
            df_file = extract_word_egemaps_for_file(file_id, group_hint=group)
        except Exception as e:
            print(f"[ERROR] Failed for {file_id} ({group}): {e}")
            continue

        if df_file.empty:
            print(f"[WARN] No valid words for {file_id} ({group})")
            continue

        all_rows.append(df_file)
        print(f"[INFO] {file_id} ({group}): {len(df_file)} word rows")

    if not all_rows:
        print("[WARN] No word rows extracted for any file.")
        return pd.DataFrame()

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(WORD_EGEMAPS_CSV, index=False)
    print(f"[INFO] Wrote word-level eGeMAPS to {WORD_EGEMAPS_CSV}")

    return full_df

if __name__ == "__main__":
    extract_all_word_egemaps()