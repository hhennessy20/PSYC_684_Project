from pathlib import Path

# Root of the project: PSYC_684_Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# -----------------------------
# ADReSS dataset locations
# -----------------------------

# ADReSS train root inside ./data/train_Data
ADRESS_TRAIN_ROOT = PROJECT_ROOT / "data" / "train_Data"

# Full-wave audio (cc / cd)
ADRESS_FULL_WAVE_DIR = ADRESS_TRAIN_ROOT / "Full_wave_enhanced_audio"
ADRESS_CC_DIR = ADRESS_FULL_WAVE_DIR / "cc"
ADRESS_CD_DIR = ADRESS_FULL_WAVE_DIR / "cd"

# Diarized full-wave audio (cc / cd) - patient-only segments
ADRESS_DIARIZED_DIR = ADRESS_TRAIN_ROOT / "Diarized_full_wave_enhanced_audio"
ADRESS_DIARIZED_CC_DIR = ADRESS_DIARIZED_DIR / "cc"
ADRESS_DIARIZED_CD_DIR = ADRESS_DIARIZED_DIR / "cd"

# Normalized audio chunks (cc / cd)
ADRESS_NORMALIZED_DIR = ADRESS_TRAIN_ROOT / "Normalised_audio-chunks"
ADRESS_NORMALIZED_CC_DIR = ADRESS_NORMALIZED_DIR / "cc"
ADRESS_NORMALIZED_CD_DIR = ADRESS_NORMALIZED_DIR / "cd"

# Transcriptions (.cha) (cc / cd)
ADRESS_TRANSCRIPTION_DIR = ADRESS_TRAIN_ROOT / "transcription"
ADRESS_TRANSCRIPT_CC_DIR = ADRESS_TRANSCRIPTION_DIR / "cc"
ADRESS_TRANSCRIPT_CD_DIR = ADRESS_TRANSCRIPTION_DIR / "cd"

# -----------------------------
# Project data directories
# -----------------------------
# Everything under ./data
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"     # filled by prepare_adress_train.py
TEST_AUDIO_DIR = DATA_DIR / "test_audio"       # optional / future use
PATIENT_AUDIO_DIR = DATA_DIR / "patient_audio" # diarization output (PAR only)
DIARIZATION_DIR = DATA_DIR / "diarization"     # RTTM-like files
FEATURES_RAW_DIR = DATA_DIR / "features_raw"   # opensmile CSVs
FEATURES_AGG_DIR = DATA_DIR / "features_agg"   # X, y, scaler, model, etc.

LABELS_CSV = DATA_DIR / "labels_train.csv"

MODELS_DIR = DATA_DIR / "models"
MODEL_CKPT_PATH = MODELS_DIR / "best_adress_cnn.pt"

# -----------------------------
# Audio / training settings
# -----------------------------
TARGET_SR = 16000
TARGET_CHANNELS = 1
#FFMPEG_BIN = "/usr/bin/ffmpeg" 

RANDOM_SEED = 42
N_JOBS = -1

