from pathlib import Path

# Root folders
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# ADRESS_TRAIN_ROOT = Path("/mnt/D/STudy/RIT HW and Assignments/Graduate Speech Processing (PSYC 684)/Group project/ADReSS-IS2020-train/ADReSS-IS2020-data/train")
ADRESS_TRAIN_ROOT = Path("/home/preritsm/MyWork/PSYC_684_Project/data/train_Data")  # <-- EDIT THIS

# ADReSS audio folders
ADRESS_FULL_WAVE_DIR = ADRESS_TRAIN_ROOT / "Full_wave_enhanced_audio"
ADRESS_CC_DIR = ADRESS_FULL_WAVE_DIR / "cc"   # controls
ADRESS_CD_DIR = ADRESS_FULL_WAVE_DIR / "cd"   # dementia

# Data directories we use in our project
DATA_DIR = SRC_DIR / "data"
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TEST_AUDIO_DIR = DATA_DIR / "test_audio"
PATIENT_AUDIO_DIR = DATA_DIR / "patient_audio"
DIARIZATION_DIR = DATA_DIR / "diarization"
FEATURES_RAW_DIR = DATA_DIR / "features_raw"
FEATURES_AGG_DIR = DATA_DIR / "features_agg"

LABELS_CSV = DATA_DIR / "labels_train.csv"


# Audio standardization
TARGET_SR = 16000
TARGET_CHANNELS = 1
FFMPEG_BIN = "/usr/bin/ffmpeg"  # path to ffmpeg binary

HF_TOKEN = "Add your token"  # HuggingFace token for pyannote models

# Training settings
RANDOM_SEED = 42
N_JOBS = -1
