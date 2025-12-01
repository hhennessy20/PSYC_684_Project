
# 🧠 ADReSS Alzheimer’s Classification Pipeline – Internal Progress Report  
## ✅ Overview
This document summarizes the current working pipeline, major components implemented, and next steps for continuation.  
The project now supports a full end-to-end Alzheimer’s classification system using:

- Transcript-based **speaker diarization** (pylangacq)
- **Patient-only audio** generation
- **openSMILE eGeMAPS** feature extraction (Python API)
- **Feature aggregation** (utterance → recording)
- **XGBoost baseline classifier**

The pipeline is now stable and reproducible.

---

# 📂 Current Project Structure (Relevant Folders)


```

.  
├── data/  
│ ├── train_Data/ # Raw ADReSS dataset (CC/CD)  
│ ├── train_audio/ # Prepared SXXX.wav copies  
│ ├── patient_audio/ # Extracted PAR-only audio  
│ ├── diarization/ # RTTM files  
│ ├── features_raw/ # openSMILE CSVs  
│ ├── features_agg/ # Final ML-ready features  
│ ├── labels_train.csv # (ID, label) mapping  
│  
└── src/  
├── diarization/  
│ ├── diarize.py # New pylangacq diarizer  
│ ├── audio_prep.py  
│ └── role_heuristics.py  
│  
├── features/  
│ ├── extract_egemaps.py # openSMILE extraction  
│ └── aggregate_features.py  
│  
├── models/  
│ ├── xgb_baseline.py # XGBoost pipeline  
│  
├── config.py # Global paths + settings

```

---

# 🚀 Pipeline Status

## 1️⃣ Data Preparation — **Complete**
**Script:** `data/prepare_adress_train.py`

What it does:
- Copies ADReSS `cc/` and `cd/` WAV files → `data/train_audio/`
- Generates `labels_train.csv`
- Ensures consistent file IDs (`SXXX`)

**Status:** Stable

---

## 2️⃣ Audio Standardization — **Complete**
**Script:** `src/diarization/audio_prep.py`

What it does:
- Converts WAVs to mono
- Normalizes sample rate (16 kHz)
- Ensures consistent amplitude range

**Status:** Stable

---

## 3️⃣ Speaker Diarization (Transcript-Based) — **Complete**
**Script:** `src/diarization/diarize.py`

Key features:
- Uses **pylangacq** to parse `.cha` files (handles multi-line utterances)
- Extracts all segments with timestamps for each speaker
- Produces:
  - **RTTM** files (`data/diarization/SXXX.rttm`)
  - **Patient-only audio** (`SXXX_patient.wav`)
- Handles ms/seconds ambiguity
- Eliminates timestamp inconsistencies

**Status:** Fully working

---

## 4️⃣ Acoustic Feature Extraction (eGeMAPS) — **Complete**
**Script:** `src/features/extract_egemaps.py`

What it does:
- Uses Python **opensmile**
- Extracts **88-dimensional eGeMAPSv02** features
- Writes one CSV per patient audio file → `data/features_raw/`

**Status:** Stable + validated

---

## 5️⃣ Feature Aggregation — **Complete**
**Script:** `src/features/aggregate_features.py`

What it does:
- Reads raw feature CSVs
- Aggregates time sequences → single feature vector per recording
  - Mean
  - Std Dev
  - Percentiles (configurable)
- Saves:
  - `X.npy` — features  
  - `y.npy` — labels  
  - `scaler.pkl` — standardization scaler

**Status:** Working

---

## 6️⃣ XGBoost Baseline — **Complete**
**Script:** `src/models/xgb_baseline.py`

Features:
- Loads `X.npy` + `y.npy`
- Train/val split
- Standardization included
- XGBoost with tuned hyperparameters
- Computes:
  - Accuracy  
  - F1  
  - ROC-AUC  
- Saves model → `xgb_model.json`

**Status:** Working and reproducible

---

# 🧩 Full Connected Pipeline

The full working pipeline is:


```

prepare_adress_train.py  
→ audio_prep.py  
→ diarize.py  
→ extract_egemaps.py  
→ aggregate_features.py  
→ xgb_baseline.py

```

Run in this order:

```bash
# 1. Prepare ADReSS → train_audio + labels
python data/prepare_adress_train.py

# 2. Normalize WAV files
python -m src.diarization.audio_prep

# 3. Transcript-based diarization → RTTM + patient audio
python -m src.diarization.diarize

# 4. Extract eGeMAPS using opensmile
python -m src.features.extract_egemaps

# 5. Aggregate features into final matrix
python -m src.features.aggregate_features

# 6. Train XGBoost model
python -m src.models.xgb_baseline

```

The above pipeline is fully functional and validated end-to-end.