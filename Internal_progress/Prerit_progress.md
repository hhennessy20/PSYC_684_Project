---

# **Team README – Current Progress & How to Run the Pipeline**

This document is for **internal team use**.
It summarizes what is **already implemented**, what each script does, how to **run the pipeline**, and what still needs to be completed.

---

## ✅ **What’s Done So Far**

### **1. Dataset Integration (ADReSS-IS 2020)**

* Script automatically imports audio files from the ADReSS cc/cd folders.
* Generates:

  * `train_audio/` containing all WAVs
  * `labels_train.csv` with IDs + labels (0=CC, 1=CD)

**Script:** `src/data/prepare_adress_train.py`
**Status:** ✔ Working

---

### **2. Audio Standardization (Pure Python)**

* Every WAV file is normalized to:

  * 16 kHz
  * Mono channel
* Uses `soundfile` + `torchaudio`
* No `ffmpeg` needed.

**Script:** `src/diarization/audio_prep.py`
**Status:** ✔ Working

---

### **3. Speaker Diarization (pyannote.audio)**

* Performs diarization on each full WAV recording.
* Assigns patient role using “longest speaker” heuristic.
* Extracts and concatenates **patient-only speech**.

**Outputs:**

* `patient_audio/ID_patient.wav`
* `diarization/ID.rttm`

**Script:** `src/diarization/diarize.py`
**Status:** ✔ Ongoing (Dependency Issues) 

---

### **4. Acoustic Features (Python openSMILE)**

* Extracts **eGeMAPSv02 Functionals** (1 vector per file).
* No manual config files needed.

**Outputs:**

* `features_raw/ID_eGeMAPS.csv`

**Script:** `src/features/extract_egemaps.py`
**Status:** ✔ Working

---

### **5. Feature Aggregation + Scaling**

* Stacks all extracted feature rows into:

  * `X_train.npy`
  * `y_train.npy`
  * `ids_train.npy`
* Fits a StandardScaler → saves:

  * `scaler.joblib`
  * `X_train_scaled.npy`

**Script:** `src/features/aggregate_features.py`
**Status:** ✔ Working

---

### **6. Baseline AD Classifier (Technique 1)**

* Uses XGBoost:

  * Randomized hyperparameter search
  * Stratified K-fold CV (5-fold)
* Saves trained model:

  * `xgb_model.joblib`

**Script:** `src/models/xgb_baseline.py`
**Status:** ✔ Working

---

## 🚀 **How to Run the Pipeline (Team Commands)**

From project root:

```bash
source adclass/bin/activate   # activate venv

python -m src.data.prepare_adress_train    # copy ADReSS audio + create labels
python -m src.diarization.audio_prep       # standardize audio
python -m src.diarization.diarize          # diarize + extract patient speech
python -m src.features.extract_egemaps     # compute eGeMAPS (opensmile)
python -m src.features.aggregate_features  # build X, y + scaler
python -m src.models.xgb_baseline          # train XGBoost baseline
```

Once these steps run:

* Patient-only audio is ready
* Features are ready
* Model is trained
* We can now test additional models (Technique 2)

---

## 📌 Important Directories (for the team)

```
src/data/train_audio/          -> cc/cd wavs copied from dataset
src/data/patient_audio/        -> diarized patient-only audio
src/data/features_raw/         -> raw eGeMAPS CSVs (1 row/file)
src/data/features_agg/         -> X, y, scaler, and trained model
src/models/                    -> technique1 and placeholder for technique2
```

---

## 🔮 **Next Tasks (Team TODO)**

### **Technique 2 (second classifier) — NEEDS IMPLEMENTATION**

Possible options:

* SVM on eGeMAPS
* MLP classifier
* CNN on log-mel spectrograms
* PPG-based acoustic model
* Acoustic + linguistic fusion

### **Add evaluation metrics**

* Confusion matrix
* ROC-AUC curve
* Balanced accuracy
* Subject-wise predictions CSV

### **Refactoring**

* Add `run_pipeline.py` to automate all stages
* Create helper functions for reuse

### **Testing & Documentation**

* Add tests for each module
* Update README once Technique 2 is done

---

## 👥 **Team Notes**

* All scripts must be run from the **project root** using `python -m ...`
* All imports use the `src` package:

  ```python
  from src import config
  ```
* Do **not** commit large WAV files — only scripts, configs, and models.

---

