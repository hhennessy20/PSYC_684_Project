# PSYC_684_Project

Alzheimer's detection using audio analysis and machine learning.

## Project Structure

```
├── data/
│   ├── train_Data/          # ADReSS challenge dataset (see data/train_Data/README.md)
│   ├── models/              # Saved model checkpoints
│   ├── pdsm_out/            # PDSM output files
│   ├── ppg_out/             # Phoneme posteriorgram outputs
│   └── saliencies/          # GradSHAP saliency maps
├── src/
│   ├── pdsm/                # PDSM analysis pipeline (see src/pdsm/README.md)
│   ├── models/
│   │   ├── cnn/             # CNN training & interpretation (see src/models/cnn/README.md)
│   │   ├── phoneme_posteriorgram/  # PPG extraction (see src/models/phoneme_posteriorgram/README.md)
│   │   └── XGBoost models   # (see src/models/README.md)
│   ├── features/            # Feature extraction (see src/features/README.md)
│   ├── diarization/         # Speaker diarization (see src/diarization/README.md)
│   └── config.py            # Project-wide configuration
├── requirements.txt         # XGBoost/features dependencies
└── src/pdsm/pdsm_environment.yml  # PDSM/CNN conda environment
```

## Environments

This project has **two separate environments**:

### 1. PDSM/CNN Environment (Conda)

For running anything in `src/pdsm/` or `src/models/cnn/`:

```bash
conda env create -f src/pdsm/pdsm_environment.yml
conda activate pdsm
```

### 2. XGBoost/Features Environment (pip)

For running anything in `src/models/` (except cnn), `src/features/`, or `src/diarization/`:

```bash
pip install -r requirements.txt
```

## Sub-READMEs

Each major folder contains its own README with specific instructions:

- `src/pdsm/README.md` - PDSM generation and experiments
- `src/models/README.md` - XGBoost model training
- `src/models/cnn/README.md` - CNN training and GradSHAP interpretation
- `src/models/phoneme_posteriorgram/README.md` - MFA alignment and PPG extraction
- `src/features/README.md` - eGeMAPS feature extraction
- `src/diarization/README.md` - Speaker diarization pipeline
- `data/train_Data/README.md` - ADReSS dataset description
