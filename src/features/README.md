# Feature Extraction

Extracts eGeMAPS acoustic features from audio files using openSMILE.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### Extract eGeMAPS (File-Level)

```bash
python -m src.features.extract_egemaps
```

Extracts eGeMAPS functionals for each patient audio file. Output: `data/features_raw/`.

### Word-Level eGeMAPS

```bash
python -m src.features.word_egemaps
```

Extracts eGeMAPS features for each word interval using MFA TextGrids. Output: `data/features/word_egemaps_diarized.csv`.

### Phoneme-Level eGeMAPS

```bash
python -m src.features.phoneme_egemaps
```

Extracts eGeMAPS features for each phoneme interval using MFA TextGrids. Output: `data/features/phoneme_egemaps_diarized.csv`.

### Aggregate Features

```bash
python -m src.features.aggregate_features
```

Aggregates raw features into training matrices. Output: `data/features_agg/`.

## Output

- `data/features_raw/` - Per-file eGeMAPS CSVs
- `data/features/` - Word and phoneme-level feature CSVs
- `data/features_agg/` - Aggregated numpy arrays (X, y, scaler)

