# XGBoost Models

XGBoost classifiers for Alzheimer's detection using eGeMAPS features.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### Baseline Model

```bash
python -m src.models.xgb_baseline
```

Trains a baseline XGBoost model with hyperparameter search.

### Word-Level eGeMAPS Model

```bash
python -m src.models.train_xgb_word_egemaps
```

Trains XGBoost on word-level eGeMAPS features extracted from MFA alignments.

### Phoneme-Level eGeMAPS Model

```bash
python -m src.models.train_xgb_phoneme_egemaps
```

Trains XGBoost on phoneme-level eGeMAPS features extracted from MFA alignments.

### TreeSHAP Explanations

```bash
python -m src.models.explain_xgb_word_treeshap
```

Generates TreeSHAP explanations for the word-level model.

## Output

- `data/models/` - Saved model files (`.joblib`)
- `data/` - Feature importance and SHAP analysis CSVs

