# src/models/train_xgb_word_egemaps.py

"""
Train an XGBoost classifier on word-level eGeMAPS features.

Input:
    - data/features/word_egemaps_diarized.csv
        columns:
            group (cc/cd)
            file (e.g., S077_patient)
            speaker
            word
            start, end, duration
            <eGeMAPS features...>
    - data/labels_train.csv
        columns (assumed):
            id    (e.g., 'S077')
            label ('AD' or 'CN')

Output (saved under data/features_agg/word_xgb/):
    - xgb_word_model.joblib
    - xgb_word_best_params.json
    - xgb_word_feature_names.npy
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

from config import WORD_EGEMAPS_CSV, DATA_DIR


def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load labels_train.csv and map label strings to binary.
    Assumes:
        id: 'S001', 'S077', ...
        label: 'AD' or 'CN'
    """
    df = pd.read_csv(labels_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"labels_train.csv must contain 'id' and 'label' columns. Found: {df.columns}"
        )

    label_map = {"AD": 1, "CN": 0}
    df = df[df["label"].isin(label_map.keys())].copy()
    df["y"] = df["label"].map(label_map)
    return df[["id", "y"]]


def load_word_egemaps() -> pd.DataFrame:
    df = pd.read_csv(WORD_EGEMAPS_CSV)
    # Add subject_id = 'S001' from 'S001_patient'
    df["subject_id"] = df["file"].str.split("_").str[0]
    return df


def prepare_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Join word eGeMAPS with subject-level labels.
    Returns:
        X (DataFrame of features), y (ndarray of labels)
    """
    labels_path = DATA_DIR / "labels_train.csv"
    labels_df = load_labels(labels_path)
    feat_df = load_word_egemaps()

    merged = feat_df.merge(
        labels_df,
        left_on="subject_id",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )

    print(f"[INFO] Merged {len(merged)} word rows with labels.")

    meta_cols = [
        "group",
        "file",
        "speaker",
        "word",
        "start",
        "end",
        "duration",
        "subject_id",
        "id",
        "y",
    ]

    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].copy()
    y = merged["y"].values.astype(int)

    X = X.fillna(0.0)

    print(f"[INFO] Feature matrix shape: {X.shape}, labels shape: {y.shape}")
    return X, y, feature_cols


def build_model():
    """
    Return a base XGBClassifier and a param distribution for random search.
    """
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    return base_model, param_distributions


def train_and_tune_xgb_word():
    X, y, feature_names = prepare_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    base_model, param_distributions = build_model()

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    print("[INFO] Starting RandomizedSearchCV for word-level XGB...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"[INFO] Best params: {search.best_params_}")
    print(f"[INFO] Best CV ROC-AUC: {search.best_score_:.4f}")

    # Evaluate on train and val
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
    ]:
        y_pred_proba = best_model.predict_proba(X_split)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y_split, y_pred)
        roc = roc_auc_score(y_split, y_pred_proba)

        print(f"[{split_name.upper()}] ACC = {acc:.4f}, ROC-AUC = {roc:.4f}")
        if split_name == "val":
            print("[VAL] classification report:")
            print(classification_report(y_split, y_pred))

    # Save model + metadata
    out_dir = DATA_DIR / "features_agg" / "word_xgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_word_model.joblib"
    params_path = out_dir / "xgb_word_best_params.json"
    feat_path = out_dir / "xgb_word_feature_names.npy"

    joblib.dump(best_model, model_path)
    np.save(feat_path, np.array(feature_names))

    with params_path.open("w") as f:
        json.dump(search.best_params_, f, indent=2)

    print(f"[INFO] Saved best word XGB model to {model_path}")
    print(f"[INFO] Saved best params to {params_path}")
    print(f"[INFO] Saved feature names to {feat_path}")


if __name__ == "__main__":
    train_and_tune_xgb_word()
