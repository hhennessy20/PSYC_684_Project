# src/models/explain_xgb_word_treeshap.py
"""
TreeSHAP explanations for the word-level XGBoost eGeMAPS model.

This script:
  1) Rebuilds the same X, y, and metadata used for training
  2) Computes TreeSHAP values
  3) Saves:
       - Global feature importance (mean |SHAP| per feature)
       - Per-word segment importance (SHAP total, prediction, label)
       - Sample of full SHAP matrix for deeper analysis

Run from repo root:

    python -m src.models.explain_xgb_word_treeshap

Requirements:

    pip install shap
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import DATA_DIR, WORD_EGEMAPS_CSV
from src.models import train_xgb_word_egemaps as train_word


# ---------------------------------------------------------------------
# Dataset reconstruction with metadata
# ---------------------------------------------------------------------

def build_word_dataset_with_meta():
    """
    Rebuild the joined word-level dataset with:
      - X: feature matrix
      - y: labels
      - feature_names: list of feature columns
      - meta: DataFrame with subject_id, file, word, start, end, duration, y

    This reuses the *same* logic as training (load_labels, load_word_egemaps,
    _try_merge_with_subject_id), so SHAP is computed on the same X.
    """
    labels_path = DATA_DIR / "labels_train.csv"
    labels_df = train_word.load_labels(labels_path)
    feat_df = train_word.load_word_egemaps()

    if feat_df.empty:
        raise RuntimeError(f"{WORD_EGEMAPS_CSV} is empty – run word_egemaps extraction first.")

    merged, mapping_name = train_word._try_merge_with_subject_id(feat_df, labels_df)
    print(f"[INFO] (SHAP) Merged {len(merged)} word rows with labels using mapping '{mapping_name}'.")

    # Meta columns NOT fed into the model
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
        "ID",       # opensmile ID, if present
        "name",     # just in case
    ]

    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].copy()
    y = merged["y"].values.astype(int)
    X = X.fillna(0.0)

    if X.shape[0] == 0:
        raise RuntimeError("After merging, no rows left for SHAP. Check IDs in word_egemaps + labels_train.csv.")

    # Build a compact meta DF aligned row-wise with X
    meta = merged.copy()
    # Derive subject_id if not already present (should be set by _try_merge_with_subject_id)
    if "subject_id" not in meta.columns:
        meta["subject_id"] = meta["file"].astype(str).str.split("_").str[0]

    meta_df = meta[[
        "subject_id",
        "file",
        "word",
        "start",
        "end",
        "duration",
        "y",
    ]].reset_index(drop=True)

    print(f"[INFO] (SHAP) X shape = {X.shape}, y shape = {y.shape}, meta rows = {len(meta_df)}")
    return X, y, feature_cols, meta_df


# ---------------------------------------------------------------------
# TreeSHAP analysis
# ---------------------------------------------------------------------

def run_treeshap_for_word_xgb():
    # --- Load trained model + feature names ---
    model_dir = DATA_DIR / "features_agg" / "word_xgb"
    model_path = model_dir / "xgb_word_model.joblib"
    feat_names_path = model_dir / "xgb_word_feature_names.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained word XGB model not found at {model_path}")
    if not feat_names_path.exists():
        print("[WARN] Feature names .npy not found; using columns from dataset instead.")

    model = joblib.load(model_path)
    print(f"[INFO] Loaded word-level XGB model from {model_path}")

    # --- Rebuild dataset with metadata ---
    X, y, feature_names_from_train, meta_df = build_word_dataset_with_meta()

    # Sanity: if feat_names file exists, ensure they match
    if feat_names_path.exists():
        saved_feature_names = np.load(feat_names_path, allow_pickle=True)
        if list(saved_feature_names) != list(feature_names_from_train):
            print("[WARN] Saved feature names differ from current dataset columns.")
            print("       Using dataset columns for SHAP; check consistency if results look odd.")
        feature_names = list(feature_names_from_train)
    else:
        feature_names = list(feature_names_from_train)

    # --- Choose background for TreeSHAP ---
    n_background = min(1000, len(X))
    background = X.sample(n=n_background, random_state=42)
    print(f"[INFO] Using {n_background} background samples for TreeSHAP.")

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_names=feature_names,
    )

    # --- Compute SHAP values ---
    # If dataset is huge, you can subsample manually here.
    print("[INFO] Computing SHAP values for all word segments...")
    shap_values = explainer.shap_values(X)  # shape: [n_samples, n_features]

    if isinstance(shap_values, list):
        # For binary XGBClassifier, shap_values can be [class0, class1]
        # We'll use class1 (AD-positive) explanations
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # --- Global feature importance (mean |SHAP|) ---
    global_importance = shap_df.abs().mean().sort_values(ascending=False)
    print("[INFO] Top 10 global word-level eGeMAPS features (by mean |SHAP|):")
    print(global_importance.head(10))

    # --- Per-word segment importance ---
    shap_total = shap_df.abs().sum(axis=1)
    shap_total.name = "shap_total"

    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Build per-segment summary DF
    seg_df = meta_df.copy()
    seg_df["y_pred_proba"] = y_pred_proba
    seg_df["y_pred"] = y_pred
    seg_df["shap_total"] = shap_total.values

    # --- Prepare output directory ---
    out_dir = model_dir / "treeshap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Global feature importance
    global_path = out_dir / "word_treeshap_global_feature_importance.csv"
    global_importance.to_frame(name="mean_abs_shap").to_csv(global_path)
    print(f"[INFO] Saved global feature importance to {global_path}")

    # 2) Full per-segment table
    segments_path = out_dir / "word_treeshap_segments.csv"
    seg_df.to_csv(segments_path, index=False)
    print(f"[INFO] Saved per-segment SHAP summary to {segments_path}")

    # 3) Save SHAP values for a sample (for plots / deeper analysis)
    max_sample_for_full = 20000
    if len(X) > max_sample_for_full:
        sample_idx = np.random.RandomState(42).choice(
            len(X), size=max_sample_for_full, replace=False
        )
        shap_sample = shap_values[sample_idx, :]
        meta_sample = seg_df.iloc[sample_idx].reset_index(drop=True)
    else:
        shap_sample = shap_values
        meta_sample = seg_df.reset_index(drop=True)

    shap_sample_path = out_dir / "word_treeshap_shap_values_sampled.npy"
    meta_sample_path = out_dir / "word_treeshap_sample_metadata.csv"

    np.save(shap_sample_path, shap_sample)
    meta_sample.to_csv(meta_sample_path, index=False)

    print(f"[INFO] Saved sampled SHAP matrix to {shap_sample_path}")
    print(f"[INFO] Saved corresponding metadata to {meta_sample_path}")

    print("[INFO] TreeSHAP analysis for word-level XGB completed.")


if __name__ == "__main__":
    run_treeshap_for_word_xgb()
