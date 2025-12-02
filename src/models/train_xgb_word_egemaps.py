# src/models/train_xgb_word_egemaps.py

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.config import WORD_EGEMAPS_CSV, DATA_DIR


# -------------------- LABELS -------------------- #

def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load labels_train.csv and map label strings to binary.

    We try to be robust:
      - auto-detect ID column name
      - auto-detect label column name
      - infer a binary mapping from the unique label values

    Returns a DataFrame with columns: ['id', 'y']
    """
    df = pd.read_csv(labels_path)

    if df.empty:
        raise ValueError(f"{labels_path} is empty – check the file.")

    # --- detect ID column ---
    id_candidates = ["id", "ID", "subject", "Subject", "SUBJECT_ID", "subject_id", "ID_SUBJECT"]
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(
            f"Could not find an ID column in {labels_path}. "
            f"Tried: {id_candidates}. Found columns: {list(df.columns)}"
        )
    if id_col != "id":
        df = df.rename(columns={id_col: "id"})

    # --- detect label column ---
    label_candidates = ["label", "Label", "diagnosis", "Diagnosis", "group", "Group", "dx", "DX"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            f"Could not find a label column in {labels_path}. "
            f"Tried: {label_candidates}. Found columns: {list(df.columns)}"
        )
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # --- infer mapping from unique label values ---
    unique_labels = sorted(df["label"].dropna().unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected exactly 2 unique label values for binary classification, "
            f"but found {len(unique_labels)}: {unique_labels}"
        )

    # Map first value -> 0, second -> 1 (we'll print what that means)
    label_to_int = {unique_labels[0]: 0, unique_labels[1]: 1}
    print(f"[INFO] Inferred label mapping: {label_to_int}")

    df["y"] = df["label"].map(label_to_int)

    # Keep only id + y
    df_out = df[["id", "y"]].copy()
    return df_out


# -------------------- FEATURES -------------------- #

def load_word_egemaps() -> pd.DataFrame:
    df = pd.read_csv(WORD_EGEMAPS_CSV)
    # keep as-is; we’ll derive subject_id in prepare_dataset()
    return df


def _try_merge_with_subject_id(feat_df: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Try several patterns to derive subject_id from feat_df['file'] and
    merge with labels_df['id']. Return the first non-empty merge.
    """

    file_series = feat_df["file"].astype(str)

    # candidate 1: 'S077_patient' -> 'S077'
    cand1 = file_series.str.split("_").str[0]

    # candidate 2: extract 'S###' via regex
    cand2 = file_series.str.extract(r'(S\d+)')[0]

    # candidate 3: digits only; maybe labels use '077' instead of 'S077'
    digits = file_series.str.extract(r'(\d+)')[0]
    if digits.notna().all():
        # if labels look like 'S001', prefix 'S'
        if labels_df["id"].astype(str).str.startswith("S").all():
            cand3 = "S" + digits
        else:
            cand3 = digits
    else:
        cand3 = None

    candidates = [("split_prefix", cand1), ("regex_Snum", cand2)]
    if cand3 is not None:
        candidates.append(("digits_based", cand3))

    for name, sid in candidates:
        tmp = feat_df.copy()
        tmp["subject_id"] = sid

        merged = tmp.merge(
            labels_df,
            left_on="subject_id",
            right_on="id",
            how="inner",
            validate="many_to_one",
        )
        if len(merged) > 0:
            print(f"[INFO] Using subject_id mapping '{name}': merged {len(merged)} rows.")
            return merged, name

    # If nothing worked, print some debug info and fail explicitly
    print("[ERROR] Could not match any subject_id pattern to labels.")
    print("Example feature 'file' values:", file_series.unique()[:10])
    print("Example label 'id' values:", labels_df['id'].astype(str).unique()[:10])
    raise RuntimeError("No overlap between word_egemaps_diarized.csv and labels_train.csv IDs.")


def prepare_dataset():
    """
    Join word eGeMAPS with subject-level labels.

    Returns:
        X (DataFrame of features), y (ndarray of labels), feature_names (list)
    """
    labels_path = DATA_DIR / "labels_train.csv"
    labels_df = load_labels(labels_path)
    feat_df = load_word_egemaps()

    if feat_df.empty:
        raise RuntimeError(f"{WORD_EGEMAPS_CSV} is empty – run word_egemaps extraction first.")

    merged, mapping_name = _try_merge_with_subject_id(feat_df, labels_df)

    print(f"[INFO] Merged {len(merged)} word rows with labels using mapping '{mapping_name}'.")

    # meta columns NOT to feed into XGBoost
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
        "ID",       # opensmile's ID column, if present
        "name",     # (just in case)
    ]

    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].copy()
    y = merged["y"].values.astype(int)

    X = X.fillna(0.0)

    print(f"[INFO] Feature matrix shape: {X.shape}, labels shape: {y.shape}")
    # If still empty, bail early with a clear message
    if X.shape[0] == 0:
        raise RuntimeError("After merging, no rows left for training. Check ID formats in labels_train.csv and word_egemaps CSV.")

    return X, y, feature_cols


# -------------------- MODEL -------------------- #

def build_model():
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
