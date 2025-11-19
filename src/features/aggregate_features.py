from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

import config


def load_feature_csv(csv_path: Path) -> pd.Series:
    """
    Load one eGeMAPS CSV (already Functionals from opensmile)
    and return the numeric feature row as a Series.
    Assumes a single row per file.
    """
    df = pd.read_csv(csv_path)

    # Optionally drop 'ID' column if we added it
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(f"No numeric columns in {csv_path}")

    # There should be exactly one row; use iloc[0] as the feature vector
    return numeric_df.iloc[0]


def build_feature_matrix(
    raw_dir: Path,
    labels_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X (features), y (labels), ids for TRAIN set,
    using the functionals from opensmile.

    raw_dir: directory with *_eGeMAPS.csv files
    labels_csv: CSV with columns: ID,label
    """
    labels_df = pd.read_csv(labels_csv)
    labels_df["ID"] = labels_df["ID"].astype(str)

    X_list, y_list, id_list = [], [], []

    all_csv = list(raw_dir.glob("*_eGeMAPS.csv"))
    print(f"[AGG] Found {len(all_csv)} eGeMAPS CSVs")

    for csv_path in tqdm(all_csv, desc="Building feature matrix"):
        file_id = csv_path.stem.replace("_eGeMAPS", "")

        row = labels_df[labels_df["ID"] == file_id]
        if row.empty:
            # likely a test file or something not in labels; skip for training
            continue

        label = int(row["label"].iloc[0])
        feat_series = load_feature_csv(csv_path)

        X_list.append(feat_series.values)
        y_list.append(label)
        id_list.append(file_id)

    X = np.vstack(X_list)
    y = np.array(y_list)
    ids = np.array(id_list)
    return X, y, ids


def save_features(X: np.ndarray, y: np.ndarray, ids: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X)
    np.save(out_dir / "y_train.npy", y)
    np.save(out_dir / "ids_train.npy", ids)


def fit_and_save_scaler(X: np.ndarray, out_dir: Path) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, out_dir / "scaler.joblib")
    np.save(out_dir / "X_train_scaled.npy", X_scaled)


if __name__ == "__main__":
    X, y, ids = build_feature_matrix(config.FEATURES_RAW_DIR, config.LABELS_CSV)
    save_features(X, y, ids, config.FEATURES_AGG_DIR)
    fit_and_save_scaler(X, config.FEATURES_AGG_DIR)
