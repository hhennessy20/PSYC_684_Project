from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

from src import config


def load_features(out_dir: Path):
    X = np.load(out_dir / "X_train_scaled.npy")
    y = np.load(out_dir / "y_train.npy")
    ids = np.load(out_dir / "ids_train.npy")
    return X, y, ids


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return neg / pos


def get_base_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_SEED,
    )


def hyperparam_search(X: np.ndarray, y: np.ndarray, spw: float) -> Dict[str, Any]:
    base_model = get_base_model(spw)

    param_distributions = {
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 2, 5, 10],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        "n_estimators": [100, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_SEED,
    )

    search.fit(X, y)
    print(f"[HP] Best AUC: {search.best_score_:.4f}")
    print(f"[HP] Best params: {search.best_params_}")
    return search.best_params_


def train_final_model(
    X: np.ndarray, y: np.ndarray, best_params: Dict[str, Any], spw: float
) -> XGBClassifier:
    model = get_base_model(spw)
    model.set_params(**best_params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    aucs, bals, f1s = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_proba))
        bals.append(balanced_accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))

        print(
            f"[CV] Fold {fold}: AUC={aucs[-1]:.3f}, "
            f"BalAcc={bals[-1]:.3f}, F1={f1s[-1]:.3f}"
        )

    print(f"[CV] Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"[CV] Mean BalAcc: {np.mean(bals):.3f}")
    print(f"[CV] Mean F1: {np.mean(f1s):.3f}")

    # retrain on full data
    model = get_base_model(spw)
    model.set_params(**best_params)
    model.fit(X, y)
    return model


def main():
    X, y, ids = load_features(config.FEATURES_AGG_DIR)
    spw = compute_scale_pos_weight(y)
    print(f"[TRAIN] scale_pos_weight = {spw:.3f}")

    best_params = hyperparam_search(X, y, spw)
    model = train_final_model(X, y, best_params, spw)

    model_path = config.FEATURES_AGG_DIR / "xgb_model.joblib"
    joblib.dump(model, model_path)
    print(f"[TRAIN] Saved final XGB model to {model_path}")


if __name__ == "__main__":
    main()
