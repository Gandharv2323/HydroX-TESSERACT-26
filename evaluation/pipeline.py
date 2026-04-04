from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def time_aware_split_indices(n: int, train: float = 0.70, val: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def evaluate_if(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_train[y_train == 0])
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(Xn)

    Xt = scaler.transform(X_test)
    raw = model.decision_function(Xt)
    score_min = float(raw.min())
    score_max = float(raw.max())
    norm = np.clip((raw - score_min) / (score_max - score_min + 1e-12), 0, 1)
    anomaly_score = 1.0 - norm

    y_bin = (y_test != 0).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_bin, anomaly_score)),
        "pr_auc": float(average_precision_score(y_bin, anomaly_score)),
    }


def evaluate_rf_cv(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y[te], pred)),
                "macro_f1": float(f1_score(y[te], pred, average="macro")),
            }
        )

    return {
        "folds": fold_metrics,
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
    }


def evaluate_rf_holdout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def evaluate_lstm_holdout(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_metrics_json(metrics: dict[str, Any], out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
