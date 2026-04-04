from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def train_fusion_model(
    if_scores: np.ndarray,
    rf_fault_probs: np.ndarray,
    latent: np.ndarray,
    y_binary: np.ndarray,
    out_path: Path | str,
) -> dict[str, Any]:
    """Train a meta fusion model y = f(if, rf, h)."""
    if_s = np.asarray(if_scores, dtype=np.float32).reshape(-1, 1)
    rf_s = np.asarray(rf_fault_probs, dtype=np.float32).reshape(-1, 1)
    h = np.asarray(latent, dtype=np.float32)
    y = np.asarray(y_binary, dtype=np.int32)

    X = np.hstack([if_s, rf_s, h])

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X, y)

    p = clf.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }

    payload = {
        "model": clf,
        "feature_layout": {
            "if": 1,
            "rf": 1,
            "latent": int(h.shape[1]),
        },
        "metrics": metrics,
    }

    pth = Path(out_path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    with open(pth, "wb") as fh:
        pickle.dump(payload, fh)

    return metrics


def load_fusion_model(path: Path | str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def predict_fused_score(
    model_bundle: dict[str, Any],
    if_score: float,
    rf_fault_prob: float,
    latent: np.ndarray,
) -> float:
    layout = model_bundle.get("feature_layout", {})
    latent_dim = int(layout.get("latent", 0))
    h = np.asarray(latent, dtype=np.float32).reshape(-1)

    if len(h) < latent_dim:
        h = np.pad(h, (0, latent_dim - len(h)))
    elif len(h) > latent_dim:
        h = h[:latent_dim]

    x = np.concatenate(
        [
            np.array([float(if_score), float(rf_fault_prob)], dtype=np.float32),
            h.astype(np.float32),
        ]
    )[None, :]
    clf = model_bundle["model"]
    return float(np.clip(clf.predict_proba(x)[0, 1], 0.0, 1.0))
