from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def _f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    return (2.0 * precision * recall) / (precision + recall + 1e-12)


def calibrate_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    method: str = "f1_optimal",
    min_recall: float = 0.90,
) -> dict[str, Any]:
    """
    Calibrate anomaly threshold where score is interpreted as:
      higher score => more anomalous.
    """
    y_true = np.asarray(y_true).astype(int)
    anomaly_scores = np.asarray(anomaly_scores).astype(float)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)

    if method == "f1_optimal":
        # precision/recall arrays are len(thresholds)+1; align by dropping last element
        f1 = _f1(precision[:-1], recall[:-1])
        idx = int(np.argmax(f1))
        threshold = float(pr_thresholds[idx])
        meta = {
            "f1": float(f1[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
        }
    elif method == "recall_guard":
        candidates = np.where(recall[:-1] >= min_recall)[0]
        if len(candidates) == 0:
            idx = int(np.argmax(recall[:-1]))
        else:
            # choose highest precision among recall-satisfying thresholds
            idx = int(candidates[np.argmax(precision[:-1][candidates])])
        threshold = float(pr_thresholds[idx])
        meta = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
        }
    elif method == "quantile":
        # Fallback only; prefer calibrate_threshold_unsupervised when normal-only scores are available.
        threshold = float(np.percentile(anomaly_scores, 95.0))
        meta = {
            "quantile": 95.0,
            "note": "label-agnostic fallback quantile over provided scores",
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)

    return {
        "if_threshold": threshold,
        "method": method,
        "meta": meta,
        "roc_points": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_points": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def save_threshold_config(result: dict[str, Any], out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "if_threshold": float(result["if_threshold"]),
        "method": result["method"],
        "meta": result.get("meta", {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def calibrate_threshold_unsupervised(
    normal_scores: np.ndarray,
    quantile: float = 95.0,
) -> dict[str, Any]:
    """Calibrate IF anomaly threshold from normal-score distribution only.

    Parameters
    ----------
    normal_scores : np.ndarray
        Anomaly scores for known/assumed normal windows.
    quantile : float
        Upper quantile for anomaly trigger (e.g., 95 = top 5% as anomalies).
    """
    scores = np.asarray(normal_scores, dtype=float)
    q = float(np.clip(quantile, 50.0, 99.99))
    threshold = float(np.percentile(scores, q))
    return {
        "if_threshold": threshold,
        "method": "quantile_normal",
        "meta": {
            "quantile": q,
            "n_normal": int(len(scores)),
        },
    }


def load_threshold_config(path: Path | str, default: float = 0.55) -> float:
    p = Path(path)
    if not p.exists():
        return float(default)
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data.get("if_threshold", default))
