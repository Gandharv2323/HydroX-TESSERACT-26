"""
evaluation/report.py — Full evaluation report.

Spec §10:
  IF  : ROC-AUC, precision-recall curve
  RF  : confusion matrix, F1-score (per-class + macro)
  LSTM: MAE, RMSE

Run standalone: python evaluation/report.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.features import extract_batch
from pipeline.fault_classifier import FAULT_LABELS
from training.generate_data import generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("report")

MODELS_DIR = _ROOT / "models"
REPORTS_DIR = _ROOT / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_if() -> tuple:
    path = MODELS_DIR / "isolation_forest.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Train first: {path}")
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle["model"], bundle["scaler"], bundle["score_min"], bundle["score_max"]


def _load_rf():
    from pipeline.fault_classifier import FaultClassifier
    clf = FaultClassifier()
    clf.load(MODELS_DIR / "fault_classifier.pkl")
    return clf


def _load_lstm():
    from pipeline.rul_lstm import RULPredictor
    rp = RULPredictor()
    rp.load(MODELS_DIR / "rul_lstm.pt")
    return rp


# ------------------------------------------------------------------
# Individual evaluators
# ------------------------------------------------------------------

def evaluate_isolation_forest(X_feats: np.ndarray, y_cls: np.ndarray) -> dict:
    log.info("Evaluating Isolation Forest...")
    iso, scaler, s_min, s_max = _load_if()
    X_s = scaler.transform(X_feats)
    raw = iso.decision_function(X_s)
    norm_scores = np.clip((raw - s_min) / (s_max - s_min + 1e-12), 0, 1)
    anomaly_score = 1.0 - norm_scores   # 1 = anomaly, 0 = normal

    y_bin = (y_cls != 0).astype(int)
    auc   = roc_auc_score(y_bin, anomaly_score)
    ap    = average_precision_score(y_bin, anomaly_score)

    log.info(f"  IF ROC-AUC = {auc:.4f}")
    log.info(f"  IF Avg-Precision = {ap:.4f}")
    return {"roc_auc": round(auc, 4), "avg_precision": round(ap, 4)}


def evaluate_random_forest(X_feats: np.ndarray, y_cls: np.ndarray) -> dict:
    log.info("Evaluating 5-class Random Forest...")
    clf = _load_rf()

    X_s = clf._scaler.transform(X_feats)    # type: ignore[union-attr]
    y_pred = clf._model.predict(X_s)        # type: ignore[union-attr]

    acc    = accuracy_score(y_cls, y_pred)
    macro_f1 = f1_score(y_cls, y_pred, average="macro")
    cm     = confusion_matrix(y_cls, y_pred)
    report = classification_report(
        y_cls, y_pred,
        target_names=[FAULT_LABELS[c] for c in sorted(FAULT_LABELS)],
    )

    log.info(f"\n{report}")
    log.info(f"  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    log.info(f"  Confusion matrix:\n{cm}")

    return {
        "accuracy":       round(acc, 4),
        "macro_f1":       round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_lstm(X_wins: np.ndarray, y_rul: np.ndarray) -> dict:
    log.info("Evaluating LSTM RUL predictor...")
    try:
        rp = _load_lstm()
    except Exception as exc:
        log.warning(f"  LSTM load failed: {exc}")
        return {"error": str(exc)}

    preds = np.array([rp.predict(X_wins[i]) for i in range(len(X_wins))])
    mae   = float(np.mean(np.abs(preds - y_rul)))
    rmse  = float(np.sqrt(np.mean((preds - y_rul) ** 2)))
    log.info(f"  LSTM MAE={mae:.2f}h  RMSE={rmse:.2f}h")
    return {"mae_hours": round(mae, 2), "rmse_hours": round(rmse, 2)}


# ------------------------------------------------------------------
# Main report
# ------------------------------------------------------------------

def generate_report(n_per_class: int = 200) -> dict:
    log.info("=" * 60)
    log.info("  Generating Evaluation Report")
    log.info("=" * 60)

    log.info("Generating test data...")
    X_wins, y_cls, y_rul, X_feats = generate(n_per_class=n_per_class, seed=99)

    report: dict = {}

    try:
        report["isolation_forest"] = evaluate_isolation_forest(X_feats, y_cls)
    except Exception as exc:
        log.error(f"IF eval error: {exc}")
        report["isolation_forest"] = {"error": str(exc)}

    try:
        report["random_forest"] = evaluate_random_forest(X_feats, y_cls)
    except Exception as exc:
        log.error(f"RF eval error: {exc}")
        report["random_forest"] = {"error": str(exc)}

    try:
        report["lstm_rul"] = evaluate_lstm(X_wins, y_rul)
    except Exception as exc:
        log.error(f"LSTM eval error: {exc}")
        report["lstm_rul"] = {"error": str(exc)}

    # Save JSON (remove non-serialisable items)
    def _clean(d: dict) -> dict:
        return {k: (_clean(v) if isinstance(v, dict) else v)
                for k, v in d.items() if not isinstance(v, str) or k != "classification_report"}

    out_path = REPORTS_DIR / "evaluation_report.json"
    with open(out_path, "w") as fh:
        json.dump(_clean(report), fh, indent=2)
    log.info(f"\nReport saved → {out_path}")

    return report


if __name__ == "__main__":
    generate_report()
