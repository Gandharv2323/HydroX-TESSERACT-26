"""
training/train_all.py — One-shot training orchestrator.

Execution order (Spec §19):
  1. Generate synthetic data
  2. Train Isolation Forest  (unsupervised, normal only)
  3. Train 5-class Random Forest (supervised, all classes)
  4. Train LSTM RUL predictor
  5. Save all artefacts to models/

Run:  python training/train_all.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# Make project root importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from training.generate_data import generate
from pipeline.fault_classifier import FaultClassifier
from pipeline.rul_lstm import RULPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_all")

MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_isolation_forest(X_features: np.ndarray, y_class: np.ndarray) -> dict:
    """
    Spec §6: Train IF on NORMAL class only (class_id=0).
    Normalise scores to [0, 1] using training-set min/max.
    """
    log.info("─" * 60)
    log.info("Training Isolation Forest (unsupervised, normal data only)...")

    X_normal = X_features[y_class == 0]
    log.info(f"  Normal samples: {len(X_normal)}")

    scaler = StandardScaler()
    X_n_s  = scaler.fit_transform(X_normal)

    t0  = time.perf_counter()
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_n_s)
    elapsed = time.perf_counter() - t0
    log.info(f"  Trained in {elapsed:.2f}s")

    # Score calibration on full training set
    X_all_s = scaler.transform(X_features)
    raw_scores = iso.decision_function(X_all_s)   # higher = more normal
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    norm_scores = np.clip((raw_scores - score_min) / (score_max - score_min + 1e-12), 0, 1)

    bundle = {
        "model":      iso,
        "scaler":     scaler,
        "score_min":  score_min,
        "score_max":  score_max,
    }
    out_path = MODELS_DIR / "isolation_forest.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(bundle, fh)
    log.info(f"  Saved → {out_path}")

    # ROC-AUC: anomaly = class != 0
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    y_bin = (y_class != 0).astype(int)     # 1 = fault, 0 = normal
    # score for fault = 1 - norm_score  (lower IF score → more anomalous)
    auc  = roc_auc_score(y_bin, 1.0 - norm_scores)
    ap   = average_precision_score(y_bin, 1.0 - norm_scores)
    log.info(f"  ROC-AUC={auc:.4f}  Avg-Precision={ap:.4f}")

    return {"roc_auc": round(auc, 4), "avg_precision": round(ap, 4)}


def train_random_forest(X_features: np.ndarray, y_class: np.ndarray) -> dict:
    """Spec §7: Train 5-class RandomForest classifier."""
    log.info("─" * 60)
    log.info("Training 5-class Random Forest classifier...")

    clf = FaultClassifier()
    result = clf.fit(X_features, y_class, n_estimators=200, max_depth=12)
    out_path = MODELS_DIR / "fault_classifier.pkl"
    clf.save(out_path)
    log.info(f"  train_accuracy={result['train_accuracy']}")
    return result


def train_lstm(X_windows: np.ndarray, y_rul: np.ndarray) -> dict:
    """Perfect Architecture v3: BiLSTM+SensorGate+Attention, log-space targets."""
    log.info("─" * 60)
    log.info("Training LSTM v3 (Perfect Architecture — log-space + WarmRestarts)...")

    predictor = RULPredictor(
        input_size    = X_windows.shape[2],
        hidden_size   = 128,
        num_layers    = 2,
        dropout       = 0.3,
        bidirectional = True,
        use_attention = True,
        log_targets   = True,    # ablation fix: compresses 5-600h to 1.6-6.4 range
    )
    result = predictor.fit(
        X_windows,
        y_rul,
        epochs      = 120,
        lr          = 3e-4,
        patience    = 20,
        batch_size  = 128,
        huber_delta = 0.5,      # log-space delta — aggressive outlier robustness
        label_noise = 0.02,     # reduced: log-space compresses noise naturally
        t0_restart  = 30,       # WarmRestarts: restart every 30 epochs
    )
    out_path = MODELS_DIR / "rul_lstm.pt"
    predictor.save(out_path)
    log.info(
        f"  MAE={result.get('val_mae_h')}h  RMSE={result.get('val_rmse_h')}h  "
        f"Coverage={result.get('coverage', 0):.1%}"
    )
    return result




def main() -> None:
    t_start = time.perf_counter()

    log.info("=" * 60)
    log.info("  Intelligent Predictive Maintenance — Training Pipeline")
    log.info("=" * 60)

    # Step 1: Generate data
    log.info("Step 1/4 — Generating synthetic training data...")
    X_wins, y_cls, y_rul, X_feats = generate(n_per_class=300)

    # Step 2: Isolation Forest
    log.info("Step 2/4 — Isolation Forest...")
    if_metrics = train_isolation_forest(X_feats, y_cls)

    # Step 3: 5-class RF
    log.info("Step 3/4 — 5-class Random Forest...")
    rf_metrics = train_random_forest(X_feats, y_cls)

    # Step 4: LSTM
    log.info("Step 4/4 — LSTM RUL predictor...")
    try:
        lstm_metrics = train_lstm(X_wins, y_rul)
    except Exception as exc:
        log.warning(f"  LSTM training skipped: {exc}")
        lstm_metrics = {"note": str(exc)}

    total = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(f"  Training complete in {total:.1f}s")
    log.info(f"  IF    → models/isolation_forest.pkl  (AUC={if_metrics['roc_auc']})")
    log.info(f"  RF    → models/fault_classifier.pkl  (acc={rf_metrics['train_accuracy']})")
    log.info(f"  LSTM  → models/rul_lstm.pt")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
