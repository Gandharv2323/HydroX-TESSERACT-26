"""
training/train_real_csv.py
==========================
Train Isolation Forest + Random Forest classifier on the real Kaggle
pump-sensor-data.csv (220 k rows, 52 sensors, 3 labels).

Design decisions
----------------
* Temporal split (no shuffling) — 70 % train / 15 % val / 15 % test.
* NO class balancing — the natural imbalance (NORMAL 93%, RECOVERING 6%,
  BROKEN <0.01%) is preserved to reflect real-world conditions.
* Only sensor columns are used (drop timestamp, Unnamed index, label).
* NaN columns are dropped; remaining NaNs are forward-filled then 0-filled.
* Saves artefacts to  models/real/  so they don't overwrite the synthetic ones.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

# Make parent importable when run standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LABEL_COL      = "machine_status"
TIMESTAMP_COL  = "timestamp"
DROP_COLS      = {"Unnamed: 0", "timestamp", "machine_status"}

# Label → int mapping (preserves natural ordering: 0=normal, 1=recovering, 2=broken)
LABEL_MAP: dict[str, int] = {
    "NORMAL":     0,
    "RECOVERING": 1,
    "BROKEN":     2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Anomaly = anything that is NOT normal
ANOMALY_LABELS = {"RECOVERING", "BROKEN"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stratified_temporal_split(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified temporal split: each class is split 70/15/15 independently
    in chronological order, then the index sets are merged and sorted.

    This guarantees:
    - ALL classes appear in every split.
    - Temporal ordering is preserved WITHIN each class.
    - The overall dataset proportions are maintained (no artificial balancing).
    """
    unique_classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]   # already in chronological order
        n_cls = len(cls_indices)
        t_end = int(n_cls * train_ratio)
        v_end = int(n_cls * (train_ratio + val_ratio))

        train_idx.extend(cls_indices[:t_end].tolist())
        val_idx.extend(  cls_indices[t_end:v_end].tolist())
        test_idx.extend( cls_indices[v_end:].tolist())

    # Sort each split so rows stay in dataset time order
    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx),   dtype=np.int64),
        np.array(sorted(test_idx),  dtype=np.int64),
    )


def load_and_clean(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, drop unusable cols, ffill NaNs.
    Returns (X: DataFrame of sensor floats, y: Series of int labels).
    """
    print(f"[real_csv] Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"[real_csv] Raw shape: {df.shape}")

    # Drop useless columns
    sensor_cols = [c for c in df.columns if c not in DROP_COLS]

    # Drop columns that are entirely NaN
    all_nan = [c for c in sensor_cols if df[c].isna().all()]
    if all_nan:
        print(f"[real_csv] Dropping all-NaN columns: {all_nan}")
        sensor_cols = [c for c in sensor_cols if c not in all_nan]

    X = df[sensor_cols].copy()

    # Forward-fill then zero-fill remaining NaNs
    X = X.ffill().fillna(0.0).astype(np.float32)

    # Encode labels
    raw_labels = df[LABEL_COL].str.upper().map(LABEL_MAP)
    assert not raw_labels.isna().any(), "Unknown label values found"
    y = raw_labels.astype(np.int32)

    print(f"[real_csv] Features: {X.shape[1]}  |  Rows: {len(X)}")
    vc = pd.Series(df[LABEL_COL].str.upper()).value_counts()
    for lbl, cnt in vc.items():
        print(f"           {lbl:<12} {cnt:>7}  ({100*cnt/len(df):.2f}%)")

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training functions
# ──────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Train IF on NORMAL-only training samples (unsupervised anomaly detector).
    Returns evaluation metrics.
    """
    normal_mask = y_train == LABEL_MAP["NORMAL"]
    X_normal    = X_train[normal_mask]
    contamination = max(0.001, float((y_train != 0).sum() / len(y_train)))

    print(f"\n[IF] Fitting on {len(X_normal):,} normal samples  "
          f"(contamination={contamination:.4f}) …")

    clf = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_normal)

    # Evaluate on val set
    scores   = -clf.score_samples(X_val)          # higher = more anomalous
    preds_bin = (clf.predict(X_val) == -1).astype(int)  # -1 = anomaly
    true_bin  = (y_val != 0).astype(int)

    # Compute AUC-ROC
    try:
        auc = roc_auc_score(true_bin, scores)
    except Exception:
        auc = float("nan")

    metrics: dict[str, Any] = {
        "val_auc_roc":    round(float(auc), 4),
        "contamination":  round(contamination, 4),
        "n_train_normal": int(len(X_normal)),
        "n_val":          int(len(y_val)),
        "val_anomaly_rate": round(float(true_bin.mean()), 4),
    }
    print(f"[IF]  Val AUC-ROC : {auc:.4f}")
    print(f"[IF]  Val anomaly rate (true): {true_bin.mean():.4f}")

    path = out_dir / "isolation_forest.pkl"
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"[IF]  Saved → {path}")

    return metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Train RF 3-class classifier (NORMAL / RECOVERING / BROKEN).
    Evaluates on val set (which has anomalies) and separately on test set.
    class_weight='balanced' reweights internally without resampling.
    """
    print(f"\n[RF] Training 3-class classifier on {len(X_train):,} samples "
          f"(class_weight=balanced) …")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",    # handles imbalance without resampling
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    def _eval_split(X: np.ndarray, y: np.ndarray, split_name: str) -> dict[str, Any]:
        unique_classes = np.unique(y)
        target_names_here = [ID_TO_LABEL[i] for i in sorted(unique_classes)]
        y_pred = clf.predict(X)
        y_pred_filtered = np.where(np.isin(y_pred, unique_classes), y_pred, unique_classes[0])

        rep = classification_report(
            y, y_pred_filtered,
            labels=unique_classes,
            target_names=target_names_here,
            output_dict=True,
            zero_division=0,
        )
        print(f"\n[RF] {split_name} classification report:")
        print(classification_report(
            y, y_pred_filtered,
            labels=unique_classes,
            target_names=target_names_here,
            zero_division=0,
        ))
        cm = confusion_matrix(y, y_pred_filtered, labels=unique_classes)
        print(f"[RF] {split_name} confusion matrix:")
        print(cm)
        return {
            "macro_f1":    round(float(rep["macro avg"]["f1-score"]), 4),
            "accuracy":    round(float(rep["accuracy"]), 4),
            "n_samples":   int(len(y)),
            "class_dist":  {ID_TO_LABEL[i]: int((y == i).sum()) for i in sorted(ID_TO_LABEL)},
            "confusion_matrix": cm.tolist(),
        }

    metrics_val  = _eval_split(X_val,  y_val,  "Validation")
    metrics_test = _eval_split(X_test, y_test, "Test      ")

    # Top feature importances
    fi = sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:10]
    print("\n[RF] Top-10 features:")
    for name, imp in fi:
        print(f"   {name:<20}  {imp:.4f}")

    metrics: dict[str, Any] = {
        "val":  metrics_val,
        "test": metrics_test,
        "note": "Test split is temporally last 15% — may contain fewer anomaly classes",
    }

    path = out_dir / "rf_classifier_real.pkl"
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"\n[RF]  Saved → {path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def train_on_real_csv(
    csv_path: str | Path,
    out_dir:  str | Path = Path("models/real"),
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    X_df, y = load_and_clean(csv_path)
    feature_names = X_df.columns.tolist()
    X = X_df.values    # numpy array (N, n_sensors)
    y_arr = y.values

    n = len(X)
    print(f"\n[split] Total={n:,} rows — applying stratified 70/15/15 temporal split …")
    print(f"         (each class split independently to ensure all classes in every split)")
    train_idx, val_idx, test_idx = _stratified_temporal_split(y_arr)

    X_train, y_train = X[train_idx], y_arr[train_idx]
    X_val,   y_val   = X[val_idx],   y_arr[val_idx]
    X_test,  y_test  = X[test_idx],  y_arr[test_idx]

    def _dist(y_split: np.ndarray) -> str:
        parts = []
        for lbl, cid in sorted(LABEL_MAP.items()):
            cnt = int((y_split == cid).sum())
            pct = 100.0 * cnt / max(1, len(y_split))
            parts.append(f"{lbl}={cnt:,}({pct:.1f}%)")
        return '  '.join(parts)

    print(f"  Train : {len(train_idx):>7,} rows  | {_dist(y_train)}")
    print(f"  Val   : {len(val_idx):>7,} rows  | {_dist(y_val)}")
    print(f"  Test  : {len(test_idx):>7,} rows  | {_dist(y_test)}")

    # 2. Save test split for the TestDatasetPlayer
    np.save(out_dir / "X_test.npy",  X_test.astype(np.float32))
    np.save(out_dir / "y_test.npy",  y_test.astype(np.int32))
    print(f"\n[split] Test split saved → {out_dir}/X_test.npy  ({len(X_test):,} rows)")

    # 3. Train models
    metrics_if = train_isolation_forest(X_train, y_train, X_val, y_val, out_dir)
    metrics_rf = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, out_dir)

    # 4. Save metadata
    meta: dict[str, Any] = {
        "csv_path":       str(csv_path),
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "label_map":      LABEL_MAP,
        "split": {
            "total": n,
            "train": len(train_idx),
            "val":   len(val_idx),
            "test":  len(test_idx),
            "balanced": False,
        },
        "isolation_forest": metrics_if,
        "random_forest":    metrics_rf,
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{'='*55}")
    print(f"  Real-CSV training complete")
    print(f"  IF  AUC-ROC      : {metrics_if['val_auc_roc']}")
    print(f"  RF  Val Macro-F1 : {metrics_rf['val']['macro_f1']}")
    print(f"  RF  Val Accuracy : {metrics_rf['val']['accuracy']}")
    print(f"  RF  Test Accuracy: {metrics_rf['test']['accuracy']}  (note: test={metrics_rf['test']['class_dist']})")
    print(f"  Artefacts        : {out_dir}/")
    print(f"{'='*55}")

    return meta
