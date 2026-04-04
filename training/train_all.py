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

import argparse
import json
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
from pipeline.buffer import WINDOW_SIZE, SENSORS
from pipeline.features import extract_batch
from pipeline.representation import build_hybrid_feature_batch, HYBRID_DIM
from models.shared_latent import SharedLatentRuntime
from data_pipeline.loader import SensorDataLoader
from data_pipeline.preprocessing import (
    append_missingness_mask_columns,
    build_percentile_bounds,
    clip_outliers_percentile,
    smooth_noise_rolling_mean,
)
from calibration.threshold import (
    calibrate_threshold_unsupervised,
    save_threshold_config,
)
from calibration.fusion_meta import train_fusion_model
from dataset_loader import detect_kaggle_advanced_format, KaggleAdvancedLoader
from evaluation.pipeline import (
    evaluate_if,
    evaluate_rf_cv,
    evaluate_rf_holdout,
    evaluate_lstm_holdout,
    save_metrics_json,
    time_aware_split_indices,
    validate_temporal_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_all")

MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR = _ROOT / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)
with open(_ROOT / "config.json", "r", encoding="utf-8") as _cf:
    _CONFIG = json.load(_cf)


def _windows_from_df(df_internal, step: int = 1, cols: list[str] | None = None):
    use_cols = cols or SENSORS
    values = df_internal[use_cols].to_numpy(dtype=np.float32)
    if len(values) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, len(use_cols)), dtype=np.float32)
    windows = []
    for i in range(0, len(values) - WINDOW_SIZE + 1, step):
        windows.append(values[i:i + WINDOW_SIZE])
    return np.stack(windows, axis=0)


def _load_real_data(real_csv: str | None) -> dict:
    if not real_csv:
        return {"available": False}

    path = Path(real_csv)
    if not path.exists():
        log.warning(f"Real CSV path not found: {path}")
        return {"available": False}

    # ── Kaggle auto-detect ────────────────────────────────────────────────────
    if detect_kaggle_advanced_format(str(path)):
        log.info("Step 0 — Kaggle pump-sensor-data format detected. Using KaggleAdvancedLoader...")
        try:
            return _load_kaggle_real_data(str(path))
        except Exception as exc:
            log.warning(f"  KaggleAdvancedLoader failed: {exc} — falling back to SensorDataLoader")

    # ── Strict internal schema CSV ────────────────────────────────────────────
    log.info("Step 0 — Loading and preprocessing real sensor data (strict schema)...")
    loader = SensorDataLoader()
    strict_df = loader.load_csv(str(path))
    strict_df = append_missingness_mask_columns(strict_df, [f"sensor_{i}" for i in range(1, 8)])

    internal_df = loader.to_internal_schema(strict_df)
    internal_df = clip_outliers_percentile(internal_df, SENSORS)
    internal_df = smooth_noise_rolling_mean(internal_df, SENSORS, window=3)

    bounds = build_percentile_bounds(internal_df, SENSORS)
    log.info(f"  Real rows after cleanup: {len(internal_df)}")

    windows = _windows_from_df(internal_df)
    feats = extract_batch(windows) if len(windows) else np.empty((0, 84), dtype=np.float32)

    y_cls = None
    if "fault_class" in strict_df.columns:
        label_map = {
            "normal": 0, "bearing_fault": 1, "cavitation": 2,
            "dry_run": 3, "misalignment": 4,
        }
        labels = strict_df["fault_class"].astype(str).str.lower().map(label_map)
        labels = labels.dropna().astype(int)
        if len(labels) >= WINDOW_SIZE:
            y_cls = labels.to_numpy()[WINDOW_SIZE - 1:]

    y_rul = None
    if "rul_hours" in strict_df.columns and len(strict_df["rul_hours"].dropna()) >= WINDOW_SIZE:
        y_rul_full = strict_df["rul_hours"].to_numpy(dtype=np.float32)
        y_rul = y_rul_full[WINDOW_SIZE - 1:]

    mask_cols = [f"mask_{i}" for i in range(1, 8)]
    mask_windows = _windows_from_df(strict_df, cols=mask_cols) if all(c in strict_df.columns for c in mask_cols) else None

    return {
        "available": True,
        "strict_df": strict_df,
        "internal_df": internal_df,
        "windows": windows,
        "features": feats,
        "mask_windows": mask_windows,
        "y_cls": y_cls,
        "y_rul": y_rul,
        "bounds": bounds,
    }


def _load_kaggle_real_data(csv_path: str) -> dict:
    """Load and map Kaggle pump-sensor-data.csv for the advanced pipeline."""
    import numpy as np

    adv_loader = KaggleAdvancedLoader()
    df = adv_loader.load(csv_path)

    # Build windows from the 7 mapped sensor columns
    windows = _windows_from_df(df, cols=SENSORS)
    log.info(f"  Kaggle windows: {len(windows)}, sensor cols: {SENSORS}")

    feats = extract_batch(windows) if len(windows) else np.empty((0, 84), dtype=np.float32)

    # Class labels: align to window-end index
    y_cls_full = df["fault_class"].to_numpy(dtype=np.int32)
    y_cls = y_cls_full[WINDOW_SIZE - 1:] if len(y_cls_full) >= WINDOW_SIZE else None

    # RUL labels: align to window-end index
    y_rul_full = df["rul_hours"].to_numpy(dtype=np.float32)
    y_rul = y_rul_full[WINDOW_SIZE - 1:] if len(y_rul_full) >= WINDOW_SIZE else None

    # No structured missingness masks for Kaggle data — use zero masks
    mask_windows = np.zeros_like(windows, dtype=np.float32) if len(windows) else None

    # Clip extreme outliers per-column
    bounds = {}
    for col in SENSORS:
        col_data = df[col].to_numpy(dtype=np.float32)
        bounds[col] = {
            "p01": float(np.percentile(col_data, 1)),
            "p99": float(np.percentile(col_data, 99)),
        }

    n_normal = int((y_cls == 0).sum()) if y_cls is not None else 0
    n_fault  = int((y_cls != 0).sum()) if y_cls is not None else 0
    log.info(
        f"  Kaggle advanced: {len(windows):,} windows — "
        f"normal={n_normal:,}, fault={n_fault:,} | "
        f"rul range=[{float(y_rul.min()):.1f}, {float(y_rul.max()):.1f}]h"
        if y_rul is not None else "  Kaggle advanced: no RUL labels"
    )

    return {
        "available": True,
        "windows": windows,
        "features": feats,
        "mask_windows": mask_windows,
        "y_cls": y_cls,
        "y_rul": y_rul,
        "bounds": bounds,
        "source": "kaggle_advanced",
    }


def train_isolation_forest(
    X_features: np.ndarray,
    y_class: np.ndarray,
    contamination: float,
) -> dict:
    """
    Spec §6: Train IF on NORMAL class only (class_id=0).
    Normalise scores to [0, 1] using robust training percentiles (p5/p95).
    """
    log.info("─" * 60)
    log.info("Training Isolation Forest (unsupervised, normal data only)...")

    X_normal = X_features[y_class == 0]
    log.info(f"  Normal samples: {len(X_normal)}")

    scaler = StandardScaler()
    X_n_s  = scaler.fit_transform(X_normal)

    t0  = time.perf_counter()
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X_n_s)
    elapsed = time.perf_counter() - t0
    log.info(f"  Trained in {elapsed:.2f}s")

    # Robust score scaling from normal-train score distribution.
    scores_train = iso.decision_function(X_n_s)
    score_p5 = float(np.percentile(scores_train, 5.0))
    score_p95 = float(np.percentile(scores_train, 95.0))
    if score_p95 <= score_p5:
        score_p95 = score_p5 + 1e-6

    # Score calibration on full training set
    X_all_s = scaler.transform(X_features)
    raw_scores = iso.decision_function(X_all_s)   # higher = more normal
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    norm_scores = np.clip((raw_scores - score_p5) / (score_p95 - score_p5 + 1e-12), 0, 1)

    bundle = {
        "model":      iso,
        "scaler":     scaler,
        "score_p5":   score_p5,
        "score_p95":  score_p95,
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
    normal_scores = (1.0 - norm_scores)[y_bin == 0]
    anomaly_scores = (1.0 - norm_scores)[y_bin == 1]
    overlap = float(np.mean(anomaly_scores <= np.percentile(normal_scores, 95.0))) if len(normal_scores) and len(anomaly_scores) else 0.0
    log.info(f"  ROC-AUC={auc:.4f}  Avg-Precision={ap:.4f}  overlap@normal95={overlap:.4f}")
    log.info(
        "  IF scaling (robust): "
        f"p5={score_p5:.6f}, p95={score_p95:.6f}, raw_min={score_min:.6f}, raw_max={score_max:.6f}"
    )

    return {"roc_auc": round(auc, 4), "avg_precision": round(ap, 4)}


def train_random_forest(X_features: np.ndarray, y_class: np.ndarray) -> dict:
    """Spec §7: Train 5-class RandomForest classifier."""
    log.info("─" * 60)
    log.info("Training 5-class Random Forest classifier...")

    clf = FaultClassifier()
    result = clf.fit(
        X_features,
        y_class,
        n_estimators=200,
        max_depth=12,
        calibration_method="isotonic",
    )
    out_path = MODELS_DIR / "fault_classifier.pkl"
    clf.save(out_path)
    log.info(f"  train_accuracy={result['train_accuracy']}")
    return result


def train_shared_latent(X_windows: np.ndarray, y_class: np.ndarray, y_rul: np.ndarray) -> tuple[SharedLatentRuntime, dict]:
    """Train and persist representation-consistent shared encoder."""
    log.info("─" * 60)
    log.info("Training shared latent multi-task encoder...")

    enc = SharedLatentRuntime(in_channels=X_windows.shape[2], hidden_dim=128, n_faults=5)
    metrics = enc.fit(
        X_windows,
        y_class,
        y_rul,
        epochs=20,
        batch_size=128,
        lr=1e-3,
        lambda_anom=1.0,
        lambda_cls=1.0,
        lambda_rul=1.0,
        lambda_consistency=0.2,
        k_consistency=1.0,
    )

    out_path = MODELS_DIR / "shared_latent.pt"
    enc.save(out_path)
    log.info(f"  Saved -> {out_path}")
    log.info(f"  shared_metrics={metrics}")
    return enc, metrics


def train_lstm(X_windows: np.ndarray, y_rul: np.ndarray) -> dict:
    """Perfect Architecture v3: BiLSTM+SensorGate+Attention, log-space targets."""
    log.info("─" * 60)
    log.info("Training LSTM v3 (Perfect Architecture — log-space + WarmRestarts)...")

    # Phase 6: augment steep/fall-off tail cases to improve conformal coverage in extremes.
    low_q = float(np.percentile(y_rul, 20.0))
    tail_idx = np.where(y_rul <= low_q)[0]
    if len(tail_idx) > 0:
        X_tail = X_windows[tail_idx].copy()
        # Late-window decay ramp emphasizes abrupt end-of-life trajectories.
        ramp = np.ones((WINDOW_SIZE, 1), dtype=np.float32)
        ramp[int(WINDOW_SIZE * 0.6):] = np.linspace(1.0, 0.70, WINDOW_SIZE - int(WINDOW_SIZE * 0.6)).reshape(-1, 1)
        X_tail[:, :, :7] = X_tail[:, :, :7] * ramp[np.newaxis, :, :]
        y_tail = np.clip(y_rul[tail_idx] * 0.85, 1.0, None)
        X_windows = np.concatenate([X_windows, X_tail], axis=0)
        y_rul = np.concatenate([y_rul, y_tail], axis=0)
        log.info(f"  Tail augmentation: +{len(X_tail)} windows (q20={low_q:.1f}h)")

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


def _domain_profile(bundle: dict, X_domain: np.ndarray) -> dict:
    """Compute p5/p95 anomaly normalization profile for one domain."""
    if len(X_domain) == 0:
        return {}
    scaler = bundle["scaler"]
    model = bundle["model"]
    X_s = scaler.transform(X_domain)
    raw = model.decision_function(X_s)
    p5 = float(np.percentile(raw, 5.0))
    p95 = float(np.percentile(raw, 95.0))
    if p95 <= p5:
        p95 = p5 + 1e-6
    return {
        "p5": p5,
        "p95": p95,
        "n": int(len(raw)),
    }




def main(real_csv: str | None = None) -> None:
    t_start = time.perf_counter()

    log.info("=" * 60)
    log.info("  Intelligent Predictive Maintenance — Training Pipeline")
    log.info("=" * 60)

    real = _load_real_data(real_csv)

    # Step 1: Generate synthetic data
    log.info("Step 1/5 — Generating synthetic training data...")
    X_wins, y_cls, y_rul, X_feats, t_idx = generate(n_per_class=300, shuffle=False, return_time_index=True)
    X_masks = np.zeros_like(X_wins, dtype=np.float32)
    X_wins_seq = np.concatenate([X_wins, X_masks], axis=2)

    if not np.all(np.diff(t_idx) >= 0):
        raise RuntimeError("Temporal ordering violation: synthetic timeline not monotonic")

    # Step 2: Shared latent representation
    log.info("Step 2/5 — Shared latent encoder...")
    enc, shared_metrics = train_shared_latent(X_wins_seq, y_cls, y_rul)
    X_hybrid, _, _, H_syn = build_hybrid_feature_batch(X_wins, enc, mask_windows=X_masks)
    assert X_hybrid.shape[1] == HYBRID_DIM, f"Unexpected training hybrid dim: {X_hybrid.shape[1]}"

    X_real_hybrid = None
    if real.get("available") and len(real.get("windows", [])) > 0 and len(real.get("features", [])) > 0:
        real_masks = real.get("mask_windows")
        if real_masks is None or len(real_masks) != len(real["windows"]):
            real_masks = np.zeros_like(real["windows"], dtype=np.float32)
        X_real_hybrid, _, _, _ = build_hybrid_feature_batch(real["windows"], enc, mask_windows=real_masks)

    # Domain-shift mitigation: IF scaler/fit on real normal when available
    if X_real_hybrid is not None and len(X_real_hybrid) > 0:
        # Train IF on real normal (or full real if labels unavailable)
        y_real = real.get("y_cls")
        if y_real is not None and len(y_real) == len(X_real_hybrid):
            normal_idx = (y_real == 0)
            X_if_train = X_real_hybrid[normal_idx] if normal_idx.any() else X_real_hybrid
            y_if_aux = y_real
        else:
            X_if_train = X_real_hybrid
            y_if_aux = np.zeros(len(X_real_hybrid), dtype=int)
    else:
        X_if_train = X_hybrid[y_cls == 0]
        y_if_aux = y_cls

    # Step 3: Isolation Forest
    log.info("Step 3/5 — Isolation Forest...")
    # Reconstruct IF training arrays to preserve function signature
    if X_real_hybrid is not None and len(X_real_hybrid) > 0:
        # Compose pseudo full set for score normalization
        X_if_full = X_real_hybrid
        y_if_full = y_if_aux
    else:
        X_if_full = X_hybrid
        y_if_full = y_cls

    contamination = float(_CONFIG.get("isolation_forest", {}).get("contamination", 0.05))
    log.info(f"  Using IF contamination from config: {contamination}")
    if_metrics = train_isolation_forest(X_if_full, y_if_full, contamination=contamination)

    # Phase 6: keep domain-specific normalization profiles for real vs synthetic.
    try:
        if_bundle_path = MODELS_DIR / "isolation_forest.pkl"
        with open(if_bundle_path, "rb") as fh:
            if_bundle = pickle.load(fh)

        synth_norm = X_hybrid[y_cls == 0] if np.any(y_cls == 0) else X_hybrid
        profiles = {"synthetic": _domain_profile(if_bundle, synth_norm)}

        if X_real_hybrid is not None and len(X_real_hybrid) > 0:
            y_real = real.get("y_cls")
            if y_real is not None and len(y_real) == len(X_real_hybrid):
                real_norm = X_real_hybrid[y_real == 0] if np.any(y_real == 0) else X_real_hybrid
            else:
                real_norm = X_real_hybrid
            profiles["real"] = _domain_profile(if_bundle, real_norm)

        if_bundle["domain_profiles"] = profiles
        with open(if_bundle_path, "wb") as fh:
            pickle.dump(if_bundle, fh)
        log.info(f"  IF domain profiles saved: {list(profiles.keys())}")
    except Exception as exc:
        log.warning(f"  IF domain profile export skipped: {exc}")

    # Step 4: 5-class RF
    log.info("Step 4/5 — 5-class Random Forest...")
    # Mixed RF training when labeled real data exists
    if real.get("available") and real.get("y_cls") is not None:
        y_real = real["y_cls"]
        X_real = X_real_hybrid if X_real_hybrid is not None else np.empty((0, X_hybrid.shape[1]), dtype=np.float32)
        if len(y_real) == len(X_real) and len(X_real) > 0:
            X_rf = np.vstack([X_hybrid, X_real])
            y_rf = np.concatenate([y_cls, y_real])
        else:
            X_rf, y_rf = X_hybrid, y_cls
    else:
        X_rf, y_rf = X_hybrid, y_cls

    rf_metrics = train_random_forest(X_rf, y_rf)

    # Step 5: LSTM
    log.info("Step 5/5 — LSTM RUL predictor...")
    try:
        if real.get("available") and real.get("y_rul") is not None and len(real["windows"]) == len(real["y_rul"]):
            real_masks = real.get("mask_windows")
            if real_masks is None or len(real_masks) != len(real["windows"]):
                real_masks = np.zeros_like(real["windows"], dtype=np.float32)
            X_real_seq = np.concatenate([real["windows"], real_masks], axis=2)
            X_lstm = np.concatenate([X_wins_seq, X_real_seq], axis=0)
            y_lstm = np.concatenate([y_rul, real["y_rul"]], axis=0)
        else:
            X_lstm, y_lstm = X_wins_seq, y_rul
        lstm_metrics = train_lstm(X_lstm, y_lstm)
    except Exception as exc:
        log.warning(f"  LSTM training skipped: {exc}")
        lstm_metrics = {"note": str(exc)}

    # Step 5: Threshold calibration + evaluation
    try:
        idx_tr, idx_val, idx_te = time_aware_split_indices(len(X_rf))
        validate_temporal_split(idx_tr, idx_val, idx_te)
        if_eval = evaluate_if(X_rf[idx_tr], y_rf[idx_tr], X_rf[idx_te], y_rf[idx_te])
        rf_hold = evaluate_rf_holdout(X_rf[idx_tr], y_rf[idx_tr], X_rf[idx_te], y_rf[idx_te])
        rf_cv = evaluate_rf_cv(X_rf, y_rf, k=5)

        # derive anomaly scores from IF bundle for threshold optimization
        with open(MODELS_DIR / "isolation_forest.pkl", "rb") as fh:
            if_bundle = pickle.load(fh)
        Xte = if_bundle["scaler"].transform(X_rf[idx_te])
        raw = if_bundle["model"].decision_function(Xte)
        p5 = float(if_bundle.get("score_p5", if_bundle["score_min"]))
        p95 = float(if_bundle.get("score_p95", if_bundle["score_max"]))
        norm = np.clip((raw - p5) / (p95 - p5 + 1e-12), 0, 1)
        anomaly_scores = 1.0 - norm
        Xval = if_bundle["scaler"].transform(X_rf[idx_val])
        raw_val = if_bundle["model"].decision_function(Xval)
        norm_val = np.clip((raw_val - p5) / (p95 - p5 + 1e-12), 0, 1)
        anomaly_scores_val = 1.0 - norm_val
        y_bin_val = (y_rf[idx_val] != 0).astype(int)
        normal_scores = anomaly_scores_val[y_bin_val == 0]
        if len(normal_scores) == 0:
            normal_scores = anomaly_scores_val
        cal = calibrate_threshold_unsupervised(normal_scores, quantile=95.0)
        save_threshold_config(cal, CONFIGS_DIR / "threshold.json")

        # Train learned fusion model on holdout using [if_score, rf_fault_prob, h].
        rf_loaded = FaultClassifier()
        rf_loaded.load(MODELS_DIR / "fault_classifier.pkl")
        rf_fault_probs = []
        for row in X_rf[idx_te]:
            pred = rf_loaded.predict(row)
            rf_fault_probs.append(1.0 - float(pred["probabilities"].get("normal", 0.0)))
        rf_fault_probs = np.asarray(rf_fault_probs, dtype=np.float32)

        latent_dim = H_syn.shape[1]
        latent_te = X_rf[idx_te][:, -latent_dim:]
        y_bin_te = (y_rf[idx_te] != 0).astype(int)
        rp_fusion = RULPredictor()
        rp_fusion.load(MODELS_DIR / "rul_lstm.pt")
        X_te_wins = X_wins[idx_te]
        X_te_masks = X_masks[idx_te]
        expected_in = int(rp_fusion._kwargs.get("input_size", X_te_wins.shape[2])) if hasattr(rp_fusion, "_kwargs") else X_te_wins.shape[2]
        if expected_in == X_te_wins.shape[2] * 2:
            X_te_seq = np.concatenate([X_te_wins, X_te_masks], axis=2)
        else:
            X_te_seq = X_te_wins
        rul_preds_te = np.array([rp_fusion.predict(w) for w in X_te_seq], dtype=np.float32)

        # Phase 6: hysteresis branch training signal from IF anomaly EMA.
        hyst_te = np.zeros_like(anomaly_scores, dtype=np.float32)
        ema = 0.0
        for i, s in enumerate(anomaly_scores):
            ema = 0.8 * ema + 0.2 * float(s)
            hyst_te[i] = ema

        fusion_metrics = train_fusion_model(
            if_scores=anomaly_scores,
            rf_fault_probs=rf_fault_probs,
            latent=latent_te,
            rul_pred=rul_preds_te,
            hysteresis_signal=hyst_te,
            y_binary=y_bin_te,
            out_path=MODELS_DIR / "fusion_meta.pkl",
        )

        # LSTM holdout metrics (if trained)
        lstm_eval = {}
        if real.get("available") and real.get("y_rul") is not None and len(real["windows"]) > 0:
            rp = RULPredictor()
            rp.load(MODELS_DIR / "rul_lstm.pt")
            expected_in = int(rp._kwargs.get("input_size", real["windows"].shape[2]))
            X_eval = real["windows"]
            if expected_in == real["windows"].shape[2] * 2:
                real_masks = real.get("mask_windows")
                if real_masks is None or len(real_masks) != len(real["windows"]):
                    real_masks = np.zeros_like(real["windows"], dtype=np.float32)
                X_eval = np.concatenate([real["windows"], real_masks], axis=2)
            preds = np.array([rp.predict(w) for w in X_eval], dtype=np.float32)
            lstm_eval = evaluate_lstm_holdout(real["y_rul"], preds)

        metrics = {
            "if": if_eval,
            "rf_holdout": rf_hold,
            "rf_cv": rf_cv,
            "lstm_holdout": lstm_eval,
            "calibration": {
                "if_threshold": cal["if_threshold"],
                "method": cal["method"],
            },
            "shared_latent": shared_metrics,
            "fusion_meta": fusion_metrics,
        }
        save_metrics_json(metrics, _ROOT / "evaluation" / "reports" / "pipeline_metrics.json")
        log.info(
            f"  Calibrated IF threshold={cal['if_threshold']:.4f} ({cal['method']})"
        )
    except Exception as exc:
        log.warning(f"  Evaluation/calibration skipped: {exc}")

    total = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(f"  Training complete in {total:.1f}s")
    log.info(f"  IF    → models/isolation_forest.pkl  (AUC={if_metrics['roc_auc']})")
    log.info(f"  RF    → models/fault_classifier.pkl  (acc={rf_metrics['train_accuracy']})")
    log.info(f"  LSTM  → models/rul_lstm.pt")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HydroX models")
    parser.add_argument(
        "--real-csv",
        type=str,
        default=None,
        help="Optional real dataset CSV path for domain-shift mitigation",
    )
    args = parser.parse_args()
    main(real_csv=args.real_csv)
