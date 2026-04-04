"""
evaluation/ablation_study.py — Empirical ablation study for the PdM pipeline.

Two ablation axes are evaluated:
  ① LSTM RUL Regressor — 4 progressive variants on fixed train/val split
  ② Fault Classifier — RF with/without FFT features

Run from project root:
    .venv312\Scripts\python pump_twin\evaluation\ablation_study.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── make internal imports work ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.generate_data import generate
from pipeline.features import (
    N_SENSORS, N_TIME_FEATURES, N_FFT_FEATURES,
    _time_domain_features, _fft_features,
)
from pipeline.rul_lstm import RULPredictor

# ── reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# fmt helpers
SEP  = "─" * 66
SEP2 = "═" * 66
def hr(c="─"): print(c * 66)


# ══════════════════════════════════════════════════════════════════════════
# Data generation (shared across all ablations)
# ══════════════════════════════════════════════════════════════════════════

print(SEP2)
print("  ABLATION STUDY — Predictive Maintenance Pipeline")
print(SEP2)
print("\n[1/2] Generating dataset (n_per_class=500, seed=42)...")

X_windows, y_class, y_rul, X_features = generate(n_per_class=500, seed=SEED)

# time-domain-only features (49 dims: 7 sensors × 7 stats)
print("[1/2] Building time-domain-only features (no FFT)...")
X_time_only = np.vstack([
    np.concatenate([_time_domain_features(X_windows[i, :, col]) for col in range(N_SENSORS)])
    for i in range(len(X_windows))
]).astype(np.float32)

# Train/val split (fixed seed)
X_win_tr,  X_win_val,  y_rul_tr,  y_rul_val  = train_test_split(X_windows,  y_rul,    test_size=0.2, random_state=SEED)
X_feat_tr, X_feat_val, y_cls_tr,  y_cls_val  = train_test_split(X_features, y_class,  test_size=0.2, random_state=SEED)
X_time_tr, X_time_val                         = train_test_split(X_time_only, test_size=0.2, random_state=SEED)

FAULT_NAMES = ["normal", "bearing_fault", "cavitation", "dry_run", "misalignment"]

print(f"  Train: {len(X_win_tr):,}   Val: {len(X_win_val):,}\n")


# ══════════════════════════════════════════════════════════════════════════
# ABLATION ①: LSTM / BiLSTM / +Huber / +Attention
# ══════════════════════════════════════════════════════════════════════════

hr("═")
print("  ABLATION ①: LSTM Architecture")
hr("═")
print(f"  {'Variant':<36} {'Val MAE(h)':>10} {'Val RMSE(h)':>12}  {'Time(s)':>8}")
hr()

# Production training config — identical to main_train.py
EPOCHS_FULL    = 120
PATIENCE_FULL  = 20
BATCH_FULL     = 128
LR_HUBER       = 3e-4
LR_MSE         = 1e-3   # MSE landscape is steeper, higher LR works
LABEL_NOISE    = 0.04
HUBER_DELTA    = 50.0

LSTM_VARIANTS = [
    {
        "name":          "① Baseline  (Uni-LSTM, MSE, no-Attn)",
        "bidirectional": False,
        "hidden_size":   64,
        "num_layers":    2,
        "dropout":       0.2,
        "loss":          "mse",
        "attention":     False,
        "epochs":        EPOCHS_FULL,
        "lr":            LR_MSE,
        "patience":      PATIENCE_FULL,
    },
    {
        "name":          "② +BiLSTM   (Bi-LSTM,  MSE, no-Attn)",
        "bidirectional": True,
        "hidden_size":   128,
        "num_layers":    2,
        "dropout":       0.3,
        "loss":          "mse",
        "attention":     False,
        "epochs":        EPOCHS_FULL,
        "lr":            LR_MSE,
        "patience":      PATIENCE_FULL,
    },
    {
        "name":          "③ +HuberLoss(Bi-LSTM, Huber, no-Attn)",
        "bidirectional": True,
        "hidden_size":   128,
        "num_layers":    2,
        "dropout":       0.3,
        "loss":          "huber",
        "attention":     False,
        "epochs":        EPOCHS_FULL,
        "lr":            LR_HUBER,
        "patience":      PATIENCE_FULL,
    },
    {
        "name":          "④ Full Model(Bi-LSTM, Huber, +Attn) ✓",
        "bidirectional": True,
        "hidden_size":   128,
        "num_layers":    2,
        "dropout":       0.3,
        "loss":          "huber",
        "attention":     True,
        "epochs":        EPOCHS_FULL,
        "lr":            LR_HUBER,
        "patience":      PATIENCE_FULL,
    },
]

lstm_results: list[dict] = []

for cfg in LSTM_VARIANTS:
    t0 = time.time()

    predictor = RULPredictor(
        input_size    = N_SENSORS,
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        dropout       = cfg["dropout"],
        bidirectional = cfg["bidirectional"],
        use_attention = cfg["attention"],     # ← ablation flag
    )

    predictor.fit(
        X_win_tr, y_rul_tr,
        epochs     = cfg["epochs"],
        lr         = cfg["lr"],
        patience   = cfg["patience"],
        batch_size = BATCH_FULL,
        huber_delta= HUBER_DELTA if cfg["loss"] == "huber" else None,
        label_noise= LABEL_NOISE,
    )

    # Validate
    preds = np.array([predictor.predict(X_win_val[i]) for i in range(len(X_win_val))])
    mae   = float(np.mean(np.abs(preds - y_rul_val)))
    rmse  = float(np.sqrt(np.mean((preds - y_rul_val) ** 2)))
    elapsed = time.time() - t0

    lstm_results.append({"name": cfg["name"], "mae": mae, "rmse": rmse, "time": elapsed})
    print(f"  {cfg['name']:<36} {mae:>10.2f} {rmse:>12.2f}  {elapsed:>8.1f}s")

hr()

# Compute gains relative to baseline
base_mae  = lstm_results[0]["mae"]
base_rmse = lstm_results[0]["rmse"]
print("\n  Improvement vs Baseline (↓ lower is better)")
hr()
print(f"  {'Variant':<36} {'ΔMAE':>10} {'ΔRMSE':>10}")
hr()
for r in lstm_results[1:]:
    d_mae  = r["mae"]  - base_mae
    d_rmse = r["rmse"] - base_rmse
    sign_m = "↓" if d_mae  < 0 else "↑"
    sign_r = "↓" if d_rmse < 0 else "↑"
    print(f"  {r['name']:<36} {sign_m}{abs(d_mae):>9.2f}h {sign_r}{abs(d_rmse):>9.2f}h")
hr()


# ══════════════════════════════════════════════════════════════════════════
# ABLATION ②: Feature Engineering (Time vs Time+FFT) for RF classifier
# ══════════════════════════════════════════════════════════════════════════

hr("═")
print("\n  ABLATION ②: Feature Engineering (Fault Classifier)")
hr("═")

def _train_rf_and_eval(X_tr, X_val, y_tr, y_val, label=""):
    t0  = time.time()
    rf  = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    preds   = rf.predict(X_val)
    f1_mac  = float(f1_score(y_val, preds, average="macro"))
    f1_wt   = float(f1_score(y_val, preds, average="weighted"))
    elapsed = time.time() - t0
    cm      = confusion_matrix(y_val, preds)

    print(f"\n  Variant: {label}")
    hr()
    print(f"  Macro   F1 : {f1_mac:.4f}   ({f1_mac*100:.1f}%)")
    print(f"  Weighted F1: {f1_wt:.4f}   ({f1_wt*100:.1f}%)")
    print(f"  Train time : {elapsed:.1f}s   |  Features: {X_tr.shape[1]}")
    print(f"\n  Per-class F1:")
    per_class = f1_score(y_val, preds, average=None, labels=[0,1,2,3,4])
    for i, (name, score) in enumerate(zip(FAULT_NAMES, per_class)):
        bar = "█" * int(score * 20)
        print(f"    {name:<16} {score:.3f}  |{bar:<20}|")

    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = "  " + "".join(f"  {n[:5]:>5}" for n in FAULT_NAMES)
    print(header)
    for i, row in enumerate(cm):
        cells = "".join(f"  {v:>5}" for v in row)
        marker = " ←" if row[i] == row.max() else ""
        print(f"  {FAULT_NAMES[i][:5]:>5}{cells}{marker}")
    hr()

    return {"f1_macro": f1_mac, "f1_weighted": f1_wt, "features": X_tr.shape[1]}


res_time = _train_rf_and_eval(
    X_time_tr, X_time_val, y_cls_tr, y_cls_val,
    label="Time-Domain ONLY  (7 stats × 7 sensors = 49 features)"
)

res_full = _train_rf_and_eval(
    X_feat_tr, X_feat_val, y_cls_tr, y_cls_val,
    label="Time + FFT Domain  (12 features × 7 sensors = 84 features) ✓"
)

# Summary table for ②
hr("═")
print("\n  Summary — Feature Engineering Ablation")
hr()
print(f"  {'Variant':<36} {'Features':>8} {'Macro F1':>10} {'Wtd F1':>10} {'Gain':>10}")
hr()
gain_mac = res_full["f1_macro"] - res_time["f1_macro"]
gain_wt  = res_full["f1_weighted"] - res_time["f1_weighted"]
print(f"  {'Time-Domain Only':<36} {res_time['features']:>8} {res_time['f1_macro']:>10.4f} {res_time['f1_weighted']:>10.4f}")
print(f"  {'Time + FFT (Full) ✓':<36} {res_full['features']:>8} {res_full['f1_macro']:>10.4f} {res_full['f1_weighted']:>10.4f} {'↑'+f'{gain_mac*100:.1f}%':>10}")
hr()


# ══════════════════════════════════════════════════════════════════════════
# ABLATION ③: Anomaly Detection — RF contamination baseline vs Isolation Forest
# ══════════════════════════════════════════════════════════════════════════

hr("═")
print("\n  ABLATION ③: Anomaly Detector (Isolation Forest vs Score Threshold)")
hr("═")

# Normal data = class 0
X_normal      = X_feat_tr[y_cls_tr == 0]
X_anom_val    = X_feat_val
true_anom     = (y_cls_val != 0).astype(int)   # 1 = anomalous

# Isolation Forest
t0 = time.time()
iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1)
iso.fit(X_normal)
iso_preds = (iso.predict(X_anom_val) == -1).astype(int)  # -1 = anomaly → 1
iso_f1    = float(f1_score(true_anom, iso_preds, average="binary"))
iso_time  = time.time() - t0

# Naive threshold: flag if any feature > 3σ from training mean (no learned model)
feat_mean = X_normal.mean(axis=0)
feat_std  = X_normal.std(axis=0) + 1e-8
z_scores  = np.abs((X_anom_val - feat_mean) / feat_std)
naive_pred = (z_scores.max(axis=1) > 3.0).astype(int)
naive_f1  = float(f1_score(true_anom, naive_pred, average="binary"))

print(f"  {'Detector':<32} {'F1 (Anomaly)':>14}")
hr()
print(f"  {'Naïve 3σ Threshold (no model)':<32} {naive_f1:>14.4f}")
print(f"  {'Isolation Forest (trained) ✓':<32} {iso_f1:>14.4f}   ({iso_time:.1f}s)")
gain_iso = iso_f1 - naive_f1
print(f"\n  IF improvement over naïve baseline: ↑{gain_iso*100:.1f}%")
hr()


# ══════════════════════════════════════════════════════════════════════════
# Final summary table
# ══════════════════════════════════════════════════════════════════════════

hr("═")
print("\n  ABLATION STUDY — COMPLETE SUMMARY")
hr("═")
print()
print("  ① LSTM RUL Architecture")
hr()
print(f"  {'Variant':<36} {'MAE(h)':>8} {'RMSE(h)':>9} {'RelMAE':>8}")
hr()
for r in lstm_results:
    rel = (r["mae"] / base_mae - 1) * 100
    sign = ("↑ baseline" if r is lstm_results[0] else
            f"↑{abs(rel):.1f}% worse" if rel > 0 else f"↓{abs(rel):.1f}% better ✓")
    print(f"  {r['name']:<36} {r['mae']:>8.2f} {r['rmse']:>9.2f}   {sign}")
hr()

print()
print("  ② Fault Classifier Feature Engineering")
hr()
print(f"  {'Time-Domain Only (49 feats)':<36} Wtd-F1: {res_time['f1_weighted']:.4f}")
print(f"  {'Time + FFT (84 feats) ✓':<36} Wtd-F1: {res_full['f1_weighted']:.4f}  (↑{gain_wt*100:.1f}%)")
hr()

print()
print("  ③ Anomaly Detection")
hr()
print(f"  {'Naïve 3σ Threshold':<36} F1: {naive_f1:.4f}")
print(f"  {'Isolation Forest ✓':<36} F1: {iso_f1:.4f}  (↑{gain_iso*100:.1f}%)")
hr("═")
print()
print("  Done. Use these numbers as proof of each component's contribution.")
hr("═")
