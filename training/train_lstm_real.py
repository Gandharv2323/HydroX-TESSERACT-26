"""
training/train_lstm_real.py
============================
Train the BiLSTM RUL predictor on real pump-sensor-data.csv.

Key design decisions
--------------------
* RUL labels  : time-to-next-failure computed from BROKEN event timestamps.
                Each row's RUL = minutes until next BROKEN event / 60 (hours).
                Rows after the last BROKEN event get RUL = rul_cap (720 h).
* Windowing   : sliding windows of WINDOW_SIZE=50 rows, stride=5 (for speed).
                Each window's label = RUL of its LAST row (most informative).
* Split       : stratified temporal split (reuses the same logic as train_real_csv).
                Windows are assigned to a split based on their last-row index.
* Architecture: same BiLSTM v3 (RULPredictor) — but input_size=51 sensors.
* Output      : models/real/lstm_rul_real.pt

Run:
    uv run python training/train_lstm_real.py
    uv run python training/train_lstm_real.py --csv pump-sensor-data.csv --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.rul_lstm import RULPredictor

# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50       # steps per window (same as synthetic pipeline)
STRIDE      = 5        # stride for sliding windows (reduces dataset size ~5×)
RUL_CAP     = 720.0   # hours — capped RUL for normal/post-failure rows
LABEL_COL   = "machine_status"
DROP_COLS   = {"Unnamed: 0", "timestamp", "machine_status"}
LABEL_MAP   = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}


# ──────────────────────────────────────────────────────────────────────────────
# RUL computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_rul_hours(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    rul_cap: float     = RUL_CAP,
) -> np.ndarray:
    """
    For each row compute RUL = minutes until the NEXT BROKEN event / 60.
    Rows with no future BROKEN event → rul_cap.

    Returns float32 array of shape (len(df),).
    """
    ts        = pd.to_datetime(df[timestamp_col]).values  # numpy datetime64
    is_broken = (df[LABEL_COL].str.upper() == "BROKEN").values

    broken_ts = ts[is_broken]                             # timestamps of failure events
    rul        = np.full(len(df), rul_cap, dtype=np.float64)

    for fail_ts in broken_ts:
        # minutes until this failure from every row (ignore rows AFTER it)
        delta_min = (fail_ts - ts) / np.timedelta64(1, "m")  # float minutes
        mask      = delta_min >= 0                             # only rows before/at failure
        rul[mask] = np.minimum(rul[mask], delta_min[mask] / 60.0)

    return rul.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y_rul, feature_names).
      X       : (n_rows, n_sensors)  float32
      y_rul   : (n_rows,)             float32 hours
    """
    print(f"[lstm_real] Loading {csv_path} …")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    sensor_cols = [c for c in df.columns if c not in DROP_COLS]
    # Drop all-NaN columns
    all_nan = [c for c in sensor_cols if df[c].isna().all()]
    if all_nan:
        print(f"[lstm_real] Dropping all-NaN columns: {all_nan}")
        sensor_cols = [c for c in sensor_cols if c not in all_nan]

    X_df = df[sensor_cols].ffill().fillna(0.0).astype(np.float32)
    X    = X_df.values

    print(f"[lstm_real] Computing RUL labels …")
    y_rul = compute_rul_hours(df)

    print(f"[lstm_real] Rows={len(X):,}  Sensors={X.shape[1]}")
    print(f"[lstm_real] RUL  min={y_rul.min():.1f}h  max={y_rul.max():.1f}h  "
          f"median={float(np.median(y_rul)):.1f}h")

    broken_mask = df[LABEL_COL].str.upper() == "BROKEN"
    print(f"[lstm_real] BROKEN events: {broken_mask.sum()}  "
          f"(rows with RUL<24h: {int((y_rul < 24).sum()):,})")

    return X, y_rul, sensor_cols


# ──────────────────────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────────────────────

def make_windows(
    X:           np.ndarray,   # (n_rows, n_sensors)
    y_rul:       np.ndarray,   # (n_rows,)
    window_size: int = WINDOW_SIZE,
    stride:      int = STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window extraction.
    Returns:
      X_wins  : (n_windows, window_size, n_sensors)
      y_wins  : (n_windows,)  RUL at last step of window
      end_idx : (n_windows,)  row-index of window's last step (for split assignment)
    """
    windows, labels, end_indices = [], [], []
    n = len(X)
    for i in range(0, n - window_size + 1, stride):
        windows.append(X[i:i + window_size])
        labels.append(y_rul[i + window_size - 1])  # RUL of last timestep
        end_indices.append(i + window_size - 1)

    X_wins   = np.stack(windows,   axis=0).astype(np.float32)
    y_wins   = np.array(labels,    dtype=np.float32)
    end_idx  = np.array(end_indices, dtype=np.int64)

    print(f"[lstm_real] Windows={len(X_wins):,}  "
          f"shape={X_wins.shape}  "
          f"RUL_min={y_wins.min():.1f}h  RUL_max={y_wins.max():.1f}h")
    return X_wins, y_wins, end_idx


# ──────────────────────────────────────────────────────────────────────────────
# Stratified temporal split (same as train_real_csv.py)
# ──────────────────────────────────────────────────────────────────────────────

def _stratified_temporal_split_windows(
    y_row:       np.ndarray,   # (n_rows,) class labels for original rows
    end_idx:     np.ndarray,   # (n_windows,) last-row index of each window
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each WINDOW to train/val/test based on the class of its last row,
    split 70/15/15 within each class by row position.
    Returns (train_mask, val_mask, test_mask) boolean arrays of shape (n_windows,).
    """
    unique_classes = np.unique(y_row)
    train_win, val_win, test_win = [], [], []

    for cls in unique_classes:
        cls_row_idx  = np.where(y_row == cls)[0]              # row indices of this class
        n_cls        = len(cls_row_idx)
        t_end        = int(n_cls * train_ratio)
        v_end        = int(n_cls * (train_ratio + val_ratio))
        train_row_set = set(cls_row_idx[:t_end].tolist())
        val_row_set   = set(cls_row_idx[t_end:v_end].tolist())
        test_row_set  = set(cls_row_idx[v_end:].tolist())

        for i, ei in enumerate(end_idx):
            if ei in train_row_set:
                train_win.append(i)
            elif ei in val_row_set:
                val_win.append(i)
            elif ei in test_row_set:
                test_win.append(i)

    return (
        np.array(sorted(train_win), dtype=np.int64),
        np.array(sorted(val_win),   dtype=np.int64),
        np.array(sorted(test_win),  dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_on_real_csv(
    csv_path: Path,
    out_dir:  Path,
    epochs:   int   = 80,
    batch:    int   = 256,
    lr:       float = 3e-4,
    patience: int   = 15,
    stride:   int   = STRIDE,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + label
    X_rows, y_rul_rows, sensor_cols = load_csv(csv_path)
    n_sensors = X_rows.shape[1]

    # Row-level class labels (for stratified split)
    df_raw   = pd.read_csv(csv_path)
    y_row_cls = df_raw[LABEL_COL].str.upper().map(LABEL_MAP).values.astype(np.int32)

    # 2. Window generation
    print(f"\n[lstm_real] Building sliding windows (size={WINDOW_SIZE}, stride={stride}) …")
    X_wins, y_wins, end_idx = make_windows(X_rows, y_rul_rows, WINDOW_SIZE, stride)

    # 3. Stratified temporal split
    print(f"[lstm_real] Stratified temporal split …")
    tr_idx, va_idx, te_idx = _stratified_temporal_split_windows(y_row_cls, end_idx)

    X_tr, y_tr = X_wins[tr_idx], y_wins[tr_idx]
    X_va, y_va = X_wins[va_idx], y_wins[va_idx]
    X_te, y_te = X_wins[te_idx], y_wins[te_idx]

    def _rul_summary(y: np.ndarray) -> str:
        return (f"n={len(y):,}  "
                f"RUL_min={y.min():.0f}h  "
                f"RUL_med={float(np.median(y)):.0f}h  "
                f"RUL_max={y.max():.0f}h  "
                f"critical(<24h)={(y<24).sum():,}")

    print(f"  Train : {_rul_summary(y_tr)}")
    print(f"  Val   : {_rul_summary(y_va)}")
    print(f"  Test  : {_rul_summary(y_te)}")

    # Save test split
    np.save(out_dir / "X_test_lstm.npy", X_te.astype(np.float32))
    np.save(out_dir / "y_test_lstm.npy", y_te.astype(np.float32))

    # 4. Build model — input_size = n_sensors (51 for this CSV)
    print(f"\n[lstm_real] Building BiLSTM RUL predictor  (input_size={n_sensors}) …")
    model = RULPredictor(
        input_size=n_sensors,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
        log_targets=True,
    )

    # 5. Train
    print(f"\n[lstm_real] Training on {len(X_tr):,} windows …")
    fit_metrics = model.fit(
        X=X_tr,
        y=y_tr,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch,
        huber_delta=0.5,
        label_noise=0.02,
        val_split=0.10,     # internal val split from train set for early stopping
        t0_restart=20,
    )

    # 6. Recalibrate conformal on val set
    print(f"\n[lstm_real] Recalibrating conformal predictor on val set ({len(X_va):,} windows) …")
    cal_metrics = model.recalibrate_conformal(X_va, y_va)
    print(f"  Conformal q90 = {cal_metrics.get('q90', 0):.1f} h")

    # 7. Evaluate on test
    print(f"\n[lstm_real] Evaluating on test set ({len(X_te):,} windows) …")
    import torch
    model._model.eval()
    X_te_n = model._normalise(X_te.astype(np.float32))
    with torch.no_grad():
        dev = model._device
        preds_log = model._model(
            torch.tensor(X_te_n).to(dev)
        ).cpu().numpy()

    preds_real = np.expm1(preds_log) if model._log_targets else preds_log
    preds_real = np.clip(
        model._cal_slope * preds_real + model._cal_intercept, 0.0, None
    )
    mae_test  = float(np.mean(np.abs(preds_real - y_te)))
    rmse_test = float(np.sqrt(np.mean((preds_real - y_te) ** 2)))
    # Coverage at 90% conformal interval
    q = model._conformal_q90
    coverage = float(np.mean(np.abs(preds_real - y_te) <= q))

    print(f"  Test MAE  = {mae_test:.1f} h")
    print(f"  Test RMSE = {rmse_test:.1f} h")
    print(f"  Test Coverage (q90={q:.1f}h) = {coverage:.1%}")

    # 8. Save model
    model_path = out_dir / "lstm_rul_real.pt"
    model.save(model_path)

    # 9. Save metadata
    meta = {
        "csv":          str(csv_path),
        "n_sensors":    n_sensors,
        "sensor_cols":  sensor_cols,
        "window_size":  WINDOW_SIZE,
        "stride":       stride,
        "rul_cap_h":    RUL_CAP,
        "split": {
            "train_windows": int(len(X_tr)),
            "val_windows":   int(len(X_va)),
            "test_windows":  int(len(X_te)),
            "stratified":    True,
        },
        "training":     {k: v for k, v in fit_metrics.items()
                         if k not in ("train_loss", "val_loss")},
        "conformal":    cal_metrics,
        "test_eval": {
            "mae_h":    round(mae_test,  1),
            "rmse_h":   round(rmse_test, 1),
            "coverage": round(coverage,  4),
        },
    }
    with open(out_dir / "lstm_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{'='*55}")
    print(f"  LSTM training complete")
    print(f"  Train MAE (val split) : {fit_metrics.get('val_mae_h', '?')} h")
    print(f"  Test  MAE             : {mae_test:.1f} h")
    print(f"  Test  RMSE            : {rmse_test:.1f} h")
    print(f"  Conformal q90         : {q:.1f} h")
    print(f"  Model saved           : {model_path}")
    print(f"{'='*55}")

    return meta


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="pump-sensor-data.csv")
    parser.add_argument("--out",     default="models/real")
    parser.add_argument("--epochs",  type=int,   default=80)
    parser.add_argument("--batch",   type=int,   default=256)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--patience",type=int,   default=15)
    parser.add_argument("--stride",  type=int,   default=5)
    args = parser.parse_args()

    csv_path = _ROOT / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    train_lstm_on_real_csv(
        csv_path = csv_path,
        out_dir  = _ROOT / args.out,
        epochs   = args.epochs,
        batch    = args.batch,
        lr       = args.lr,
        patience = args.patience,
        stride   = args.stride,
    )
