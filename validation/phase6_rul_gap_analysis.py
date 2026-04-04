from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.rul_lstm import RULPredictor
from training.generate_data import generate


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sensor_slopes(X: np.ndarray) -> np.ndarray:
    t = np.arange(X.shape[1], dtype=np.float32)
    t = t - t.mean()
    denom = float(np.sum(t * t) + 1e-8)
    # Per-window per-sensor least-squares slope over time.
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.tensordot(Xc, t, axes=([1], [0])) / denom


def main() -> None:
    root = ROOT
    out_dir = root / "evaluation" / "reports"
    _ensure_dir(out_dir)

    rp = RULPredictor()
    rp.load(root / "models" / "rul_lstm.pt")

    X_wins, _, y_rul, _ = generate(n_per_class=160, seed=42, shuffle=False)

    expected_in = int(rp._kwargs.get("input_size", X_wins.shape[2]))
    if expected_in == X_wins.shape[2] * 2:
        X_eval = np.concatenate([X_wins, np.zeros_like(X_wins)], axis=2)
    else:
        X_eval = X_wins

    means = []
    low = []
    high = []
    for w in X_eval:
        out = rp.predict_with_uncertainty(w, n_samples=20)
        means.append(float(out["mean"]))
        low.append(float(out["conformal_low"]))
        high.append(float(out["conformal_high"]))

    pred = np.asarray(means, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    y = y_rul.astype(np.float32)

    residual = np.abs(y - pred)
    inside = (y >= low) & (y <= high)
    outside = ~inside

    slopes = _sensor_slopes(X_wins)
    slope_mag = np.linalg.norm(slopes, axis=1)

    out_idx = np.where(outside)[0]
    tail_thr = float(np.percentile(residual, 95.0))
    tail_idx = np.where(residual >= tail_thr)[0]

    summary = {
        "n": int(len(y)),
        "coverage": float(np.mean(inside)),
        "outside_count": int(np.sum(outside)),
        "outside_fraction": float(np.mean(outside)),
        "residual_mean": float(np.mean(residual)),
        "residual_p90": float(np.percentile(residual, 90.0)),
        "residual_p95": float(np.percentile(residual, 95.0)),
        "outside_run_max": int(max((len(run) for run in "".join("1" if o else "0" for o in outside).split("0")), default=0)),
        "outside_slope_mag_mean": float(np.mean(slope_mag[outside])) if np.any(outside) else 0.0,
        "inside_slope_mag_mean": float(np.mean(slope_mag[inside])) if np.any(inside) else 0.0,
        "tail_threshold_residual": tail_thr,
        "tail_count": int(len(tail_idx)),
    }

    (out_dir / "rul_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Detailed outlier rows for audit.
    rows = []
    for i in out_idx.tolist():
        rows.append({
            "idx": int(i),
            "y_true": float(y[i]),
            "y_pred": float(pred[i]),
            "low": float(low[i]),
            "high": float(high[i]),
            "residual": float(residual[i]),
            "slope_mag": float(slope_mag[i]),
            "outside": True,
        })
    (out_dir / "rul_outlier_windows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Charts
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4))
        plt.hist(residual, bins=40, color="#2f6d9f", alpha=0.85)
        plt.axvline(np.percentile(residual, 95.0), color="#bb2f2f", linestyle="--", label="p95 residual")
        plt.title("RUL Residual Distribution")
        plt.xlabel("|y_true - y_pred| (h)")
        plt.ylabel("Count")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "rul_residual_hist.png", dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.scatter(y, residual, s=10, alpha=0.5, c="#3a7a3a")
        plt.axhline(np.percentile(residual, 95.0), color="#bb2f2f", linestyle="--")
        plt.title("Residual vs True RUL")
        plt.xlabel("True RUL (h)")
        plt.ylabel("Residual (h)")
        fig.tight_layout()
        fig.savefig(out_dir / "rul_residual_vs_true.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
