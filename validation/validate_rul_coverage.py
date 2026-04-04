"""
validation/validate_rul_coverage.py — Phase 2: RUL conformal interval coverage test.

Computes empirical coverage of the 90% conformal interval on a held-out test set.

Pass condition:
    - empirical coverage in [88%, 92%] for a 90% target
    - average interval width <= 220h (±110h)

Run: python validation/validate_rul_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.rul_lstm import RULPredictor
from training.generate_data import generate
from pipeline.buffer import WINDOW_SIZE

MODELS_DIR = _ROOT / "models"


def main() -> None:
    print("=" * 62)
    print("  Phase 2 — RUL Conformal Interval Coverage Validation")
    print("=" * 62)

    lstm_path = MODELS_DIR / "rul_lstm.pt"
    if not lstm_path.exists():
        print(f"  ERROR: {lstm_path} not found. Run: python main_train.py")
        sys.exit(1)

    print("\n  Loading LSTM model ...")
    rp = RULPredictor()
    rp.load(lstm_path)

    print(f"  conformal_q90   = {rp._conformal_q90:.2f} h")
    print(f"  conformal_n     = {rp._conformal_n}")
    print(f"  conformal_alpha = {rp._conformal_alpha}")

    # Generate held-out test data (different seed from training)
    print("\n  Generating held-out test set (seed=42, n_per_class=160) ...")
    X_wins, _, y_rul, _ = generate(n_per_class=160, seed=42, shuffle=False)

    # Determine LSTM input size
    expected_in = int(rp._kwargs.get("input_size", X_wins.shape[2]))
    if expected_in == X_wins.shape[2] * 2:
        X_eval = np.concatenate(
            [X_wins, np.zeros_like(X_wins, dtype=np.float32)], axis=2
        )
    else:
        X_eval = X_wins

    # Run predictions
    print(f"  Running {len(X_eval)} predictions ...")
    preds_mean   = []
    conf_lows    = []
    conf_highs   = []

    for i, w in enumerate(X_eval):
        out = rp.predict_with_uncertainty(w, n_samples=20)
        preds_mean.append(out["mean"])
        conf_lows.append(out["conformal_low"])
        conf_highs.append(out["conformal_high"])

    preds_mean  = np.array(preds_mean,  dtype=np.float32)
    conf_lows   = np.array(conf_lows,   dtype=np.float32)
    conf_highs  = np.array(conf_highs,  dtype=np.float32)
    y_true      = y_rul.astype(np.float32)

    # Empirical coverage: fraction of true RUL values inside [low, high]
    inside       = (y_true >= conf_lows) & (y_true <= conf_highs)
    coverage     = float(inside.mean())

    mae  = float(np.mean(np.abs(preds_mean - y_true)))
    rmse = float(np.sqrt(np.mean((preds_mean - y_true) ** 2)))
    avg_width = float(np.mean(conf_highs - conf_lows))

    print(f"\n  N windows        = {len(y_true)}")
    print(f"  MAE             = {mae:.2f} h")
    print(f"  RMSE            = {rmse:.2f} h")
    print(f"  Avg interval    = ±{avg_width/2:.1f} h  (width={avg_width:.1f} h)")
    print(f"  Empirical coverage (90% target) = {coverage:.1%}")

    target      = 1.0 - rp._conformal_alpha  # e.g. 0.90
    lower_bound = target - 0.02              # 88%
    upper_bound = target + 0.02              # 92%
    pass_cov    = lower_bound <= coverage <= upper_bound
    pass_width  = avg_width <= 220.0

    print(f"\n  Target coverage : {target:.0%}  (acceptable band {lower_bound:.0%}-{upper_bound:.0%})")
    print(f"  Coverage test   : {'PASS OK' if pass_cov else 'FAIL X'}  ({coverage:.1%})")
    print(f"  Width test      : {'PASS OK' if pass_width else 'FAIL X'}  (width={avg_width:.1f}h, target<=220h)")

    # Distribution of residuals vs interval width
    residuals = np.abs(y_true - preds_mean)
    print(f"\n  Residual stats  : mean={residuals.mean():.1f}h  p90={np.percentile(residuals,90):.1f}h  "
          f"max={residuals.max():.1f}h")
    print(f"  conformal_q90   : {rp._conformal_q90:.1f} h")

    print("\n" + "=" * 62)
    overall = pass_cov and pass_width and len(y_true) >= 750
    print(f"  Sample size test: {'PASS OK' if len(y_true) >= 750 else 'FAIL X'}  (n={len(y_true)})")
    print(f"  OVERALL: {'PASS OK' if overall else 'FAIL X — revisit RUL calibration/training'}")
    print("=" * 62)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
