"""
validation/validate_rul_coverage.py — Phase 2: RUL conformal interval coverage test.

Computes empirical coverage of the 90% conformal interval on a held-out test set.

Pass condition: empirical coverage >= 88%

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
    print("\n  Generating held-out test set (seed=42) ...")
    X_wins, _, y_rul, _ = generate(n_per_class=150, seed=42, shuffle=False)

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

    print(f"\n  MAE             = {mae:.2f} h")
    print(f"  RMSE            = {rmse:.2f} h")
    print(f"  Avg interval    = ±{avg_width/2:.1f} h  (width={avg_width:.1f} h)")
    print(f"  Empirical coverage (90% target) = {coverage:.1%}")

    target      = 1.0 - rp._conformal_alpha  # e.g. 0.90
    lower_bound = target - 0.02              # allow ≥ 88%
    pass_cov    = coverage >= lower_bound

    print(f"\n  Target coverage : {target:.0%}  (min acceptable {lower_bound:.0%})")
    print(f"  Coverage test   : {'PASS OK' if pass_cov else 'FAIL X'}  ({coverage:.1%})")

    # Distribution of residuals vs interval width
    residuals = np.abs(y_true - preds_mean)
    print(f"\n  Residual stats  : mean={residuals.mean():.1f}h  p90={np.percentile(residuals,90):.1f}h  "
          f"max={residuals.max():.1f}h")
    print(f"  conformal_q90   : {rp._conformal_q90:.1f} h")

    print("\n" + "=" * 62)
    print(f"  OVERALL: {'PASS OK' if pass_cov else 'FAIL X — rerun main_train.py after rul_lstm.py fix'}")
    print("=" * 62)
    sys.exit(0 if pass_cov else 1)


if __name__ == "__main__":
    main()
