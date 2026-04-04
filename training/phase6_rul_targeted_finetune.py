from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.rul_lstm import RULPredictor
from training.generate_data import generate


def main() -> None:
    root = ROOT
    model_path = root / "models" / "rul_lstm.pt"

    rp = RULPredictor()
    rp.load(model_path)

    # Training pool separate from held-out validator seed.
    X_wins, _, y_rul, _ = generate(n_per_class=220, seed=123, shuffle=False)

    expected_in = int(rp._kwargs.get("input_size", X_wins.shape[2]))
    if expected_in == X_wins.shape[2] * 2:
        X_seq = np.concatenate([X_wins, np.zeros_like(X_wins)], axis=2)
    else:
        X_seq = X_wins

    # Strict temporal split: train first, calibration after.
    n = len(X_seq)
    n_cal = max(160, int(n * 0.20))
    X_train, X_cal = X_seq[:-n_cal], X_seq[-n_cal:]
    y_train, y_cal = y_rul[:-n_cal], y_rul[-n_cal:]

    ft = rp.fine_tune_tail(
        X_train,
        y_train,
        epochs=18,
        lr=7e-5,
        batch_size=128,
        tail_quantile=0.30,
        tail_weight=3.5,
        perturb_std=0.02,
        oversample_factor=3,
    )

    cal = rp.recalibrate_conformal(
        X_cal,
        y_cal,
        alpha=0.10,
        tail_quantile=0.30,
        tail_alpha=0.04,
    )

    rp.save(model_path)

    out = {
        "fine_tune": ft,
        "recalibration": cal,
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "model_path": str(model_path),
    }
    (root / "evaluation" / "reports" / "rul_targeted_finetune.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
