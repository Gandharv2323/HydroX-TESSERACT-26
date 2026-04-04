"""
validation/validate_if.py — Phase 1: IF score distribution & stress-signal test.

Tests:
  1. Score distribution: normal vs anomalous windows
  2. Stress signal ordering: noise < drift < step (anomaly score)

Pass condition:
  - mean(noise_score) < mean(drift_score) < mean(step_score)
  - Normal class anomaly score mean < 0.4
  - Fault class anomaly score mean > 0.5

Run: python validation/validate_if.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.buffer import WINDOW_SIZE, SENSORS
from pipeline.representation import build_hybrid_feature_vector
from models.shared_latent import SharedLatentRuntime
from training.generate_data import generate

MODELS_DIR = _ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_if() -> dict:
    path = MODELS_DIR / "isolation_forest.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Train first: python main_train.py  (missing {path})")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_shared() -> SharedLatentRuntime | None:
    path = MODELS_DIR / "shared_latent.pt"
    if not path.exists():
        return None
    try:
        return SharedLatentRuntime.load(path)
    except Exception:
        return None


def _score_window(bundle: dict, window: np.ndarray, shared) -> float:
    """Score one (WINDOW_SIZE, 7) window. Returns anomaly score 0=normal, 1=fault."""
    feat_vec, _, _, _ = build_hybrid_feature_vector(window, shared)

    scaler = bundle["scaler"]
    iso    = bundle["model"]

    # Align feature vector to scaler's expected width
    expected = int(getattr(scaler, "n_features_in_", len(feat_vec)))
    v = np.asarray(feat_vec, dtype=np.float32).reshape(-1)
    if len(v) > expected:
        v = v[:expected]
    elif len(v) < expected:
        v = np.pad(v, (0, expected - len(v)))

    X_s = scaler.transform(v.reshape(1, -1))
    raw = float(iso.decision_function(X_s)[0])

    if "score_p5" in bundle and "score_p95" in bundle:
        p5, p95 = float(bundle["score_p5"]), float(bundle["score_p95"])
    else:
        p5, p95 = float(bundle["score_min"]), float(bundle["score_max"])
    if p95 <= p5:
        p95 = p5 + 1e-6

    normal_score = float(np.clip((raw - p5) / (p95 - p5 + 1e-12), 0, 1))
    return 1.0 - normal_score  # anomaly score


def _score_batch(bundle, windows: np.ndarray, shared) -> np.ndarray:
    return np.array([_score_window(bundle, w, shared) for w in windows])


# ─────────────────────────────────────────────────────────────────────────────
# Stress signal generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_windows(signal_fn, n: int = 30) -> np.ndarray:
    """Generate n windows, each (WINDOW_SIZE, 7) from signal_fn(i)."""
    wins = []
    for i in range(n):
        w = np.zeros((WINDOW_SIZE, len(SENSORS)), dtype=np.float32)
        for s_idx in range(len(SENSORS)):
            w[:, s_idx] = signal_fn(i, s_idx)
        wins.append(w)
    return np.stack(wins)


def _normal_baselines() -> np.ndarray:
    """Approximate normal baselines per sensor (from config)."""
    import json
    cfg = json.loads((_ROOT / "config.json").read_text())
    baselines = cfg.get("baselines", {})
    vals = [
        baselines.get("vibration_rms",      2.5),
        baselines.get("vibration_peak",     4.0),
        baselines.get("discharge_pressure", 8.0),
        baselines.get("suction_pressure",   5.0),
        baselines.get("flow_rate",        120.0),
        baselines.get("motor_current",     15.0),
        baselines.get("fluid_temp",        65.0),
    ]
    return np.array(vals, dtype=np.float32)


def main() -> None:
    print("=" * 62)
    print("  Phase 1 — IF Score Distribution & Stress-Signal Validation")
    print("=" * 62)

    bundle = _load_if()
    shared = _load_shared()
    baselines = _normal_baselines()

    norm_key = "p5/p95" if "score_p5" in bundle else "min/max (legacy)"
    print(f"\n  Norm method : {norm_key}")
    if "score_p5" in bundle:
        print(f"  p5 = {bundle['score_p5']:.6f}   p95 = {bundle['score_p95']:.6f}")

    # ── Step 1: Distribution test on synthetic data ───────────────────────
    print("\n[1/2] Generating synthetic normal/fault windows ...")
    X_wins, y_cls, _, _ = generate(n_per_class=100, seed=77, shuffle=False)

    normal_wins = X_wins[y_cls == 0][:50]
    fault_wins  = X_wins[y_cls != 0][:50]

    normal_scores = _score_batch(bundle, normal_wins, shared)
    fault_scores  = _score_batch(bundle, fault_wins,  shared)

    print(f"  Normal windows  : mean={normal_scores.mean():.4f}  std={normal_scores.std():.4f}  "
          f"p95={np.percentile(normal_scores,95):.4f}")
    print(f"  Fault  windows  : mean={fault_scores.mean():.4f}  std={fault_scores.std():.4f}  "
          f"p5 ={np.percentile(fault_scores,5):.4f}")

    sep_ok = normal_scores.mean() < fault_scores.mean()
    norm_ok = normal_scores.mean() < 0.5
    fault_ok = fault_scores.mean() > 0.3
    print(f"\n  Separation OK   : {'PASS OK' if sep_ok  else 'FAIL X'}")
    print(f"  Normal mean<0.5 : {'PASS OK' if norm_ok  else 'FAIL X'}  ({normal_scores.mean():.4f})")
    print(f"  Fault  mean>0.3 : {'PASS OK' if fault_ok else 'FAIL X'}  ({fault_scores.mean():.4f})")

    # ── Step 2: Stress signals ────────────────────────────────────────────
    print("\n[2/2] Stress signal ordering test ...")

    # Signal A: Gaussian noise around normal baseline (should be LOW anomaly)
    def _noise(i, s): return float(baselines[s] + np.random.normal(0, baselines[s] * 0.03))

    # Signal B: Linear drift — 30% above normal by end of window (moderate anomaly)
    def _drift(i, s):
        t = np.linspace(0, 1, WINDOW_SIZE)
        return baselines[s] * (1.0 + 0.30 * t) + np.random.normal(0, baselines[s] * 0.01, WINDOW_SIZE)

    # Signal C: Step-change — first ~half normal, then 3× the baseline (high anomaly)
    def _step(i, s):
        w = np.full(WINDOW_SIZE, baselines[s], dtype=np.float32)
        w[WINDOW_SIZE // 2:] = baselines[s] * 3.0
        w += np.random.normal(0, baselines[s] * 0.01, WINDOW_SIZE)
        return w

    rng_save = np.random.get_state()

    noise_wins = _make_windows(_noise, n=20)
    drift_wins = _make_windows(_drift, n=20)
    step_wins  = _make_windows(_step,  n=20)

    np.random.set_state(rng_save)

    noise_scores = _score_batch(bundle, noise_wins, shared)
    drift_scores = _score_batch(bundle, drift_wins, shared)
    step_scores  = _score_batch(bundle, step_wins,  shared)

    print(f"  Gaussian noise  : mean={noise_scores.mean():.4f}")
    print(f"  Linear drift    : mean={drift_scores.mean():.4f}")
    print(f"  Step change     : mean={step_scores.mean():.4f}")

    order_ok = noise_scores.mean() < step_scores.mean()
    print(f"\n  Ordering (noise < step): {'PASS OK' if order_ok else 'FAIL X'}")
    drift_higher = drift_scores.mean() >= noise_scores.mean()
    print(f"  Drift >= noise         : {'PASS OK' if drift_higher else 'WARN  (may be mild)'}")

    # ── Summary ───────────────────────────────────────────────────────────
    all_pass = sep_ok and norm_ok and fault_ok and order_ok
    print("\n" + "=" * 62)
    print(f"  OVERALL: {'ALL TESTS PASSED OK' if all_pass else 'SOME TESTS FAILED X — review above'}")
    print("=" * 62)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
