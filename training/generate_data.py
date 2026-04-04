"""
training/generate_data.py — Synthetic labeled training data for all 5 fault classes.

Generates windowed samples (window_size=50, n_sensors=7) with:
  - Class-balanced distribution across all 5 fault types
  - Realistic RUL targets derived from degradation severity
  - Misalignment class (not in SensorSimulator) modelled analytically here

Output
------
  X_windows : ndarray (n_total, 50, 7)        raw sensor windows
  y_class   : ndarray (n_total,)  int labels  0-4
  y_rul     : ndarray (n_total,)  float       hours to failure
  feature_X : ndarray (n_total, 84)           extracted features (for RF/IF)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make parent importable when run standalone
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.buffer import SENSORS, WINDOW_SIZE
from pipeline.features import extract_batch


# ------------------------------------------------------------------
# Baselines (mirrors config.json)
# ------------------------------------------------------------------

_BASELINES: dict[str, float] = {
    "vibration_rms":      2.1,
    "vibration_peak":     4.8,
    "discharge_pressure": 4.2,
    "suction_pressure":   1.8,
    "flow_rate":         120.0,
    "motor_current":      18.5,
    "fluid_temp":         42.0,
}

_RNG = np.random.default_rng(42)


def _noise(val: float, pct: float = 0.03) -> float:
    return val + _RNG.normal(0, abs(val) * pct)


def _make_window(
    mode: str,
    severity: float = 0.0,   # 0.0 (mild) → 1.0 (severe)
) -> np.ndarray:
    """
    Generate one (WINDOW_SIZE, 7) window for a given fault mode and severity.
    """
    b = _BASELINES
    rows = []

    for step in range(WINDOW_SIZE):
        # ---- Build reading per mode --------------------------------------
        if mode == "normal":
            r = {s: _noise(b[s]) for s in SENSORS}

        elif mode == "bearing_fault":
            # Progressive vibration + motor current rise
            r = {
                "vibration_rms":      _noise(b["vibration_rms"]      * (1.0 + severity * 2.2)),
                "vibration_peak":     _noise(b["vibration_peak"]      * (1.0 + severity * 2.8)),
                "discharge_pressure": _noise(b["discharge_pressure"]),
                "suction_pressure":   _noise(b["suction_pressure"]),
                "flow_rate":          _noise(b["flow_rate"]),
                "motor_current":      _noise(b["motor_current"]       * (1.0 + severity * 0.18)),
                "fluid_temp":         _noise(b["fluid_temp"]          + severity * 10.0),
            }

        elif mode == "cavitation":
            r = {
                "vibration_rms":      _noise(b["vibration_rms"]  * _RNG.uniform(1.5, 2.0)),
                "vibration_peak":     _noise(b["vibration_peak"] * _RNG.uniform(1.8, 2.5)),
                "discharge_pressure": _noise(b["discharge_pressure"]),
                "suction_pressure":   _noise(b["suction_pressure"] * _RNG.uniform(0.40, 0.65)),
                "flow_rate":          _noise(b["flow_rate"]       * _RNG.uniform(0.65, 0.82)),
                "motor_current":      _noise(b["motor_current"]   * _RNG.uniform(0.88, 0.96)),
                "fluid_temp":         _noise(b["fluid_temp"]),
            }

        elif mode == "dry_run":
            mc_factor = 1.5 + severity * 0.8
            r = {
                "vibration_rms":      _noise(b["vibration_rms"]  * _RNG.uniform(1.3, 1.8)),
                "vibration_peak":     _noise(b["vibration_peak"] * _RNG.uniform(1.2, 1.6)),
                "discharge_pressure": _noise(b["discharge_pressure"] * 0.50),
                "suction_pressure":   _noise(b["suction_pressure"]   * 0.08),
                "flow_rate":          _noise(b["flow_rate"]          * 0.03),
                "motor_current":      _noise(b["motor_current"]      * mc_factor),
                "fluid_temp":         _noise(b["fluid_temp"]         + severity * 25.0),
            }

        elif mode == "misalignment":
            # Misalignment: elevated vibration at 2× fundamental + radial load
            # Creates a characteristic high-frequency vibration signature detectable via FFT
            phase = 2.0 * np.pi * step / WINDOW_SIZE
            vib_extra = b["vibration_rms"] * severity * 1.6 * (1.0 + 0.4 * np.sin(2 * phase))
            r = {
                "vibration_rms":      _noise(b["vibration_rms"]      + vib_extra),
                "vibration_peak":     _noise(b["vibration_peak"]      + vib_extra * 1.3),
                "discharge_pressure": _noise(b["discharge_pressure"]  * (1.0 - severity * 0.05)),
                "suction_pressure":   _noise(b["suction_pressure"]),
                "flow_rate":          _noise(b["flow_rate"]           * (1.0 - severity * 0.06)),
                "motor_current":      _noise(b["motor_current"]       * (1.0 + severity * 0.12)),
                "fluid_temp":         _noise(b["fluid_temp"]          + severity * 5.0),
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Ensure sensor order matches SENSORS constant
        rows.append([r[s] for s in SENSORS])

    return np.array(rows, dtype=np.float32)   # (WINDOW_SIZE, 7)


def _rul_for(mode: str, severity: float) -> float:
    """
    Realistic RUL (hours) based on fault mode and severity.
      - Normal: 400–600 hrs
      - Early fault (sev<0.3): 150–400 hrs
      - Mid fault  (sev<0.6): 50–150 hrs
      - Severe     (sev>=0.6): 5–50 hrs
    """
    if mode == "normal":
        return float(_RNG.uniform(400, 600))

    # Degradation curve: faster for acute faults
    speed = {"bearing_fault": 1.0, "cavitation": 1.2, "dry_run": 1.5, "misalignment": 0.8}
    factor = speed.get(mode, 1.0)
    base = max(5.0, 500.0 * (1.0 - severity) ** 2 / factor)
    return float(_RNG.uniform(base * 0.8, base * 1.2))


# ------------------------------------------------------------------
# Main generator
# ------------------------------------------------------------------

def generate(
    n_per_class: int = 300,
    seed:        int = 42,
    shuffle: bool = True,
    return_time_index: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate balanced multi-class training data.

    Parameters
    ----------
    n_per_class : samples per fault class (normal gets 2x for balance)

    Returns
    -------
    X_windows : (n_total, WINDOW_SIZE, 7)
    y_class   : (n_total,)      int labels 0-4
    y_rul     : (n_total,)      float hours
    X_features: (n_total, 84)  feature vectors
    """
    global _RNG
    _RNG = np.random.default_rng(seed)

    modes = [
        ("normal",        0),
        ("bearing_fault", 1),
        ("cavitation",    2),
        ("dry_run",       3),
        ("misalignment",  4),
    ]

    per_mode_windows: dict[str, list[np.ndarray]] = {m: [] for m, _ in modes}
    per_mode_labels: dict[str, list[int]] = {m: [] for m, _ in modes}
    per_mode_rul: dict[str, list[float]] = {m: [] for m, _ in modes}

    for mode_name, class_id in modes:
        n = n_per_class
        sev_values = (
            np.zeros(n) if mode_name == "normal"
            else _RNG.uniform(0.1, 0.95, size=n)
        )
        for k in range(n):
            sev = float(sev_values[k])
            w   = _make_window(mode_name, severity=sev)
            per_mode_windows[mode_name].append(w)
            per_mode_labels[mode_name].append(class_id)
            per_mode_rul[mode_name].append(_rul_for(mode_name, sev))

    # Temporal interleave (no leakage via shuffling, but avoids one-class contiguous blocks)
    windows_list: list[np.ndarray] = []
    labels_list: list[int] = []
    rul_list: list[float] = []
    max_len = max(len(per_mode_windows[m]) for m, _ in modes)
    for i in range(max_len):
        for mode_name, _ in modes:
            if i < len(per_mode_windows[mode_name]):
                windows_list.append(per_mode_windows[mode_name][i])
                labels_list.append(per_mode_labels[mode_name][i])
                rul_list.append(per_mode_rul[mode_name][i])

    X_wins = np.stack(windows_list, axis=0)          # (n, 50, 7)
    y_cls  = np.array(labels_list,  dtype=np.int32)  # (n,)
    y_rul  = np.array(rul_list,     dtype=np.float32)# (n,)

    print(f"[generate_data] Extracting features for {len(X_wins)} windows...")
    X_feats = extract_batch(X_wins)                  # (n, 84)

    t_idx = np.arange(len(X_wins), dtype=np.int64)
    if shuffle:
        idx = _RNG.permutation(len(X_wins))
        X_wins, y_cls, y_rul, X_feats, t_idx = X_wins[idx], y_cls[idx], y_rul[idx], X_feats[idx], t_idx[idx]

    class_counts = {m: int((y_cls == c).sum()) for m, c in modes}
    print(f"[generate_data] Done — total={len(X_wins)}, class_counts={class_counts}")

    if return_time_index:
        return X_wins, y_cls, y_rul, X_feats, t_idx
    return X_wins, y_cls, y_rul, X_feats
