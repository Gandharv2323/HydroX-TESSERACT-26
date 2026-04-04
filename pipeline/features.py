"""
pipeline/features.py — Time-domain + frequency-domain feature extraction.

Spec §4:
  Time-domain (7): mean, std, min, max, slope, delta, rolling_variance
  FFT (5):         fft_mean, fft_max, fft_energy, dominant_freq, spectral_entropy

One feature vector per sensor × both domains = 7 sensors × 12 = 84 features.
Feature order is FIXED and must NEVER change between training and inference.
"""
from __future__ import annotations

import numpy as np

# Feature count constants — used externally for validation
N_TIME_FEATURES = 7
N_FFT_FEATURES  = 5
N_SENSORS       = 7          # matches buffer.SENSORS
N_FEATURES      = N_SENSORS * (N_TIME_FEATURES + N_FFT_FEATURES)  # 84


# ------------------------------------------------------------------
# Low-level extractors
# ------------------------------------------------------------------

def _time_domain_features(signal: np.ndarray) -> np.ndarray:
    """
    7 time-domain statistics for a 1-D signal of length window_size.

    Order (fixed): mean, std, min, max, slope, delta, rolling_variance
    """
    n = len(signal)
    mean = float(np.mean(signal))
    std  = float(np.std(signal, ddof=0))
    mn   = float(np.min(signal))
    mx   = float(np.max(signal))

    # OLS slope — linear regression coefficient
    x     = np.arange(n, dtype=float)
    x_bar = x.mean()
    s_xx  = float(np.sum((x - x_bar) ** 2))
    slope = float(np.sum((x - x_bar) * (signal - mean)) / (s_xx + 1e-12))

    delta = float(signal[-1] - signal[0])

    # Rolling variance over last 10 samples
    rolling_var = float(np.var(signal[-10:], ddof=0))

    return np.array([mean, std, mn, mx, slope, delta, rolling_var], dtype=np.float32)


def _fft_features(signal: np.ndarray) -> np.ndarray:
    """
    5 frequency-domain features using real FFT magnitude spectrum.

    Order (fixed): fft_mean, fft_max, fft_energy, dominant_freq, spectral_entropy
    """
    win = np.hanning(len(signal))
    sig_w = signal * win
    mag = np.abs(np.fft.rfft(sig_w))   # magnitude spectrum, length n//2 + 1

    fft_mean   = float(np.mean(mag))
    fft_max    = float(np.max(mag))
    fft_energy = float(np.sum(mag ** 2))
    dom_freq   = float(np.argmax(mag))  # index = dominant frequency bin

    # Spectral entropy (Shannon, over normalised power spectrum)
    psd      = mag ** 2 + 1e-12
    psd_norm = psd / np.sum(psd)
    spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

    return np.array([fft_mean, fft_max, fft_energy, dom_freq, spec_ent], dtype=np.float32)


def _phase_features(signal: np.ndarray) -> np.ndarray:
    """Phase-aware descriptors from real FFT.

    Returns
    -------
    [phase_variance, phase_drift]
    """
    win = np.hanning(len(signal))
    sig_w = signal * win
    ph = np.angle(np.fft.rfft(sig_w))
    if len(ph) <= 1:
        return np.array([0.0, 0.0], dtype=np.float32)
    phase_var = float(np.var(ph, ddof=0))
    phase_drift = float(np.mean(np.abs(np.diff(ph))))
    return np.array([phase_var, phase_drift], dtype=np.float32)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract the full 84-dim feature vector from a (window_size, n_sensors) window.

    Parameters
    ----------
    window : np.ndarray, shape (window_size, n_sensors)
        Chronologically ordered sensor matrix from SlidingWindowBuffer.

    Returns
    -------
    features : np.ndarray, shape (84,)
        Concatenated [time_domain(s0), fft(s0), time_domain(s1), fft(s1), ...]
        per-sensor features in fixed column order.
    """
    if window.ndim != 2 or window.shape[1] != N_SENSORS:
        raise ValueError(
            f"Expected window shape (*, {N_SENSORS}), got {window.shape}"
        )

    parts: list[np.ndarray] = []
    for col_idx in range(window.shape[1]):
        sig = window[:, col_idx].astype(np.float64)
        parts.append(_time_domain_features(sig))
        parts.append(_fft_features(sig))

    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape == (N_FEATURES,), f"Feature dim mismatch: {vec.shape}"
    return vec


def extract_batch(windows: np.ndarray) -> np.ndarray:
    """
    Vectorised extraction over a batch of windows.

    Parameters
    ----------
    windows : np.ndarray, shape (n_samples, window_size, n_sensors)

    Returns
    -------
    np.ndarray, shape (n_samples, 84)

    Implementation: fully vectorised NumPy — no per-sample loop.
    """
    if len(windows) == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    # windows: (N, T, C)
    N, T, C = windows.shape
    sig = windows.astype(np.float64)   # (N, T, C)

    # ── Time-domain stats (all N×C at once) ──────────────────────────────
    mean_ = sig.mean(axis=1)                          # (N, C)
    std_  = sig.std(axis=1, ddof=0)                   # (N, C)
    min_  = sig.min(axis=1)                           # (N, C)
    max_  = sig.max(axis=1)                           # (N, C)

    # OLS slope per sensor (vectorised)
    x     = np.arange(T, dtype=np.float64)
    x_bar = x.mean()
    s_xx  = float(np.sum((x - x_bar) ** 2))
    x_c   = x - x_bar                                # (T,)
    slope = (sig * x_c[:, np.newaxis]).sum(axis=1) / (s_xx + 1e-12)   # (N, C)

    delta        = sig[:, -1, :] - sig[:, 0, :]      # (N, C)
    rolling_var  = sig[:, -10:, :].var(axis=1, ddof=0)  # (N, C)

    # Stack time-domain: (N, C, 7) → (N, C*7)
    td = np.stack([mean_, std_, min_, max_, slope, delta, rolling_var], axis=2)  # (N,C,7)

    # ── FFT features (all N×C at once) ───────────────────────────────────
    hann   = np.hanning(T).reshape(1, T, 1)                  # (1,T,1)
    sig_w  = sig * hann                                       # (N, T, C)
    mag    = np.abs(np.fft.rfft(sig_w, axis=1))              # (N, T//2+1, C)

    fft_mean   = mag.mean(axis=1)                             # (N, C)
    fft_max    = mag.max(axis=1)                              # (N, C)
    fft_energy = (mag ** 2).sum(axis=1)                       # (N, C)
    dom_freq   = mag.argmax(axis=1).astype(np.float64)        # (N, C)

    # Spectral entropy
    psd      = mag ** 2 + 1e-12                               # (N, F, C)
    psd_sum  = psd.sum(axis=1, keepdims=True)                 # (N, 1, C)
    psd_n    = psd / psd_sum                                  # (N, F, C)
    spec_ent = -(psd_n * np.log(psd_n + 1e-12)).sum(axis=1)  # (N, C)

    # Stack FFT: (N, C, 5)
    fd = np.stack([fft_mean, fft_max, fft_energy, dom_freq, spec_ent], axis=2)  # (N,C,5)

    # ── Combine: for each sensor concat [7 td, 5 fft] → stride sensor order ─
    # td: (N,C,7)  fd: (N,C,5) → combined: (N,C,12) → (N, C*12) = (N,84)
    combined = np.concatenate([td, fd], axis=2)               # (N, C, 12)
    # Flatten in sensor-major order: [s0_td, s0_fft, s1_td, s1_fft, ...]
    out = combined.reshape(N, C * (N_TIME_FEATURES + N_FFT_FEATURES))

    return out.astype(np.float32)


def extract_phase_features(window: np.ndarray) -> np.ndarray:
    """Extract phase descriptors per sensor + cross-sensor phase coherence.

    Output layout:
      - per sensor: [phase_variance, phase_drift]  => 14 values
      - cross-sensor: mean absolute phase-difference => 1 value
    Total: 15 values
    """
    if window.ndim != 2 or window.shape[1] != N_SENSORS:
        raise ValueError(f"Expected window shape (*, {N_SENSORS}), got {window.shape}")

    parts: list[np.ndarray] = []
    phases = []
    for col_idx in range(window.shape[1]):
        sig = window[:, col_idx].astype(np.float64)
        parts.append(_phase_features(sig))
        win = np.hanning(len(sig))
        phases.append(np.angle(np.fft.rfft(sig * win)))

    diffs = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            m = min(len(phases[i]), len(phases[j]))
            if m > 0:
                diffs.append(float(np.mean(np.abs(phases[i][:m] - phases[j][:m]))))
    cross = np.array([float(np.mean(diffs)) if diffs else 0.0], dtype=np.float32)
    return np.concatenate(parts + [cross]).astype(np.float32)
