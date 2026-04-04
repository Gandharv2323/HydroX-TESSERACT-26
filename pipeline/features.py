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
    """
    return np.vstack([extract_features(windows[i]) for i in range(len(windows))])


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
