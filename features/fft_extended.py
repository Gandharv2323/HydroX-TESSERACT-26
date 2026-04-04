from __future__ import annotations

import numpy as np


def extended_fft_features(signal: np.ndarray) -> np.ndarray:
    """
    Extended FFT descriptors with Hamming window.
    Returns: [spectral_centroid, spectral_bandwidth, harmonic_ratio]
    """
    x = signal.astype(np.float64)
    w = np.hamming(len(x))
    mag = np.abs(np.fft.rfft(x * w)) + 1e-12

    bins = np.arange(len(mag), dtype=np.float64)
    centroid = float(np.sum(bins * mag) / np.sum(mag))
    bandwidth = float(np.sqrt(np.sum(((bins - centroid) ** 2) * mag) / np.sum(mag)))

    # Harmonic ratio: energy in top-3 bins / total energy
    top_idx = np.argpartition(mag, -3)[-3:]
    harmonic_ratio = float(np.sum(mag[top_idx] ** 2) / np.sum(mag ** 2))

    return np.array([centroid, bandwidth, harmonic_ratio], dtype=np.float32)
