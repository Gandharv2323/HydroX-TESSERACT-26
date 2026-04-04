from __future__ import annotations

from typing import Optional

import numpy as np

from models.shared_latent import SharedLatentRuntime
from pipeline.features import extract_features, extract_phase_features

ENGINEERED_DIM = 84
PHASE_DIM = 15
LATENT_DIM = 128
HYBRID_DIM = ENGINEERED_DIM + PHASE_DIM + LATENT_DIM


def build_hybrid_feature_vector(
    window: np.ndarray,
    encoder: Optional[SharedLatentRuntime],
    mask_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single source-of-truth feature builder used in training/inference/evaluation.

    Returns
    -------
    hybrid_vec, engineered_vec, phase_vec, latent_vec
    """
    engineered = extract_features(window).astype(np.float32)
    phase = extract_phase_features(window).astype(np.float32)

    if encoder is None:
        latent = np.zeros((LATENT_DIM,), dtype=np.float32)
    else:
        enc_in = window
        in_ch = int(getattr(encoder, "_in_channels", window.shape[1]))
        if in_ch == window.shape[1] * 2:
            if mask_window is None:
                mask_window = np.zeros_like(window, dtype=np.float32)
            enc_in = np.concatenate([window.astype(np.float32), mask_window.astype(np.float32)], axis=1)
        latent = np.asarray(encoder.encode_window(enc_in), dtype=np.float32).reshape(-1)
        if len(latent) > LATENT_DIM:
            latent = latent[:LATENT_DIM]
        elif len(latent) < LATENT_DIM:
            latent = np.pad(latent, (0, LATENT_DIM - len(latent)), mode="constant", constant_values=0.0)

    hybrid = np.concatenate([engineered, phase, latent], axis=0).astype(np.float32)
    assert hybrid.shape[-1] == HYBRID_DIM, f"Unexpected hybrid dim: {hybrid.shape[-1]} != {HYBRID_DIM}"
    return hybrid, engineered, phase, latent


def build_hybrid_feature_batch(
    windows: np.ndarray,
    encoder: Optional[SharedLatentRuntime],
    mask_windows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hybrid_list: list[np.ndarray] = []
    eng_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    lat_list: list[np.ndarray] = []
    for i in range(len(windows)):
        m = None if mask_windows is None else mask_windows[i]
        h, z, p, l = build_hybrid_feature_vector(windows[i], encoder, mask_window=m)
        hybrid_list.append(h)
        eng_list.append(z)
        phase_list.append(p)
        lat_list.append(l)

    hybrid = np.vstack(hybrid_list).astype(np.float32)
    eng = np.vstack(eng_list).astype(np.float32)
    phase = np.vstack(phase_list).astype(np.float32)
    lat = np.vstack(lat_list).astype(np.float32)
    assert hybrid.shape[1] == HYBRID_DIM
    return hybrid, eng, phase, lat
