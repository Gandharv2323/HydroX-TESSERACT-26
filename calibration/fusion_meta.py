from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("fusion_meta")

# Phase 5: latent compressed to PCA(8) before fusion to prevent 128-dim dominance
_LATENT_PCA_DIM = 8
_DOMINANCE_LIMIT = 0.60  # L1 share above this triggers a warning


def train_fusion_model(
    if_scores: np.ndarray,
    rf_fault_probs: np.ndarray,
    latent: np.ndarray,
    rul_pred: np.ndarray,
    hysteresis_signal: np.ndarray,
    y_binary: np.ndarray,
    out_path: Path | str,
) -> dict[str, Any]:
    """Train a rebalanced meta fusion model y = f(z_if, z_rf, z_hyst, z_pca, z_rul).

    Phase 5: latent vector is PCA-compressed to _LATENT_PCA_DIM components
    before fusion to prevent high-dimensional latent from dominating.
    """
    if_s = np.asarray(if_scores,     dtype=np.float32).reshape(-1, 1)
    rf_s = np.asarray(rf_fault_probs, dtype=np.float32).reshape(-1, 1)
    h    = np.asarray(latent,         dtype=np.float32)
    rul  = np.asarray(rul_pred,       dtype=np.float32).reshape(-1, 1)
    hyst = np.asarray(hysteresis_signal, dtype=np.float32).reshape(-1, 1)
    y    = np.asarray(y_binary,       dtype=np.int32)

    # ── PCA compress latent ──────────────────────────────────────────────
    actual_pca_dim = min(_LATENT_PCA_DIM, h.shape[1], len(y) - 1)
    if actual_pca_dim < 1:
        actual_pca_dim = 1
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    h_pca = pca.fit_transform(h).astype(np.float32)   # (N, actual_pca_dim)
    logger.info(
        f"[FusionMeta] PCA: {h.shape[1]}d → {actual_pca_dim}d  "
        f"(explained var: {pca.explained_variance_ratio_.sum():.3f})"
    )

    # ── Z-score normalise all components ────────────────────────────────
    if_mean   = float(if_s.mean())
    if_std    = float(if_s.std()  + 1e-6)
    rf_mean   = float(rf_s.mean())
    rf_std    = float(rf_s.std()  + 1e-6)
    pca_mean  = h_pca.mean(axis=0, keepdims=True)
    pca_std   = h_pca.std(axis=0,  keepdims=True) + 1e-6
    rul_mean  = float(rul.mean())
    rul_std   = float(rul.std()   + 1e-6)
    hyst_mean = float(hyst.mean())
    hyst_std  = float(hyst.std() + 1e-6)

    z_if  = (if_s  - if_mean)  / if_std
    z_rf  = (rf_s  - rf_mean)  / rf_std
    z_hyst = (hyst - hyst_mean) / hyst_std
    z_pca = (h_pca - pca_mean) / pca_std
    z_rul = (rul   - rul_mean) / rul_std

    X = np.hstack([z_if, z_rf, z_hyst, z_pca, z_rul])

    clf = LogisticRegression(max_iter=300, class_weight="balanced", C=0.3)
    clf.fit(X, y)

    p     = clf.predict_proba(X)[:, 1]
    coef  = clf.coef_[0]
    if_l1     = float(abs(coef[0]))
    rf_l1     = float(abs(coef[1]))
    hyst_l1   = float(abs(coef[2]))
    latent_l1 = float(np.sum(np.abs(coef[3:3 + actual_pca_dim])))
    rul_l1    = float(abs(coef[-1]))
    total_l1  = if_l1 + rf_l1 + hyst_l1 + latent_l1 + rul_l1 + 1e-9

    contributions = {
        "if":     round(if_l1     / total_l1, 4),
        "rf":     round(rf_l1     / total_l1, 4),
        "hysteresis": round(hyst_l1 / total_l1, 4),
        "latent": round(latent_l1 / total_l1, 4),
        "rul":    round(rul_l1    / total_l1, 4),
    }
    dominant = max(contributions.values())
    if dominant > _DOMINANCE_LIMIT:
        logger.warning(
            f"[FusionMeta] Dominance warning: one component has "
            f"{dominant:.1%} L1-share (limit={_DOMINANCE_LIMIT:.0%}). "
            f"Shares={contributions}"
        )
    else:
        logger.info(f"[FusionMeta] L1 shares OK: {contributions}")

    metrics = {
        "roc_auc":         round(float(roc_auc_score(y, p)),         4),
        "avg_precision":   round(float(average_precision_score(y, p)), 4),
        "n_samples":       int(len(y)),
        "n_features":      int(X.shape[1]),
        "pca_dim":         int(actual_pca_dim),
        "pca_explained_var": round(float(pca.explained_variance_ratio_.sum()), 4),
        "contributions_l1": contributions,
        "dominant_share":  round(dominant, 4),
    }

    payload = {
        "model": clf,
        "pca":   pca,
        "feature_layout": {
            "if":          1,
            "rf":          1,
            "hysteresis":  1,
            "latent_pca":  actual_pca_dim,
            "latent_orig": int(h.shape[1]),
            "rul":         1,
        },
        "normalization": {
            "if":  {"mean": if_mean,  "std": if_std},
            "rf":  {"mean": rf_mean,  "std": rf_std},
            "hysteresis": {"mean": hyst_mean, "std": hyst_std},
            "pca": {
                "mean": pca_mean.reshape(-1).tolist(),
                "std":  pca_std.reshape(-1).tolist(),
            },
            "rul": {"mean": rul_mean, "std": rul_std},
        },
        "metrics": metrics,
    }

    pth = Path(out_path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    with open(pth, "wb") as fh:
        pickle.dump(payload, fh)

    return metrics


def load_fusion_model(path: Path | str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def predict_fused_score(
    model_bundle: dict[str, Any],
    if_score:     float,
    rf_fault_prob: float,
    latent:       np.ndarray,
    rul_pred:     float,
    hysteresis_score: float = 0.0,
) -> float:
    layout     = model_bundle.get("feature_layout", {})
    latent_pca = int(layout.get("latent_pca", layout.get("latent", 0)))
    h          = np.asarray(latent, dtype=np.float32).reshape(1, -1)

    # ── PCA compress latent ──────────────────────────────────────────────
    pca = model_bundle.get("pca")
    if pca is not None:
        # new bundle: apply stored PCA
        try:
            h_pca = pca.transform(h).astype(np.float32)  # (1, pca_dim)
        except Exception:
            # Dimension mismatch (e.g. old model): fall back to zeros
            h_pca = np.zeros((1, latent_pca), dtype=np.float32)
    else:
        # Legacy bundle without PCA: use raw latent (compatible)
        h_raw = h.reshape(-1)
        orig_dim = int(layout.get("latent", h_raw.shape[0]))
        if len(h_raw) > orig_dim:
            h_raw = h_raw[:orig_dim]
        elif len(h_raw) < orig_dim:
            h_raw = np.pad(h_raw, (0, orig_dim - len(h_raw)))
        h_pca = h_raw.reshape(1, -1)

    norm    = model_bundle.get("normalization", {})
    if_norm = norm.get("if",  {"mean": 0.0, "std": 1.0})
    rf_norm = norm.get("rf",  {"mean": 0.0, "std": 1.0})
    hyst_norm = norm.get("hysteresis", {"mean": 0.0, "std": 1.0})
    # Support both new key 'pca' and legacy key 'latent'
    pca_norm_key = "pca" if "pca" in norm else "latent"
    pca_norm = norm.get(pca_norm_key, {"mean": [0.0] * latent_pca, "std": [1.0] * latent_pca})
    rul_norm = norm.get("rul", {"mean": 0.0, "std": 1.0})

    pca_dim     = h_pca.shape[1]
    pca_mean    = np.asarray(pca_norm.get("mean", [0.0] * pca_dim), dtype=np.float32).reshape(-1)
    pca_std     = np.asarray(pca_norm.get("std",  [1.0] * pca_dim), dtype=np.float32).reshape(-1)
    if len(pca_mean) != pca_dim:
        pca_mean = np.zeros(pca_dim, dtype=np.float32)
    if len(pca_std) != pca_dim:
        pca_std  = np.ones(pca_dim, dtype=np.float32)

    z_if  = (float(if_score)     - float(if_norm.get("mean",  0.0))) / float(if_norm.get("std",  1.0) + 1e-6)
    z_rf  = (float(rf_fault_prob) - float(rf_norm.get("mean", 0.0))) / float(rf_norm.get("std",  1.0) + 1e-6)
    z_hyst = (float(hysteresis_score) - float(hyst_norm.get("mean", 0.0))) / float(hyst_norm.get("std", 1.0) + 1e-6)
    z_pca = (h_pca - pca_mean) / (pca_std + 1e-6)                      # (1, pca_dim)
    z_rul = (float(rul_pred)    - float(rul_norm.get("mean", 0.0))) / float(rul_norm.get("std", 1.0) + 1e-6)

    layout = model_bundle.get("feature_layout", {})
    has_hyst = int(layout.get("hysteresis", 0)) == 1

    if has_hyst:
        x = np.concatenate([
            np.array([[z_if, z_rf, z_hyst]], dtype=np.float32),
            z_pca.astype(np.float32),
            np.array([[z_rul]],     dtype=np.float32),
        ], axis=1)   # (1, 3 + pca_dim + 1)
    else:
        # Backward compatibility with legacy fusion bundles.
        x = np.concatenate([
            np.array([[z_if, z_rf]], dtype=np.float32),
            z_pca.astype(np.float32),
            np.array([[z_rul]],     dtype=np.float32),
        ], axis=1)   # (1, 2 + pca_dim + 1)

    clf = model_bundle["model"]
    return float(np.clip(clf.predict_proba(x)[0, 1], 0.0, 1.0))
