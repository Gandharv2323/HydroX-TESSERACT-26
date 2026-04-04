"""
pipeline/fault_classifier.py — 5-class RandomForest fault classifier.

Classes (Spec §7.1)
-------------------
  0 = normal
  1 = bearing_fault
  2 = cavitation
  3 = dry_run
  4 = misalignment

Input  : 84-dim feature vector (from pipeline.features.extract_features)
Output : fault_class (int), fault_label (str), class_probabilities (dict)

Design decisions
----------------
- Trained on window-based features (not raw readings) for temporal consistency.
- class_weight="balanced" to handle unequal class sizes.
- Scaler fitted on training features only; persisted with model bundle.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Fixed label mapping — NEVER change order after training
FAULT_LABELS: dict[int, str] = {
    0: "normal",
    1: "bearing_fault",
    2: "cavitation",
    3: "dry_run",
    4: "misalignment",
}


class FaultClassifier:
    """
    5-class supervised fault classifier operating on the 84-dim feature vector.
    """

    def __init__(self) -> None:
        self._model:   Optional[RandomForestClassifier] = None
        self._scaler:  Optional[StandardScaler]         = None
        self._trained: bool                              = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,           # shape: (n_samples, 84)
        y: np.ndarray,           # shape: (n_samples,) integer labels 0-4
        n_estimators: int = 200,
        max_depth:    int = 12,
        random_state: int = 42,
    ) -> dict:
        """
        Fit StandardScaler + RandomForest on feature vectors.

        Returns
        -------
        dict with train accuracy, per-class support counts.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled, y)

        train_acc = self._model.score(X_scaled, y)
        counts = {FAULT_LABELS[c]: int((y == c).sum()) for c in sorted(FAULT_LABELS)}
        self._trained = True

        logger.info(
            f"[FaultClassifier] Trained — n={len(X)}, train_acc={train_acc:.4f}, "
            f"class_counts={counts}"
        )
        return {"train_accuracy": round(train_acc, 4), "class_counts": counts}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vec: np.ndarray) -> dict:
        """
        Predict fault class for a single 84-dim feature vector.

        Returns
        -------
        dict:
          fault_class_id  : int
          fault_label     : str
          probabilities   : dict[str, float]   per-class confidence
          confidence      : float  highest class probability
        """
        if not self._trained or self._model is None or self._scaler is None:
            raise RuntimeError("FaultClassifier not trained.")

        X = feature_vec.reshape(1, -1)
        X_s = self._scaler.transform(X)

        cls_id    = int(self._model.predict(X_s)[0])
        proba     = self._model.predict_proba(X_s)[0]   # shape: (5,)
        proba_map = {
            FAULT_LABELS[c]: round(float(proba[i]), 4)
            for i, c in enumerate(self._model.classes_)
        }
        return {
            "fault_class_id": cls_id,
            "fault_label":    FAULT_LABELS.get(cls_id, "unknown"),
            "probabilities":  proba_map,
            "confidence":     round(float(proba.max()), 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        bundle = {
            "model":   self._model,
            "scaler":  self._scaler,
            "labels":  FAULT_LABELS,
            "trained": self._trained,
        }
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        logger.info(f"[FaultClassifier] Saved → {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        self._model   = bundle["model"]
        self._scaler  = bundle["scaler"]
        self._trained = bundle["trained"]
        logger.info(f"[FaultClassifier] Loaded from {path}")

    def is_trained(self) -> bool:
        return self._trained
