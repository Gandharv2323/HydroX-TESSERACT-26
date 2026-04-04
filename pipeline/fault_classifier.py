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
from sklearn.calibration import CalibratedClassifierCV
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
_UNKNOWN_CONFIDENCE_THRESHOLD = 0.60


class FaultClassifier:
    """
    5-class supervised fault classifier operating on the 84-dim feature vector.
    """

    def __init__(self) -> None:
        self._model:   Optional[RandomForestClassifier] = None
        self._calibrator: Optional[CalibratedClassifierCV] = None
        self._scaler:  Optional[StandardScaler]         = None
        self._trained: bool                              = False
        self._calibration_method: str                    = "none"

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
        calibration_method: str = "none",
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

        calibration_method = calibration_method.lower().strip()
        self._calibrator = None
        self._calibration_method = calibration_method
        if calibration_method in {"sigmoid", "isotonic"}:
            self._calibrator = CalibratedClassifierCV(
                self._model,
                method=calibration_method,
                cv=3,
            )
            self._calibrator.fit(X_scaled, y)

        infer_model = self._calibrator if self._calibrator is not None else self._model
        train_acc = infer_model.score(X_scaled, y)
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

        vec = np.asarray(feature_vec, dtype=np.float32).reshape(-1)
        expected = int(getattr(self._scaler, "n_features_in_", len(vec)))
        if len(vec) > expected:
            vec = vec[:expected]
        elif len(vec) < expected:
            vec = np.pad(vec, (0, expected - len(vec)), mode="constant", constant_values=0.0)

        X = vec.reshape(1, -1)
        X_s = self._scaler.transform(X)

        infer_model = self._calibrator if self._calibrator is not None else self._model
        cls_id    = int(infer_model.predict(X_s)[0])
        proba     = infer_model.predict_proba(X_s)[0]   # shape: (5,)
        proba_map = {
            FAULT_LABELS[c]: round(float(proba[i]), 4)
            for i, c in enumerate(infer_model.classes_)
        }
        max_conf = float(proba.max())
        predicted_label = FAULT_LABELS.get(cls_id, "unknown")
        predicted_id = cls_id
        if max_conf < _UNKNOWN_CONFIDENCE_THRESHOLD:
            predicted_label = "unknown"
            predicted_id = -1

        return {
            "fault_class_id": predicted_id,
            "fault_label":    predicted_label,
            "probabilities":  proba_map,
            "confidence":     round(max_conf, 4),
            "calibration":    self._calibration_method,
            "unknown_threshold": _UNKNOWN_CONFIDENCE_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        bundle = {
            "model":   self._model,
            "calibrator": self._calibrator,
            "scaler":  self._scaler,
            "labels":  FAULT_LABELS,
            "trained": self._trained,
            "calibration_method": self._calibration_method,
        }
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        logger.info(f"[FaultClassifier] Saved → {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        self._model   = bundle["model"]
        self._calibrator = bundle.get("calibrator")
        self._scaler  = bundle["scaler"]
        self._trained = bundle["trained"]
        self._calibration_method = str(bundle.get("calibration_method", "none"))
        logger.info(f"[FaultClassifier] Loaded from {path}")

    def is_trained(self) -> bool:
        return self._trained
