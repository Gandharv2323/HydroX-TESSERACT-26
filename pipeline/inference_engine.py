"""
pipeline/inference_engine.py — Real-time end-to-end inference pipeline.

Architecture (Spec §2):
  sensor_dict
    → validation_layer
    → SlidingWindowBuffer (push)
    → extract_features (time + FFT)
    → IsolationForest  → anomaly_score
    → FaultClassifier  → fault_class + confidence
    → RULPredictor     → RUL hours
    → decision_engine  → JSON output

Latency target: < 100ms per call.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.buffer import SlidingWindowBuffer
from pipeline.features import extract_features
from pipeline.fault_classifier import FaultClassifier
from pipeline.rul_lstm import RULPredictor
from calibration.threshold import load_threshold_config
from data_pipeline.preprocessing import ReadingPreprocessor

log = logging.getLogger(__name__)

# Physical sensor bounds (Spec §3.2)
_BOUNDS: dict[str, tuple[float, float]] = {
    "vibration_rms":      (0.0, 50.0),
    "vibration_peak":     (0.0, 120.0),
    "discharge_pressure": (0.0, 20.0),
    "suction_pressure":   (0.0, 15.0),
    "flow_rate":          (0.0, 600.0),
    "motor_current":      (0.0, 100.0),
    "fluid_temp":         (0.0, 200.0),
}

_DEFAULT_ANOMALY_THRESHOLD = 0.55


def _validate(sensor_dict: dict) -> list[str]:
    """Return list of violated constraint descriptions. Empty → valid."""
    violations: list[str] = []
    for sensor, (lo, hi) in _BOUNDS.items():
        val = sensor_dict.get(sensor)
        if val is None:
            violations.append(f"{sensor}: missing")
        elif not (lo <= float(val) <= hi):
            violations.append(f"{sensor}={val:.3f} out of [{lo}, {hi}]")
    return violations


class InferenceEngine:
    """
    Stateful (maintains sliding window) real-time inference engine.

    Usage
    -----
    engine = InferenceEngine(models_dir=Path("models"))
    engine.load()

    for sensor_dict in stream:
        result = engine.infer(sensor_dict)
        print(result)
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._buffer     = SlidingWindowBuffer()
        self._clf        = FaultClassifier()
        self._rul        = RULPredictor()
        self._iso        = None    # dict: model, scaler, score_min, score_max
        self._ready      = False
        self._threshold_path = models_dir.parent / "configs" / "threshold.json"
        self._anomaly_threshold = load_threshold_config(
            self._threshold_path,
            default=_DEFAULT_ANOMALY_THRESHOLD,
        )
        self._online_bounds = {k: (v[0], v[1]) for k, v in _BOUNDS.items()}
        self._pre = ReadingPreprocessor(
            sensor_names=list(_BOUNDS.keys()),
            smooth_window=3,
            clip_bounds=self._online_bounds,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all persisted model artefacts."""
        # Isolation Forest
        if_path = self._models_dir / "isolation_forest.pkl"
        if if_path.exists():
            with open(if_path, "rb") as fh:
                self._iso = pickle.load(fh)
            log.info("[InferenceEngine] IF loaded")
        else:
            log.warning("[InferenceEngine] isolation_forest.pkl not found — train first")

        # 5-class RF
        rf_path = self._models_dir / "fault_classifier.pkl"
        if rf_path.exists():
            self._clf.load(rf_path)
        else:
            log.warning("[InferenceEngine] fault_classifier.pkl not found — train first")

        # LSTM
        lstm_path = self._models_dir / "rul_lstm.pt"
        if lstm_path.exists():
            try:
                self._rul.load(lstm_path)
            except Exception as exc:
                log.warning(f"[InferenceEngine] LSTM load failed: {exc}")
        else:
            log.warning("[InferenceEngine] rul_lstm.pt not found — train first")

        self._ready = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, sensor_dict: dict) -> dict:
        """
        Run full pipeline for one timestep.

        Returns
        -------
        dict with keys: anomaly_score, fault_class, RUL, confidence,
                        state, latency_ms, sensor_error, violations
        """
        t0 = time.perf_counter()

        # ---- Validation layer -------------------------------------------
        cleaned, missing = self._pre.transform(sensor_dict)
        violations = _validate(cleaned)
        if violations:
            return {
                "sensor_error": True,
                "violations":   violations,
                "anomaly_score": None,
                "fault_class":   None,
                "RUL":           None,
                "confidence":    None,
                "state":         "sensor_error",
                "missing_mask":  missing,
                "latency_ms":    round((time.perf_counter() - t0) * 1000, 2),
            }

        # ---- Buffer update ----------------------------------------------
        self._buffer.push(cleaned)
        if not self._buffer.is_ready():
            return {
                "sensor_error":  False,
                "state":         "buffering",
                "buffer_fill":   self._buffer.fill_count(),
                "buffer_target": self._buffer.window_size,
                "missing_mask":  missing,
                "latency_ms":    round((time.perf_counter() - t0) * 1000, 2),
            }

        # ---- Feature extraction -----------------------------------------
        window   = self._buffer.get_window()          # (50, 7)
        feat_vec = extract_features(window)            # (84,)

        # ---- Isolation Forest → anomaly score ---------------------------
        anomaly_score = self._if_score(feat_vec)

        # ---- 5-class RF → fault class -----------------------------------
        if self._clf.is_trained():
            cls_result = self._clf.predict(feat_vec)
        else:
            cls_result = {"fault_label": "unknown", "fault_class_id": -1,
                          "confidence": 0.0, "probabilities": {}}

        # ---- LSTM → RUL -------------------------------------------------
        if self._rul.is_trained():
            rul_hours = self._rul.predict(window)
        else:
            # Fallback: linear estimate from anomaly score
            rul_hours = round(max(0.0, anomaly_score * 500.0), 1)

        # ---- Decision engine --------------------------------------------
        state = "anomalous" if (1.0 - anomaly_score) > self._anomaly_threshold else "normal"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "sensor_error":  False,
            "state":         state,
            "anomaly_score": round(1.0 - anomaly_score, 4),  # 1=worst, 0=healthy
            "fault_class":   cls_result.get("fault_label", "unknown"),
            "fault_class_id":cls_result.get("fault_class_id", -1),
            "probabilities": cls_result.get("probabilities", {}),
            "RUL":           rul_hours,
            "confidence":    cls_result.get("confidence", 0.0),
            "latency_ms":    latency_ms,
            "missing_mask":  missing,
            "violations":    [],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _if_score(self, feat_vec: np.ndarray) -> float:
        """Returns normalised IF score: 1.0=normal, 0.0=anomaly."""
        if self._iso is None:
            return 1.0   # Fallback when model not loaded

        X = feat_vec.reshape(1, -1)
        X_s = self._iso["scaler"].transform(X)
        raw = float(self._iso["model"].decision_function(X_s)[0])
        s_min = self._iso["score_min"]
        s_max = self._iso["score_max"]
        return float(np.clip((raw - s_min) / (s_max - s_min + 1e-12), 0.0, 1.0))
