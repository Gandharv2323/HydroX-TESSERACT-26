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
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.buffer import SlidingWindowBuffer
from pipeline.representation import build_hybrid_feature_vector
from pipeline.fault_classifier import FaultClassifier
from pipeline.rul_lstm import RULPredictor
from models.shared_latent import SharedLatentRuntime
from calibration.threshold import load_threshold_config
from calibration.score_fusion import (
    calibrate_if_score,
    calibrate_rf_probability,
    fuse_scores,
    load_fusion_config,
)
from calibration.fusion_meta import load_fusion_model, predict_fused_score
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
_HYST_ENTER = 0.60
_HYST_EXIT = 0.40
_HYST_PERSIST = 3
_MAX_GAP_STEPS = 10

_FAULT_TO_COMPONENT = {
    "normal": "none",
    "bearing_fault": "motor_bearing",
    "cavitation": "pump_inlet_hydraulics",
    "dry_run": "seal_and_fluid_path",
    "misalignment": "shaft_coupling",
    "unknown": "unknown_component",
}


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


def _derive_severity(anomaly: float, rul_h: float) -> str:
    if anomaly >= 0.80 or rul_h <= 24:
        return "critical"
    if anomaly >= 0.50 or rul_h <= 120:
        return "warning"
    return "normal"


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
        self._mask_buffer = SlidingWindowBuffer()
        self._clf        = FaultClassifier()
        self._rul        = RULPredictor()
        self._iso        = None    # dict: model, scaler, score_min, score_max
        self._shared: Optional[SharedLatentRuntime] = None
        self._fusion_meta = None
        self._ready      = False
        self._threshold_path = models_dir.parent / "configs" / "threshold.json"
        self._anomaly_threshold = load_threshold_config(
            self._threshold_path,
            default=_DEFAULT_ANOMALY_THRESHOLD,
        )
        self._fusion_cfg_path = models_dir.parent / "configs" / "score_fusion.json"
        self._fusion_cfg = load_fusion_config(self._fusion_cfg_path)
        self._fusion_meta_path = models_dir / "fusion_meta.pkl"
        self._online_bounds = {k: (v[0], v[1]) for k, v in _BOUNDS.items()}
        self._pre = ReadingPreprocessor(
            sensor_names=list(_BOUNDS.keys()),
            smooth_window=3,
            clip_bounds=self._online_bounds,
        )
        self._contradiction_counts: dict[str, int] = {
            "high_fault_prob_high_rul": 0,
            "low_anomaly_critical": 0,
        }
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=100)
        self._latency_hist_ms: deque[float] = deque(maxlen=500)
        self._missing_counts: dict[str, int] = {k: 0 for k in _BOUNDS}
        self._anom_streak = 0
        self._normal_streak = 0
        self._state = "normal"
        self._missing_streak: dict[str, int] = {k: 0 for k in _BOUNDS}

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

        # Shared latent encoder (optional, enables representation-consistent hybrid input)
        shared_path = self._models_dir / "shared_latent.pt"
        if shared_path.exists():
            try:
                self._shared = SharedLatentRuntime.load(shared_path)
                log.info("[InferenceEngine] shared_latent.pt loaded")
            except Exception as exc:
                log.warning(f"[InferenceEngine] shared encoder load failed: {exc}")
                self._shared = None
        else:
            log.warning("[InferenceEngine] shared_latent.pt not found — falling back to feature-only")

        self._fusion_meta = load_fusion_model(self._fusion_meta_path)
        if self._fusion_meta is not None:
            log.info("[InferenceEngine] fusion_meta.pkl loaded")

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
        for s, is_missing in missing.items():
            if int(is_missing) == 1:
                self._missing_streak[s] += 1
                self._missing_counts[s] += 1
            else:
                self._missing_streak[s] = 0

        long_gap = [s for s, n in self._missing_streak.items() if n > _MAX_GAP_STEPS]
        violations = _validate(cleaned)
        if long_gap:
            violations.extend([f"{s}: missing gap>{_MAX_GAP_STEPS} steps" for s in long_gap])
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
        self._mask_buffer.push(missing)
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
        mask_window = self._mask_buffer.get_window()
        hybrid_vec, _, _, latent_h = build_hybrid_feature_vector(window, self._shared, mask_window=mask_window)

        # ---- Isolation Forest → anomaly score ---------------------------
        if_normal_score = self._if_score(hybrid_vec)
        if_anomaly_raw = 1.0 - if_normal_score

        # ---- 5-class RF → fault class -----------------------------------
        if self._clf.is_trained():
            cls_result = self._clf.predict(hybrid_vec)
        else:
            cls_result = {"fault_label": "unknown", "fault_class_id": -1,
                          "confidence": 0.0, "probabilities": {}}

        probs = cls_result.get("probabilities", {})
        rf_fault_prob_raw = 1.0 - float(probs.get("normal", 0.0)) if probs else 0.0

        # ---- Score calibration + fusion ---------------------------------
        if_anomaly_cal = calibrate_if_score(if_anomaly_raw, self._fusion_cfg)
        rf_fault_prob_cal = calibrate_rf_probability(rf_fault_prob_raw, self._fusion_cfg)
        if self._fusion_meta is not None:
            fused_anomaly = predict_fused_score(
                self._fusion_meta,
                if_anomaly_cal,
                rf_fault_prob_cal,
                latent_h,
            )
        else:
            fused_anomaly = fuse_scores(if_anomaly_cal, rf_fault_prob_cal, self._fusion_cfg)

        # ---- LSTM → RUL -------------------------------------------------
        if self._rul.is_trained():
            rul_window = window
            expected_in = int(self._rul._kwargs.get("input_size", window.shape[1])) if hasattr(self._rul, "_kwargs") else window.shape[1]
            if expected_in == window.shape[1] * 2 and mask_window is not None:
                rul_window = np.concatenate([window.astype(np.float32), mask_window.astype(np.float32)], axis=1)
            rul_uq = self._rul.predict_with_uncertainty(rul_window, n_samples=30)
            rul_hours = float(rul_uq["mean"])
        else:
            # Fallback: linear estimate from anomaly score
            rul_hours = round(max(0.0, (1.0 - fused_anomaly) * 500.0), 1)
            rul_uq = {
                "mean": float(rul_hours),
                "low": float(rul_hours),
                "high": float(rul_hours),
                "std": 0.0,
                "samples": 1,
            }

        # ---- Consistency constraints ------------------------------------
        consistency_actions: list[str] = []
        # hard rule: high anomaly + high confidence caps max RUL
        if fused_anomaly >= 0.80 and float(cls_result.get("confidence", 0.0)) >= 0.70:
            if rul_hours > 120.0:
                rul_hours = 120.0
                rul_uq["mean"] = 120.0
                rul_uq["high"] = min(float(rul_uq.get("high", 120.0)), 120.0)
                consistency_actions.append("cap_rul_high_anomaly")

        severity = _derive_severity(fused_anomaly, rul_hours)
        # hard rule: very low RUL forces at least warning/critical
        if rul_hours <= 24.0 and severity != "critical":
            severity = "critical"
            consistency_actions.append("force_critical_low_rul")
        elif rul_hours <= 72.0 and severity == "normal":
            severity = "warning"
            consistency_actions.append("force_warning_low_rul")

        # ---- Contradiction telemetry ------------------------------------
        contradictions: list[str] = []
        if rf_fault_prob_cal >= 0.80 and rul_hours >= 240.0:
            key = "high_fault_prob_high_rul"
            self._contradiction_counts[key] += 1
            contradictions.append(key)
        if fused_anomaly <= 0.20 and severity == "critical":
            key = "low_anomaly_critical"
            self._contradiction_counts[key] += 1
            contradictions.append(key)

        for c in contradictions:
            event = {
                "ts": round(time.time(), 3),
                "type": c,
                "fault_class": cls_result.get("fault_label", "unknown"),
                "anomaly": round(float(fused_anomaly), 4),
                "rul": round(float(rul_hours), 2),
            }
            self._recent_events.append(event)
            log.warning(f"[InferenceEngine] Contradiction detected: {event}")

        # ---- Decision engine (hysteresis + persistence) -----------------
        state = self._update_state_hysteresis(fused_anomaly)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._latency_hist_ms.append(float(latency_ms))
        p95 = float(np.percentile(np.asarray(self._latency_hist_ms, dtype=np.float32), 95)) if self._latency_hist_ms else 0.0

        return {
            "sensor_error":  False,
            "state":         state,
            "severity":      severity,
            "anomaly_score": round(float(fused_anomaly), 4),
            "scores": {
                "if_raw": round(float(if_anomaly_raw), 4),
                "if_calibrated": round(float(if_anomaly_cal), 4),
                "rf_fault_prob_raw": round(float(rf_fault_prob_raw), 4),
                "rf_fault_prob_calibrated": round(float(rf_fault_prob_cal), 4),
                "fused": round(float(fused_anomaly), 4),
                "fusion_mode": "meta" if self._fusion_meta is not None else "weighted",
                "hysteresis": {
                    "enter": _HYST_ENTER,
                    "exit": _HYST_EXIT,
                    "persist": _HYST_PERSIST,
                },
            },
            "fault_class":   cls_result.get("fault_label", "unknown"),
            "component":     _FAULT_TO_COMPONENT.get(cls_result.get("fault_label", "unknown"), "unknown_component"),
            "fault_class_id":cls_result.get("fault_class_id", -1),
            "probabilities": cls_result.get("probabilities", {}),
            "RUL":           round(float(rul_hours), 1),
            "RUL_uncertainty": rul_uq,
            "confidence":    cls_result.get("confidence", 0.0),
            "consistency_actions": consistency_actions,
            "contradictions": contradictions,
            "telemetry": {
                "contradiction_counts": dict(self._contradiction_counts),
                "recent_contradictions": list(self._recent_events)[-5:],
                "missing_counts": dict(self._missing_counts),
                "latency": {
                    "samples": len(self._latency_hist_ms),
                    "p95_ms": round(p95, 2),
                },
            },
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

        X = self._align_to_if(feat_vec).reshape(1, -1)
        X_s = self._iso["scaler"].transform(X)
        raw = float(self._iso["model"].decision_function(X_s)[0])
        s_min = self._iso["score_min"]
        s_max = self._iso["score_max"]
        return float(np.clip((raw - s_min) / (s_max - s_min + 1e-12), 0.0, 1.0))

    def _align_to_if(self, feat_vec: np.ndarray) -> np.ndarray:
        expected = int(getattr(self._iso["scaler"], "n_features_in_", len(feat_vec)))
        v = np.asarray(feat_vec, dtype=np.float32).reshape(-1)
        if len(v) == expected:
            return v
        if len(v) > expected:
            return v[:expected]
        return np.pad(v, (0, expected - len(v)), mode="constant", constant_values=0.0)

    def _update_state_hysteresis(self, anomaly_score: float) -> str:
        if anomaly_score >= _HYST_ENTER:
            self._anom_streak += 1
            self._normal_streak = 0
        elif anomaly_score <= _HYST_EXIT:
            self._normal_streak += 1
            self._anom_streak = 0
        else:
            self._anom_streak = 0
            self._normal_streak = 0

        if self._anom_streak >= _HYST_PERSIST:
            self._state = "anomalous"
        elif self._normal_streak >= _HYST_PERSIST:
            self._state = "normal"
        return self._state
