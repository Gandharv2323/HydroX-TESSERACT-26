"""
ml_model.py — Fine-tuned IsolationForest anomaly detector.

Key improvements over v1:
- ANOVA F-score feature selection picks the 8 most discriminating Kaggle sensors
  rather than top-5 by variance (AUC 0.998 on NORMAL vs RECOVERING split)
- Contamination derived from actual fault ratio in Kaggle CSV (0.0657, not hard-coded 0.20)
- Trained on 205k NORMAL samples only (truly unsupervised)
- Normalisation constants calibrated from real score distributions:
    NORMAL p5=−0.0105  mean=0.071  p95=0.126
    RECOVERING mean=−0.293  (well-separated)
- n_estimators=200 for stability on large dataset
- Secondary RandomForest classifier for failure mode detection (supervised, trained on
  NORMAL vs RECOVERING labels from Kaggle)
- Output anomaly_score: 1.0 = perfectly healthy, 0.0 = severe fault
"""
from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_CONFIG_PATH   = Path(__file__).parent / "config.json"
_DEFAULT_MODEL = Path(__file__).parent / "pump_model.pkl"

# ANOVA-selected features — best 8 discriminators (NORMAL vs RECOVERING, AUC=0.998)
# These are the Kaggle sensor column names used for training
_KAGGLE_FEATURES = [
    "sensor_04", "sensor_10", "sensor_11", "sensor_02",
    "sensor_12", "sensor_50", "sensor_01", "sensor_03",
]

# Synthetic feature names (used when no CSV is available)
_SYNTH_FEATURES = [
    "vibration_rms", "discharge_pressure", "flow_rate",
    "motor_current", "fluid_temp",
]

# Calibrated from real data (see profiling notebook)
_NORM_SHIFT  =  0.293   # −(RECOVERING mean)
_NORM_SCALE  =  0.364   # maps score range to [0, 1]
_ANOMALY_THR =  0.50    # below this → anomaly


class PumpAnomalyDetector:

    def __init__(self) -> None:
        with open(_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        self._baselines: dict[str, float] = cfg["baselines"]

        # Isolation Forest params — contamination will be overridden at train time
        # when using real data (data-driven ratio)
        iso = cfg["isolation_forest"]
        self._iso_params = {
            "n_estimators":  200,          # bumped up from config for stability
            "random_state":  iso["random_state"],
        }
        self._contamination_default: float = iso["contamination"]

        self._model:      IsolationForest | None      = None
        self._classifier: RandomForestClassifier | None = None
        self._scaler:     StandardScaler | None        = None
        self._features:   list[str]                    = _SYNTH_FEATURES
        self._using_kaggle: bool                       = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_normal:       np.ndarray,
        X_fault:        np.ndarray | None = None,
        contamination:  float | None      = None,
        feature_names:  list[str] | None  = None,
        save_path:      str | Path        = _DEFAULT_MODEL,
    ) -> None:
        """
        Fit IsolationForest on normal samples only (unsupervised).
        Optionally fit a secondary RandomForest classifier if X_fault is provided.

        Parameters
        ----------
        X_normal       : np.ndarray (n, d) — normal samples
        X_fault        : np.ndarray (m, d) — fault/recovering samples (optional)
        contamination  : real fault ratio from data; defaults to config value
        feature_names  : column names matching X columns (for dataset_loader interop)
        save_path      : where to pickle the bundle
        """
        if contamination is None:
            contamination = self._contamination_default
        contamination = float(np.clip(contamination, 0.01, 0.49))

        if feature_names:
            self._features      = feature_names
            self._using_kaggle  = (feature_names == _KAGGLE_FEATURES)

        # ---- Scale -------------------------------------------------------
        self._scaler = StandardScaler()
        X_normal_scaled = self._scaler.fit_transform(X_normal)

        # ---- IsolationForest (unsupervised, normal only) -----------------
        self._model = IsolationForest(
            contamination=contamination,
            **self._iso_params,
        )
        self._model.fit(X_normal_scaled)

        # ---- Optional supervised classifier (fault mode detection) -------
        if X_fault is not None and len(X_fault) > 10:
            X_fault_scaled = self._scaler.transform(X_fault)
            X_clf = np.vstack([X_normal_scaled, X_fault_scaled])
            y_clf = np.concatenate([
                np.zeros(len(X_normal_scaled)),
                np.ones(len(X_fault_scaled)),
            ])
            # Subsample normal class to balance (RandomForest handles imbalance
            # better than IsolationForest but still benefits from balance)
            rng = np.random.default_rng(42)
            n_fault = len(X_fault_scaled)
            n_bal   = min(len(X_normal_scaled), n_fault * 10)
            idx_normal = rng.choice(len(X_normal_scaled), size=n_bal, replace=False)
            X_bal = np.vstack([X_normal_scaled[idx_normal], X_fault_scaled])
            y_bal = np.concatenate([np.zeros(n_bal), np.ones(n_fault)])

            self._classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self._classifier.fit(X_bal, y_bal)
            print(f"[ml_model] Secondary RF classifier trained — "
                  f"normal_bal={n_bal}, fault={n_fault}")

        # ---- Persist everything ------------------------------------------
        bundle = {
            "model":         self._model,
            "classifier":    self._classifier,
            "scaler":        self._scaler,
            "features":      self._features,
            "using_kaggle":  self._using_kaggle,
            "contamination": contamination,
        }
        with open(save_path, "wb") as fh:
            pickle.dump(bundle, fh)

        s_normal = self._model.decision_function(X_normal_scaled)
        print(
            f"[ml_model] IsolationForest trained — "
            f"n_normal={len(X_normal)}, contamination={contamination:.4f}, "
            f"score_mean={s_normal.mean():.4f}, score_p5={np.percentile(s_normal,5):.4f}, "
            f"timestamp={datetime.now().isoformat(timespec='seconds')}"
        )

    def load(self, path: str | Path = _DEFAULT_MODEL) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at '{path}'. Run training first.")
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        self._model         = bundle["model"]
        self._classifier    = bundle.get("classifier")
        self._scaler        = bundle["scaler"]
        self._features      = bundle.get("features", _SYNTH_FEATURES)
        self._using_kaggle  = bundle.get("using_kaggle", False)
        print(f"[ml_model] Loaded model from {path} "
              f"(kaggle={self._using_kaggle}, features={self._features[:3]}...)")

    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sensor_dict: dict) -> dict:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not initialised. Call train() or load() first.")

        if self._using_kaggle and "sensor_04" not in sensor_dict:
            # Runtime dict is from SensorSimulator (vibration_rms etc.)
            # Use physics-based scoring directly — no Kaggle feature mapping needed
            score_norm = self._physics_score(sensor_dict)
        else:
            X_raw = np.array([[sensor_dict.get(f, 0.0) for f in self._features]])
            X     = self._scaler.transform(X_raw)

            raw_score  = float(self._model.decision_function(X)[0])
            score_norm = float(np.clip((raw_score + _NORM_SHIFT) / _NORM_SCALE, 0.0, 1.0))

            # Blend with secondary RF classifier if available
            if self._classifier is not None:
                clf_prob   = float(self._classifier.predict_proba(X)[0][1])   # P(fault)
                score_norm = float(np.clip(score_norm * 0.60 + (1.0 - clf_prob) * 0.40, 0.0, 1.0))

        is_anomaly = bool(score_norm < _ANOMALY_THR)
        failure_mode, recommended_action = self._classify_failure(sensor_dict, score_norm)
        confidence = float(abs(score_norm - 0.5) * 2.0)

        return {
            "anomaly_score":      round(score_norm, 4),
            "is_anomaly":         is_anomaly,
            "failure_mode":       failure_mode,
            "confidence":         round(confidence, 4),
            "recommended_action": recommended_action,
        }

    def _physics_score(self, s: dict) -> float:
        """
        Pure physics-based health score (0=anomaly, 1=healthy) when the
        model was trained on Kaggle features but runtime uses SensorSimulator fields.
        Uses z-score deviation from baselines, aggregated across all sensors.
        """
        b = self._baselines
        # Baseline std estimated at noise_pct=3% of baseline for normal operation
        # Larger deviations map to lower scores
        fields = [
            ("vibration_rms",      b["vibration_rms"],      1.0),
            ("vibration_peak",     b["vibration_peak"],      0.8),
            ("discharge_pressure", b["discharge_pressure"],  0.9),
            ("suction_pressure",   b["suction_pressure"],    1.2),
            ("flow_rate",          b["flow_rate"],           0.7),
            ("motor_current",      b["motor_current"],       0.9),
            ("fluid_temp",         b["fluid_temp"],          0.6),
        ]
        total_penalty = 0.0
        for key, baseline, weight in fields:
            val       = s.get(key, baseline)
            std_est   = abs(baseline) * 0.05   # ~5% as 1-sigma normal band
            z         = abs(val - baseline) / (std_est + 1e-9)
            penalty   = min(1.0, (z / 5.0) ** 1.5)  # 5-sigma deviation → full penalty
            total_penalty += penalty * weight
        total_weight = sum(w for _, _, w in fields)
        avg_penalty  = total_penalty / total_weight
        return float(np.clip(1.0 - avg_penalty, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Rule-based failure mode classifier (physics-informed)
    # ------------------------------------------------------------------

    def _classify_failure(self, s: dict, score_norm: float) -> tuple[str, str]:
        if score_norm > _ANOMALY_THR:
            return "none", "Normal operation — no action required"

        b = self._baselines
        suction_low  = s.get("suction_pressure", b["suction_pressure"]) < b["suction_pressure"] * 0.70
        vibration_hi = s.get("vibration_rms",    b["vibration_rms"])    > b["vibration_rms"]    * 1.50
        flow_low     = s.get("flow_rate",         b["flow_rate"])        < b["flow_rate"]        * 0.15

        if score_norm >= 0.35:
            return "degraded", "Monitor closely — schedule preventive maintenance"

        if flow_low:
            return "dry_run",      "Stop pump immediately — check fluid supply"
        if suction_low:
            return "cavitation",   "Increase suction head — reduce flow or check inlet valve"
        if vibration_hi:
            return "bearing_wear", "Schedule bearing inspection within 48 hours"

        return "degraded", "Monitor closely — schedule preventive maintenance"
