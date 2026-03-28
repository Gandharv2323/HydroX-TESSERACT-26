"""
dataset_loader.py — Loads and prepares training data for PumpAnomalyDetector.

Kaggle mode:
  - Uses 8 ANOVA F-score-selected sensor columns (best discriminators, AUC=0.998)
  - NORMAL rows → X_normal
  - RECOVERING rows → X_fault  (7 BROKEN rows are too few; RECOVERING has 14k)
  - Returns feature names so ml_model can store them in the pickle bundle

Synthetic mode:
  - Mirrors the 5 canonical sensor fields from sensor_sim.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ANOVA-selected best features (matches ml_model._KAGGLE_FEATURES)
_KAGGLE_FEATURES = [
    "sensor_04", "sensor_10", "sensor_11", "sensor_02",
    "sensor_12", "sensor_50", "sensor_01", "sensor_03",
]

# Canonical synthetic features
_SYNTH_FEATURES = [
    "vibration_rms", "discharge_pressure", "flow_rate",
    "motor_current", "fluid_temp",
]


class PumpDatasetLoader:

    def get_feature_names(self, mode: str = "synthetic") -> list[str]:
        return _KAGGLE_FEATURES if mode == "kaggle" else _SYNTH_FEATURES

    # ------------------------------------------------------------------
    # Kaggle CSV loader
    # ------------------------------------------------------------------

    def load_kaggle(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load pump-sensor-data.csv and return (X_normal, X_fault).

        Uses the 8 ANOVA-selected features that achieve AUC=0.998 separating
        NORMAL from RECOVERING (14,477 samples).

        Returns
        -------
        X_normal : rows where machine_status == "NORMAL"  (~205k rows)
        X_fault  : rows where machine_status == "RECOVERING" (~14k rows)
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Validate expected columns
        missing = [f for f in _KAGGLE_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing expected columns: {missing}. "
                "Expected the Kaggle 'pump-sensor-data' dataset."
            )

        # Fill NaN with median (more robust than mean for skewed sensor data)
        df[_KAGGLE_FEATURES] = df[_KAGGLE_FEATURES].apply(
            lambda col: col.fillna(col.median())
        )

        if "machine_status" not in df.columns:
            raise ValueError("'machine_status' column not found in CSV.")

        normal_mask     = df["machine_status"].str.upper() == "NORMAL"
        recovering_mask = df["machine_status"].str.upper() == "RECOVERING"

        X_normal = df.loc[normal_mask,     _KAGGLE_FEATURES].values.astype(float)
        X_fault  = df.loc[recovering_mask, _KAGGLE_FEATURES].values.astype(float)

        print(
            f"[dataset_loader] Kaggle data loaded — "
            f"normal={len(X_normal):,}, fault/recovering={len(X_fault):,}, "
            f"features={_KAGGLE_FEATURES[:3]}..."
        )
        return X_normal, X_fault

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------

    def generate_synthetic(
        self,
        n_normal: int = 1200,
        n_fault:  int = 400,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data via SensorSimulator.

        Returns numpy arrays of shape (n, 5) using _SYNTH_FEATURES column order.
        Fault mix: cavitation 150 + bearing_wear 150 + dry_run 100.
        """
        from sensor_sim import SensorSimulator

        sim = SensorSimulator()

        def _row(reading: dict) -> list[float]:
            return [reading[f] for f in _SYNTH_FEATURES]

        sim.set_mode("normal")
        X_normal = np.array([_row(sim.get_reading(step=i)) for i in range(n_normal)])

        fault_rows: list[list[float]] = []
        sim.set_mode("cavitation")
        fault_rows.extend([_row(sim.get_reading(step=i)) for i in range(150)])
        sim.set_mode("bearing_wear")
        fault_rows.extend([_row(sim.get_reading(step=i)) for i in range(150)])
        sim.set_mode("dry_run")
        fault_rows.extend([_row(sim.get_reading(step=i)) for i in range(100)])

        X_fault = np.array(fault_rows)

        print(
            f"[dataset_loader] Synthetic data — "
            f"normal={len(X_normal)}, fault={len(X_fault)}, "
            f"features={_SYNTH_FEATURES}"
        )
        return X_normal, X_fault
