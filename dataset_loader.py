"""
dataset_loader.py — Loads and prepares training data for PumpAnomalyDetector
and the advanced ML pipeline.

Kaggle (legacy) mode:
  - Uses 8 ANOVA F-score-selected sensor columns (best discriminators, AUC=0.998)
  - NORMAL rows → X_normal
  - RECOVERING rows → X_fault (7 BROKEN rows are too few; RECOVERING has 14k)

Kaggle Advanced mode (NEW):
  - Maps sensor_XX columns → 7 HydroX internal sensor names
  - Derives fault_class (int) and rul_hours (float) from machine_status
  - Returns a DataFrame ready for train_all.py --real-csv path

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


# ---------------------------------------------------------------------------
# Kaggle → Advanced pipeline mapper
# ---------------------------------------------------------------------------

# Mapping of HydroX internal sensor names → Kaggle sensor column.
# Based on physical range analysis of pump-sensor-data.csv:
#   vibration_rms  : sensor_00  (0–2.55,   mean=2.37  mm/s)
#   vibration_peak : sensor_18  (0–4.87,   mean=2.30  mm/s, slightly > rms)
#   discharge_press: sensor_06  (0–22.25,  mean=13.5  bar)
#   suction_press  : sensor_07  (0–23.6,   mean=15.8  bar)
#   flow_rate      : sensor_04  (0–800,    mean=590   — rescaled to 0–200)
#   motor_current  : sensor_05  (0–100,    mean=73.4  A)
#   fluid_temp     : sensor_01  (0–56.7,   mean=47.6  °C)
_KAGGLE_ADVANCED_MAP: dict[str, str] = {
    "vibration_rms":      "sensor_00",
    "vibration_peak":     "sensor_18",
    "discharge_pressure": "sensor_06",
    "suction_pressure":   "sensor_07",
    "flow_rate":          "sensor_04",
    "motor_current":      "sensor_05",
    "fluid_temp":         "sensor_01",
}

# Fault class IDs (mirror training/generate_data.py CLASS_LABELS)
_STATUS_TO_CLASS: dict[str, int] = {
    "NORMAL":     0,
    "RECOVERING": 1,  # mapped to bearing_fault — closest degradation proxy
    "BROKEN":     1,  # same fault class; only 7 samples but still useful
}

# RUL (hours) base values per status
_STATUS_BASE_RUL: dict[str, float] = {
    "NORMAL":     600.0,
    "RECOVERING":  50.0,
    "BROKEN":       1.0,
}


def detect_kaggle_advanced_format(csv_path: str) -> bool:
    """Return True if the CSV looks like the Kaggle pump-sensor-data format."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=2)
        return (
            "machine_status" in df.columns
            and "sensor_00" in df.columns
            and "sensor_04" in df.columns
        )
    except Exception:
        return False


class KaggleAdvancedLoader:
    """
    Maps the raw Kaggle pump-sensor-data.csv into a DataFrame that
    train_all.py can consume via the --real-csv pathway.

    Output columns:
      timestamp, vibration_rms, vibration_peak, discharge_pressure,
      suction_pressure, flow_rate, motor_current, fluid_temp,
      fault_class (int 0..4), rul_hours (float), machine_status (str)

    Usage
    -----
    loader = KaggleAdvancedLoader()
    df = loader.load("pump-sensor-data.csv")
    """

    # flow_rate rescale: Kaggle sensor_04 is 0–800, HydroX physical range is 0–200 L/min.
    _FLOW_SCALE = 200.0 / 800.0  # 0.25

    def load(self, csv_path: str, max_rows: int | None = None) -> "pd.DataFrame":
        """
        Load and map Kaggle CSV to HydroX internal schema.

        Parameters
        ----------
        csv_path : str
            Path to pump-sensor-data.csv.
        max_rows : int | None
            Optional row cap (useful in testing / low-RAM environments).

        Returns
        -------
        pd.DataFrame with columns: timestamp, 7 sensor names,
            fault_class (int), rul_hours (float), machine_status (str)
        """
        import pandas as pd
        import numpy as np

        print(f"[KaggleAdvancedLoader] Loading {csv_path} ...")
        df_raw = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)

        required = list(_KAGGLE_ADVANCED_MAP.values()) + ["machine_status"]
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"[KaggleAdvancedLoader] Missing expected columns: {missing}. "
                "Expected the Kaggle pump-sensor-data dataset."
            )

        # ── Timestamp ────────────────────────────────────────────────────────
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
        else:
            df_raw["timestamp"] = pd.date_range(
                "2018-01-01", periods=len(df_raw), freq="1min"
            )

        # ── Map sensor columns ────────────────────────────────────────────────
        out = pd.DataFrame()
        out["timestamp"] = df_raw["timestamp"].values

        sensor_cols = list(_KAGGLE_ADVANCED_MAP.keys())
        for internal_name, kaggle_col in _KAGGLE_ADVANCED_MAP.items():
            col = df_raw[kaggle_col].copy()
            # Fill NaN with rolling median, then overall median as fallback
            col = col.fillna(
                col.rolling(window=10, min_periods=1, center=True).median()
            )
            col = col.fillna(col.median())
            out[internal_name] = col.astype(np.float32).values

        # Rescale flow_rate to HydroX physical range (0–200 L/min)
        out["flow_rate"] = (
            (out["flow_rate"] * self._FLOW_SCALE).clip(0, 200).astype(np.float32)
        )

        # ── Labels ───────────────────────────────────────────────────────────
        status_upper = df_raw["machine_status"].str.upper().fillna("NORMAL")
        out["machine_status"] = status_upper.values
        out["fault_class"] = (
            status_upper.map(_STATUS_TO_CLASS).fillna(0).astype(np.int32).values
        )

        # ── RUL generation ───────────────────────────────────────────────────
        # Within each contiguous RECOVERING segment: linearly decay 50h → 1h.
        # NORMAL rows: 600h. BROKEN rows: 1h.
        base_rul = status_upper.map(_STATUS_BASE_RUL).fillna(600.0).astype(np.float32)
        rul = base_rul.to_numpy(dtype=np.float32)

        in_recovering = (status_upper == "RECOVERING").to_numpy()
        i = 0
        while i < len(in_recovering):
            if in_recovering[i]:
                j = i
                while j < len(in_recovering) and in_recovering[j]:
                    j += 1
                seg_len = j - i
                rul[i:j] = np.linspace(50.0, 1.0, seg_len, dtype=np.float32)
                i = j
            else:
                i += 1

        out["rul_hours"] = rul

        # ── Drop rows with bad timestamps ────────────────────────────────────
        out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

        n_normal = int((out["fault_class"] == 0).sum())
        n_fault  = int((out["fault_class"] != 0).sum())
        print(
            f"[KaggleAdvancedLoader] Mapped {len(out):,} rows — "
            f"normal={n_normal:,}, fault/recovering={n_fault:,} | "
            f"sensors: {sensor_cols}"
        )
        return out
