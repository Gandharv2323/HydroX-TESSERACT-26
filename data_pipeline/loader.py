from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# Strict external schema
SCHEMA_SENSORS = [f"sensor_{i}" for i in range(1, 8)]
STRICT_COLUMNS = ["timestamp", *SCHEMA_SENSORS]

# Internal project sensor names (pipeline/buffer.py order)
INTERNAL_SENSORS = [
    "vibration_rms",
    "vibration_peak",
    "discharge_pressure",
    "suction_pressure",
    "flow_rate",
    "motor_current",
    "fluid_temp",
]

INTERNAL_TO_STRICT = dict(zip(INTERNAL_SENSORS, SCHEMA_SENSORS))
STRICT_TO_INTERNAL = dict(zip(SCHEMA_SENSORS, INTERNAL_SENSORS))
MAX_GAP_SECONDS = 5
MAX_GAP_STEPS = int(MAX_GAP_SECONDS / 0.5)


@dataclass
class SensorDataLoader:
    target_freq: str = "500ms"
    max_gap_steps: int = MAX_GAP_STEPS

    def load_csv(self, path: str) -> pd.DataFrame:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"CSV not found: {src}")

        df = pd.read_csv(src)
        df = self._ensure_timestamp(df)
        df = self._to_strict_sensor_schema(df)

        # Timestamp normalization: monotonic sort + duplicate consolidation
        df = df.sort_values("timestamp")
        df = df.groupby("timestamp", as_index=False)[SCHEMA_SENSORS].mean()

        # Resampling to 2 Hz
        df = (
            df.set_index("timestamp")
            .resample(self.target_freq)
            .mean()
            .reset_index()
        )

        # Interpolate short gaps (< 5 steps) + bounded forward fill fallback
        df[SCHEMA_SENSORS] = df[SCHEMA_SENSORS].interpolate(
            method="linear", limit=4, limit_direction="both"
        )
        df[SCHEMA_SENSORS] = df[SCHEMA_SENSORS].ffill(limit=self.max_gap_steps)

        # Drop rows that still contain NaNs after bounded filling (long gaps).
        df = df.dropna(subset=SCHEMA_SENSORS)

        # Validation filters
        df = self._drop_high_nan_ratio(df, max_nan_ratio=0.30)
        df = self._drop_flatline_rows(df)
        df = self._drop_out_of_percentile_bounds(df, low_q=0.01, high_q=0.99)

        # Final strict schema enforcement
        return df[STRICT_COLUMNS].reset_index(drop=True)

    def load_stream(self, source_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        mode = str(source_config.get("mode", "csv")).lower()
        if mode == "csv":
            path = source_config.get("path")
            if not path:
                raise ValueError("source_config['path'] is required when mode='csv'.")
            df = self.load_csv(str(path))
            for row in df.to_dict(orient="records"):
                yield {str(k): v for k, v in row.items()}
            return

        raise NotImplementedError(
            f"Unsupported stream mode '{mode}'. Currently supported: csv"
        )

    def to_internal_schema(self, df_strict: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in STRICT_COLUMNS if c not in df_strict.columns]
        if missing:
            raise ValueError(f"Strict schema columns missing: {missing}")
        out = df_strict.rename(columns=STRICT_TO_INTERNAL).copy()
        return out[["timestamp", *INTERNAL_SENSORS]]

    # ------------------------------
    # Internal helpers
    # ------------------------------

    @staticmethod
    def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "timestamp" not in out.columns:
            # Create a synthetic timestamp index when absent
            out["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(tz=timezone.utc).floor("s"),
                periods=len(out),
                freq="500ms",
            )
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out = out.dropna(subset=["timestamp"])
        return out

    @staticmethod
    def _to_strict_sensor_schema(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if all(c in out.columns for c in SCHEMA_SENSORS):
            return out[["timestamp", *SCHEMA_SENSORS]]

        # Backward compatibility: map internal names to strict schema
        if all(c in out.columns for c in INTERNAL_SENSORS):
            out = out.rename(columns=INTERNAL_TO_STRICT)
            return out[["timestamp", *SCHEMA_SENSORS]]

        missing = [c for c in SCHEMA_SENSORS if c not in out.columns]
        raise ValueError(
            "CSV does not match expected strict schema. Missing: "
            f"{missing}. Expected timestamp + sensor_1..sensor_7 "
            "or internal names vibration_rms..fluid_temp."
        )

    @staticmethod
    def _drop_high_nan_ratio(df: pd.DataFrame, max_nan_ratio: float) -> pd.DataFrame:
        na_ratio = df[SCHEMA_SENSORS].isna().mean(axis=1)
        return df.loc[na_ratio <= max_nan_ratio].copy()

    @staticmethod
    def _drop_flatline_rows(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
        # Reject rows where all sensor deltas vs previous row are effectively zero.
        delta = df[SCHEMA_SENSORS].diff().abs().fillna(1.0)
        keep = (delta > eps).any(axis=1)
        keep.iloc[0] = True
        return df.loc[keep].copy()

    @staticmethod
    def _drop_out_of_percentile_bounds(
        df: pd.DataFrame,
        low_q: float,
        high_q: float,
    ) -> pd.DataFrame:
        lo = df[SCHEMA_SENSORS].quantile(low_q)
        hi = df[SCHEMA_SENSORS].quantile(high_q)
        in_range = ((df[SCHEMA_SENSORS] >= lo) & (df[SCHEMA_SENSORS] <= hi)).all(axis=1)
        return df.loc[in_range].copy()
