from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def clip_outliers_percentile(
    df: pd.DataFrame,
    sensor_cols: list[str],
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    lo = out[sensor_cols].quantile(low_q)
    hi = out[sensor_cols].quantile(high_q)
    out[sensor_cols] = out[sensor_cols].clip(lower=lo, upper=hi, axis=1)
    return out


def smooth_noise_rolling_mean(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    out[sensor_cols] = out[sensor_cols].rolling(window=window, min_periods=1).mean()
    return out


def append_missingness_mask_columns(
    df: pd.DataFrame,
    sensor_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for idx, col in enumerate(sensor_cols, start=1):
        out[f"mask_{idx}"] = out[col].isna().astype(int)
    return out


def augment_feature_vector_with_missingness(
    feature_vec: np.ndarray,
    missing_mask: Iterable[int],
) -> np.ndarray:
    return np.concatenate([feature_vec.astype(np.float32), np.array(list(missing_mask), dtype=np.float32)])


@dataclass
class ReadingPreprocessor:
    sensor_names: list[str]
    smooth_window: int = 3
    clip_bounds: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self._hist: dict[str, deque[float]] = {
            s: deque(maxlen=self.smooth_window) for s in self.sensor_names
        }

    def transform(self, reading: dict[str, Any]) -> tuple[dict[str, float], dict[str, int]]:
        cleaned: dict[str, float] = {}
        missing: dict[str, int] = {}

        for s in self.sensor_names:
            raw = reading.get(s, None)
            miss = raw is None or (isinstance(raw, float) and np.isnan(raw))
            missing[s] = int(miss)

            if miss:
                # Graceful fallback for missing online points
                prev = self._hist[s][-1] if self._hist[s] else 0.0
                val = float(prev)
            else:
                assert raw is not None
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    missing[s] = 1
                    prev = self._hist[s][-1] if self._hist[s] else 0.0
                    val = float(prev)

            if self.clip_bounds and s in self.clip_bounds:
                lo, hi = self.clip_bounds[s]
                val = float(np.clip(val, lo, hi))

            self._hist[s].append(val)
            cleaned[s] = float(np.mean(self._hist[s]))

        return cleaned, missing


def build_percentile_bounds(
    df: pd.DataFrame,
    sensor_cols: list[str],
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> dict[str, tuple[float, float]]:
    lo = df[sensor_cols].quantile(low_q)
    hi = df[sensor_cols].quantile(high_q)
    return {c: (float(lo[c]), float(hi[c])) for c in sensor_cols}
