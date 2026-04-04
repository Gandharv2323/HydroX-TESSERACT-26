"""
pipeline/buffer.py — Sliding window time-series buffer.

Maintains a per-sensor rolling deque of the last `window_size` readings.
Ensures chronological ordering with no gaps; drops oldest on overflow.
Spec §3.3: window_size=50, stride=1.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

# Canonical sensor columns (fixed order — NEVER change after training)
SENSORS: list[str] = [
    "vibration_rms",
    "vibration_peak",
    "discharge_pressure",
    "suction_pressure",
    "flow_rate",
    "motor_current",
    "fluid_temp",
]

WINDOW_SIZE: int = 50


class SlidingWindowBuffer:
    """
    Thread-safe (GIL-protected) per-sensor sliding window.

    Usage
    -----
    buf = SlidingWindowBuffer()
    buf.push(sensor_dict)
    if buf.is_ready():
        window = buf.get_window()   # shape: (50, 7)
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        sensors: list[str] = SENSORS,
    ) -> None:
        self.window_size = window_size
        self.sensors = sensors
        self._buffers: dict[str, deque[float]] = {
            s: deque(maxlen=window_size) for s in sensors
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, sensor_dict: dict) -> None:
        """Append one timestep to every sensor buffer."""
        for s in self.sensors:
            self._buffers[s].append(float(sensor_dict.get(s, 0.0)))

    def is_ready(self) -> bool:
        """True once every sensor buffer has exactly `window_size` entries."""
        return all(len(self._buffers[s]) == self.window_size for s in self.sensors)

    def get_window(self) -> Optional[np.ndarray]:
        """
        Returns the current window as ndarray of shape (window_size, n_sensors).
        Column order matches `self.sensors` — same as SENSORS constant.
        Returns None if the buffer is not yet filled.
        """
        if not self.is_ready():
            return None
        return np.column_stack([list(self._buffers[s]) for s in self.sensors])

    def reset(self) -> None:
        """Clear all buffers (use on mode switches or sensor errors)."""
        for s in self.sensors:
            self._buffers[s].clear()

    def fill_count(self) -> int:
        """How many samples have been pushed (capped at window_size)."""
        return len(self._buffers[self.sensors[0]])
