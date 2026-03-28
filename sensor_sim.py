"""
sensor_sim.py — Synthetic pump sensor data generator.
Supports 4 fault scenario modes: normal, cavitation, bearing_wear, dry_run.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


_CONFIG_PATH = Path(__file__).parent / "config.json"


class SensorSimulator:
    VALID_MODES = {"normal", "cavitation", "bearing_wear", "dry_run"}

    def __init__(self) -> None:
        with open(_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        self._baselines: dict[str, float] = cfg["baselines"]
        self._noise_pct: float = cfg["noise_pct"]
        self.mode: str = "normal"
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {self.VALID_MODES}")
        self.mode = mode
        self._step = 0

    def reset(self) -> None:
        self.mode = "normal"
        self._step = 0

    def get_reading(self, step: int | None = None) -> dict:
        if step is None:
            step = self._step
            self._step += 1

        b = self._baselines
        p = self._noise_pct

        def _n(val: float, extra_pct: float = 0.0) -> float:
            return val + random.gauss(0, abs(val) * (p + extra_pct))

        if self.mode == "normal":
            return {
                "vibration_rms":       _n(b["vibration_rms"]),
                "vibration_peak":      _n(b["vibration_peak"]),
                "discharge_pressure":  _n(b["discharge_pressure"]),
                "suction_pressure":    _n(b["suction_pressure"]),
                "flow_rate":           _n(b["flow_rate"]),
                "motor_current":       _n(b["motor_current"]),
                "fluid_temp":          _n(b["fluid_temp"]),
            }

        elif self.mode == "cavitation":
            sp_mult   = random.uniform(0.45, 0.62)
            vrms_mult = random.uniform(1.7, 2.1)
            vpk_mult  = random.uniform(2.0, 2.8)
            fr_mult   = random.uniform(0.65, 0.80)
            mc_mult   = random.uniform(0.88, 0.96)
            return {
                "vibration_rms":       _n(b["vibration_rms"]  * vrms_mult),
                "vibration_peak":      _n(b["vibration_peak"] * vpk_mult),
                "discharge_pressure":  _n(b["discharge_pressure"]),
                "suction_pressure":    _n(b["suction_pressure"] * sp_mult),
                "flow_rate":           _n(b["flow_rate"] * fr_mult, extra_pct=0.15),
                "motor_current":       _n(b["motor_current"] * mc_mult),
                "fluid_temp":          _n(b["fluid_temp"]),
            }

        elif self.mode == "bearing_wear":
            severity = min(1.0, step * 0.008)
            return {
                "vibration_rms":       _n(b["vibration_rms"]  * (1.0 + severity * 1.8)),
                "vibration_peak":      _n(b["vibration_peak"] * (1.0 + severity * 2.2)),
                "discharge_pressure":  _n(b["discharge_pressure"]),
                "suction_pressure":    _n(b["suction_pressure"]),
                "flow_rate":           _n(b["flow_rate"]),
                "motor_current":       _n(b["motor_current"] * (1.0 + severity * 0.15)),
                "fluid_temp":          _n(b["fluid_temp"] + severity * 8.0),
            }

        elif self.mode == "dry_run":
            capped = min(step, 60)
            mc_mult = 1.6 + capped * 0.01
            vr_mult = random.uniform(1.3, 1.7)
            return {
                "vibration_rms":       _n(b["vibration_rms"]  * vr_mult),
                "vibration_peak":      _n(b["vibration_peak"] * vr_mult * 0.9),
                "discharge_pressure":  _n(b["discharge_pressure"] * 0.55),
                "suction_pressure":    _n(b["suction_pressure"] * 0.10),
                "flow_rate":           _n(b["flow_rate"] * 0.04),
                "motor_current":       _n(b["motor_current"] * mc_mult),
                "fluid_temp":          _n(b["fluid_temp"] + capped * 0.5),
            }

        # Fallback
        return {k: _n(v) for k, v in b.items()}
