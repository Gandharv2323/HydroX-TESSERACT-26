"""
health_engine.py — Computes subsystem health scores, overall health, RUL, and status.
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"


class HealthEngine:

    def __init__(self) -> None:
        with open(_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        self._baselines: dict[str, float] = cfg["baselines"]
        self._weights: dict[str, float]   = cfg["subsystem_weights"]
        self._thresholds: dict[str, int]  = cfg["health_thresholds"]
        self._rul_slope: float            = cfg["rul_slope"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, sensor_dict: dict, anomaly_result: dict) -> dict:
        base_score = anomaly_result["anomaly_score"] * 100.0  # 0–100

        b = self._baselines

        def _dev_penalty(sensor_key: str, weight: float = 40.0) -> float:
            val      = sensor_dict.get(sensor_key, b[sensor_key])
            baseline = b[sensor_key]
            deviation = abs(val - baseline) / (abs(baseline) + 1e-9)
            return deviation * weight

        # ---- Subsystem scores ------------------------------------------
        # bearing: vibration_rms, vibration_peak, motor_current
        bearing_penalty = (
            _dev_penalty("vibration_rms")    * 0.4
            + _dev_penalty("vibration_peak") * 0.35
            + _dev_penalty("motor_current")  * 0.25
        )
        bearing = float(max(0.0, min(100.0, base_score - bearing_penalty)))

        # seal: discharge_pressure, flow_rate
        seal_penalty = (
            _dev_penalty("discharge_pressure") * 0.5
            + _dev_penalty("flow_rate")        * 0.5
        )
        seal = float(max(0.0, min(100.0, base_score - seal_penalty)))

        # impeller: suction_pressure, flow_rate
        impeller_penalty = (
            _dev_penalty("suction_pressure") * 0.6
            + _dev_penalty("flow_rate")      * 0.4
        )
        impeller = float(max(0.0, min(100.0, base_score - impeller_penalty)))

        # casing: fluid_temp
        casing = float(max(0.0, min(100.0, base_score - _dev_penalty("fluid_temp"))))

        subsystem_health = {
            "bearing_front": round(bearing, 1),
            "bearing_rear":  round(bearing * 0.97, 1),   # slight asymmetry for realism
            "seal":          round(seal, 1),
            "impeller":      round(impeller, 1),
            "casing":        round(casing, 1),
        }

        # ---- Overall (bearing = avg of front/rear, matches config weight) ----
        bearing_avg = (subsystem_health["bearing_front"] + subsystem_health["bearing_rear"]) / 2.0
        overall = (
            bearing_avg * self._weights["bearing"]
            + seal       * self._weights["seal"]
            + impeller   * self._weights["impeller"]
            + casing     * self._weights["casing"]
        )
        overall = round(float(max(0.0, min(100.0, overall))), 1)

        # ---- Status --------------------------------------------------------
        if overall > self._thresholds["healthy"]:
            status = "healthy"
        elif overall >= self._thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        rul_hours = round(max(0.0, (overall - 40.0) * self._rul_slope), 1)

        return {
            "overall_health":   overall,
            "subsystem_health": subsystem_health,
            "status":           status,
            "rul_hours":        rul_hours,
        }
