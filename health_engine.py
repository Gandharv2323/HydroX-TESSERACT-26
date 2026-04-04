"""
health_engine.py — Computes subsystem health scores, overall health, RUL,
                   NPSH margin, and pump curve deviation.

Upgrades v2:
  - compute() now returns "npsh" sub-dict (UPGRADE 1)
  - PumpCurveModel: head/efficiency curves, duty status, deviation check (UPGRADE 2)
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"

# ---------------------------------------------------------------------------
# UPGRADE 2 — Pump Curve Model
# ---------------------------------------------------------------------------

class PumpCurveModel:
    """
    Polynomial model for a typical end-suction centrifugal pump.

    Head curve  : H  = 45 − 0.0003 · Q²        [H in m, Q in L/min]
    Efficiency  : η  = −0.00002·Q² + 0.008·Q   [0–1, peaks ~0.72 at Q=200]
    """

    # Duty band boundaries
    _Q_THROTTLED  = 150.0   # L/min — below this = throttled
    _Q_OPTIMAL_HI = 250.0   # L/min — above this = overloaded warning
    _Q_OVERLOADED  = 300.0  # L/min — above this = overloaded

    def operating_point(self, flow_rate: float) -> dict:
        """Return head, efficiency and duty status at given flow."""
        Q = max(0.0, float(flow_rate))
        head_m       = 45.0 - 0.0003 * Q ** 2
        head_m       = max(0.0, head_m)
        efficiency   = max(0.0, min(1.0, -0.00002 * Q ** 2 + 0.008 * Q))
        efficiency_pct = round(efficiency * 100.0, 1)

        if Q < self._Q_THROTTLED:
            duty_status = "throttled"
        elif Q <= self._Q_OPTIMAL_HI:
            duty_status = "optimal"
        else:
            duty_status = "overloaded"

        return {
            "head_m":         round(head_m, 2),
            "efficiency_pct": efficiency_pct,
            "duty_status":    duty_status,
        }

    def deviation_from_curve(self, sensor_dict: dict) -> dict:
        """
        Compare actual discharge pressure (converted to head) vs expected
        head at the current measured flow rate.

        Tolerance: ±10 %.
        """
        Q            = float(sensor_dict.get("flow_rate", 120.0))
        P_bar        = float(sensor_dict.get("discharge_pressure", 4.2))
        actual_head  = round(P_bar * 10.2, 2)            # bar → metres of head
        expected_head = round(max(0.0, 45.0 - 0.0003 * Q ** 2), 2)

        if expected_head > 0:
            deviation_pct = round(
                (actual_head - expected_head) / expected_head * 100.0, 1
            )
        else:
            deviation_pct = 0.0

        within_tolerance = bool(abs(deviation_pct) <= 10.0)

        return {
            "expected_head":    expected_head,
            "actual_head":      actual_head,
            "deviation_pct":    deviation_pct,
            "within_tolerance": within_tolerance,
        }


# ---------------------------------------------------------------------------
# Health Engine
# ---------------------------------------------------------------------------

class HealthEngine:

    def __init__(self) -> None:
        with open(_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        self._baselines: dict[str, float] = cfg["baselines"]
        self._weights: dict[str, float]   = cfg["subsystem_weights"]
        self._thresholds: dict[str, int]  = cfg["health_thresholds"]
        self._rul_slope: float            = cfg["rul_slope"]
        self.pump_curve                   = PumpCurveModel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, sensor_dict: dict, anomaly_result: dict) -> dict:
        base_score = anomaly_result["anomaly_score"] * 100.0   # 0–100

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

        # ---- UPGRADE 1: NPSH -----------------------------------------------
        VAPOUR_PRESSURE_HEAD = 0.24    # metres, water at ~42 °C
        NPSHR                = 3.5     # rated minimum for this pump class
        sp   = float(sensor_dict.get("suction_pressure", 1.8))
        npsha       = round((sp * 10.2) - VAPOUR_PRESSURE_HEAD, 2)
        npsh_margin = round(npsha - NPSHR, 2)

        if npsh_margin > 0.5:
            npsh_status = "safe"
        elif npsh_margin > 0.0:
            npsh_status = "warning"
        else:
            npsh_status = "cavitation_risk"

        npsh = {
            "npsha":  npsha,
            "npshr":  NPSHR,
            "margin": npsh_margin,
            "status": npsh_status,
        }

        return {
            "overall_health":   overall,
            "subsystem_health": subsystem_health,
            "status":           status,
            "rul_hours":        rul_hours,
            "npsh":             npsh,
        }
