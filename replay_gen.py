"""
replay_gen.py — Generates replay.json for offline demo (120 snapshots).

Segments:
   0– 29: normal        — all healthy, throttle 1.0
  30– 49: bearing_wear  — gradual degradation (steps 0-19)
  50– 69: cavitation    — NPSH drops below safe (steps 0-19)
  70– 89: dry_run       — rapid deterioration (steps 0-19)
  90–119: normal        — gradual recovery to 90+

Each snapshot includes the complete v2 schema:
  timestamp, mode, step, throttle_factor, sensors (with shaft_rpm),
  anomaly, health (with npsh), pump_curve.

Usage:
  python replay_gen.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dataset_loader import PumpDatasetLoader
from health_engine import HealthEngine
from ml_model import PumpAnomalyDetector
from sensor_sim import SensorSimulator

_BASE_DIR   = Path(__file__).parent
_MODEL_PATH = _BASE_DIR / "pump_model.pkl"
_OUT_PATH   = _BASE_DIR / "replay.json"


def _make_frame(
    sim: SensorSimulator,
    detector: PumpAnomalyDetector,
    engine: HealthEngine,
    mode: str,
    sim_step: int,
    frame_idx: int,
    throttle: float = 1.0,
) -> dict:
    """Build a single full-schema state snapshot."""
    sim.throttle_factor = throttle
    sensors = sim.get_reading(step=sim_step)
    anomaly = detector.predict(sensors)
    health  = engine.compute(sensors, anomaly)

    op_pt = engine.pump_curve.operating_point(sensors.get("flow_rate", 120.0))
    dev   = engine.pump_curve.deviation_from_curve(sensors)
    pump_curve = {
        "head_m":           op_pt["head_m"],
        "efficiency_pct":   op_pt["efficiency_pct"],
        "duty_status":      op_pt["duty_status"],
        "deviation_pct":    dev["deviation_pct"],
        "within_tolerance": dev["within_tolerance"],
    }

    return {
        "timestamp":       time.time(),
        "mode":            mode,
        "step":            frame_idx,
        "throttle_factor": throttle,
        "sensors":         sensors,
        "anomaly":         anomaly,
        "health":          health,
        "pump_curve":      pump_curve,
    }


def main() -> None:
    sim      = SensorSimulator()
    detector = PumpAnomalyDetector()
    engine   = HealthEngine()
    loader   = PumpDatasetLoader()

    # ── Ensure model exists ────────────────────────────────────────────
    if _MODEL_PATH.exists():
        detector.load(_MODEL_PATH)
    else:
        print("[replay_gen] pump_model.pkl not found — generating synthetic model …")
        X_normal, _ = loader.generate_synthetic()
        detector.train(X_normal, save_path=_MODEL_PATH)

    frames: list[dict] = []

    # ── Segment A: normal (frames 0–29) ───────────────────────────────
    sim.set_mode("normal")
    for i in range(30):
        frames.append(_make_frame(sim, detector, engine, "normal", i, i, throttle=1.0))

    # ── Segment B: bearing_wear (frames 30–49) ────────────────────────
    sim.set_mode("bearing_wear")
    for i in range(20):
        # accel step so severity builds visibly in 20 frames
        frames.append(_make_frame(sim, detector, engine, "bearing_wear",
                                  i * 5, 30 + i, throttle=1.0))

    # ── Segment C: cavitation (frames 50–69) ─────────────────────────
    sim.set_mode("cavitation")
    for i in range(20):
        frames.append(_make_frame(sim, detector, engine, "cavitation",
                                  i, 50 + i, throttle=0.75))

    # ── Segment D: dry_run (frames 70–89) ────────────────────────────
    sim.set_mode("dry_run")
    for i in range(20):
        frames.append(_make_frame(sim, detector, engine, "dry_run",
                                  i, 70 + i, throttle=0.1))

    # ── Segment E: recovery / normal (frames 90–119) ──────────────────
    sim.set_mode("normal")
    for i in range(30):
        frames.append(_make_frame(sim, detector, engine, "normal",
                                  i, 90 + i, throttle=min(1.0, 0.5 + i * 0.017)))

    with open(_OUT_PATH, "w") as fh:
        json.dump(frames, fh, indent=2)

    print(f"[replay_gen] ✓ replay.json written — {len(frames)} frames → {_OUT_PATH}")

    # Quick validation
    assert len(frames) == 120, f"Expected 120, got {len(frames)}"
    assert "npsh" in frames[0]["health"], "npsh missing from health"
    assert "pump_curve" in frames[0], "pump_curve missing"
    assert "shaft_rpm" in frames[0]["sensors"], "shaft_rpm missing from sensors"
    print("[replay_gen] ✓ All schema validation assertions passed.")
    print("All 8 upgrades complete. Backend ready.")


if __name__ == "__main__":
    main()
