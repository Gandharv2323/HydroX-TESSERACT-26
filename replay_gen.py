"""
replay_gen.py — Generates replay.json for offline demo (90 snapshots).

Segments:
  0–29  : normal mode       → health ~90-95 (all green)
  30–59 : bearing_wear mode → health ramps down from ~90 to ~30
  60–89 : normal again      → gradual recovery (simulate post-maintenance)

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

_BASE_DIR  = Path(__file__).parent
_MODEL_PATH = _BASE_DIR / "pump_model.pkl"
_OUT_PATH   = _BASE_DIR / "replay.json"


def main() -> None:
    sim      = SensorSimulator()
    detector = PumpAnomalyDetector()
    engine   = HealthEngine()
    loader   = PumpDatasetLoader()

    # Ensure model exists
    if _MODEL_PATH.exists():
        detector.load(_MODEL_PATH)
    else:
        print("[replay_gen] pump_model.pkl not found — generating synthetic model …")
        X_normal, _ = loader.generate_synthetic()
        detector.train(X_normal, save_path=_MODEL_PATH)

    frames: list[dict] = []

    # ------ Segment A: normal (0–29) ----------------------------------------
    sim.set_mode("normal")
    for step in range(30):
        sensors = sim.get_reading(step=step)
        anomaly = detector.predict(sensors)
        health  = engine.compute(sensors, anomaly)
        frames.append({
            "timestamp": time.time(),
            "mode":      "normal",
            "step":      step,
            "sensors":   sensors,
            "anomaly":   anomaly,
            "health":    health,
        })

    # ------ Segment B: bearing_wear (30–59) — severity ramps -----------------
    sim.set_mode("bearing_wear")
    for step in range(30):
        sensors = sim.get_reading(step=step * 4)   # accelerated ramp for demo
        anomaly = detector.predict(sensors)
        health  = engine.compute(sensors, anomaly)
        frames.append({
            "timestamp": time.time(),
            "mode":      "bearing_wear",
            "step":      30 + step,
            "sensors":   sensors,
            "anomaly":   anomaly,
            "health":    health,
        })

    # ------ Segment C: recovery / normal (60–89) -----------------------------
    sim.set_mode("normal")
    for step in range(30):
        sensors = sim.get_reading(step=step)
        anomaly = detector.predict(sensors)
        health  = engine.compute(sensors, anomaly)
        frames.append({
            "timestamp": time.time(),
            "mode":      "normal",
            "step":      60 + step,
            "sensors":   sensors,
            "anomaly":   anomaly,
            "health":    health,
        })

    with open(_OUT_PATH, "w") as fh:
        json.dump(frames, fh, indent=2)

    print(f"[replay_gen] ✓ replay.json written — {len(frames)} frames → {_OUT_PATH}")


if __name__ == "__main__":
    main()
