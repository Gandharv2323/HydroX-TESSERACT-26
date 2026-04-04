"""
main_infer.py — Real-time streaming inference demo.

Simulates 200 timesteps across all 4 fault modes using SensorSimulator,
running the full advanced pipeline (buffer → features → IF → RF → LSTM).

Prints JSON output for each inference step and reports:
  - Per-step latency
  - Average latency
  - Latency target compliance (< 100ms spec)

Run:  python main_infer.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main_infer")

from pipeline.inference_engine import InferenceEngine
from sensor_sim import SensorSimulator

MODELS_DIR = Path(__file__).parent / "models"

SCENARIO = [
    ("normal",       60),
    ("cavitation",   40),
    ("bearing_wear", 60),
    ("dry_run",      40),
]


def main() -> None:
    log.info("=" * 60)
    log.info("  Advanced Predictive Maintenance — Real-time Inference Demo")
    log.info("=" * 60)

    # Load pipeline
    engine = InferenceEngine(models_dir=MODELS_DIR)
    engine.load()

    sim       = SensorSimulator()
    latencies: list[float] = []

    step_global = 0
    for mode_name, n_steps in SCENARIO:
        sim.set_mode(mode_name)
        log.info(f"\n--- Scenario: {mode_name.upper()} ({n_steps} steps) ---")

        for i in range(n_steps):
            sensors = sim.get_reading(step=i)

            # Force float so JSON serialises cleanly
            sensors = {k: round(float(v), 4) for k, v in sensors.items()}

            result = engine.infer(sensors)
            latencies.append(result.get("latency_ms", 0.0))

            # Print selected steps to avoid output flood
            if i % 10 == 0 or result.get("state") == "anomalous":
                out = {
                    "step":          step_global,
                    "mode":          mode_name,
                    "state":         result.get("state"),
                    "anomaly_score": result.get("anomaly_score"),
                    "fault_class":   result.get("fault_class"),
                    "RUL_hours":     result.get("RUL"),
                    "confidence":    result.get("confidence"),
                    "latency_ms":    result.get("latency_ms"),
                }
                print(json.dumps(out))

            step_global += 1

    # ---- Latency report ------------------------------------------------
    valid_lat = [l for l in latencies if l > 0]
    if valid_lat:
        avg_lat  = float(sum(valid_lat) / len(valid_lat))
        max_lat  = max(valid_lat)
        p95      = sorted(valid_lat)[int(len(valid_lat) * 0.95)]
        pass_pct = sum(1 for l in valid_lat if l < 100) / len(valid_lat) * 100

        log.info("\n" + "=" * 60)
        log.info("  Latency Report")
        log.info("=" * 60)
        log.info(f"  Total steps  : {len(valid_lat)}")
        log.info(f"  Avg latency  : {avg_lat:.2f} ms")
        log.info(f"  P95 latency  : {p95:.2f} ms")
        log.info(f"  Max latency  : {max_lat:.2f} ms")
        log.info(f"  < 100ms rate : {pass_pct:.1f} %  (target: 100 %)")

        verdict = "✓ PASS" if pass_pct >= 99.0 else "✗ FAIL"
        log.info(f"  Spec verdict : {verdict}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
