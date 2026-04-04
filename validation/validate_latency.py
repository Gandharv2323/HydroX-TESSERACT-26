"""
validation/validate_latency.py — Phase 4: Inference latency profiling.

Runs 500 back-to-back infer() calls and checks p95 < 350ms, max < 450ms.

Run: python validation/validate_latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.inference_engine import InferenceEngine
from sensor_sim import SensorSimulator

P95_TARGET_MS = 350.0
MAX_TARGET_MS = 450.0
N_CALLS       = 500


def main() -> None:
    print("=" * 62)
    print("  Phase 4 — Inference Latency Profiling")
    print(f"  N={N_CALLS} calls  |  p95 < {P95_TARGET_MS}ms  |  max < {MAX_TARGET_MS}ms")
    print("=" * 62)

    eng = InferenceEngine(models_dir=_ROOT / "models")
    sim = SensorSimulator()
    sim.set_mode("normal")

    print("\n  Warming up (10 calls) ...")
    for i in range(10):
        reading = sim.get_reading(step=i)
        eng.infer(reading)

    print(f"  Running {N_CALLS} timed calls ...")
    latencies_ms: list[float] = []

    for i in range(N_CALLS):
        reading = sim.get_reading(step=i + 10)
        t0  = time.perf_counter()
        eng.infer(reading)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat)

    lats = np.array(latencies_ms)
    avg  = float(lats.mean())
    p50  = float(np.percentile(lats, 50))
    p95  = float(np.percentile(lats, 95))
    p99  = float(np.percentile(lats, 99))
    mx   = float(lats.max())

    print(f"\n  avg   = {avg:.2f} ms")
    print(f"  p50   = {p50:.2f} ms")
    print(f"  p95   = {p95:.2f} ms  (target < {P95_TARGET_MS}ms)")
    print(f"  p99   = {p99:.2f} ms")
    print(f"  max   = {mx:.2f} ms  (target < {MAX_TARGET_MS}ms)")

    p95_ok  = p95 < P95_TARGET_MS
    max_ok  = mx  < MAX_TARGET_MS

    print(f"\n  p95 < {P95_TARGET_MS}ms : {'PASS OK' if p95_ok else 'FAIL X'}  ({p95:.1f}ms)")
    print(f"  max < {MAX_TARGET_MS}ms : {'PASS OK' if max_ok else 'FAIL X'}  ({mx:.1f}ms)")

    # Per-stage breakdown (from engine's own latency tracking)
    if hasattr(eng, "_latency_hist_ms") and len(eng._latency_hist_ms):
        eng_lats = np.array(list(eng._latency_hist_ms))
        print(f"\n  Engine internal latency (last {len(eng_lats)} calls):")
        print(f"    p50={np.percentile(eng_lats,50):.1f}ms  "
              f"p95={np.percentile(eng_lats,95):.1f}ms  "
              f"max={eng_lats.max():.1f}ms")

    all_pass = p95_ok and max_ok
    print("\n" + "=" * 62)
    print(f"  OVERALL: {'PASS OK' if all_pass else 'FAIL X — optimize bottleneck stages'}")
    print("=" * 62)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
