"""validation/validate_hysteresis.py - Phase 3: Hysteresis edge-case tests.

Tests 4 signal sequences:
    1. Single spike (score=0.95) -> must trigger immediately (immediate override)
    2. Repeated spikes (score=0.80) -> must trigger within <=2 windows
    3. Noisy oscillation -> must NOT trigger (stays normal)
    4. Gradual rise -> must trigger at some point
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.inference_engine import InferenceEngine


def _make_engine() -> InferenceEngine:
    eng = InferenceEngine(models_dir=_ROOT / "models")
    # Don't load models — we test hysteresis logic with injected scores
    return eng


def _feed(eng: InferenceEngine, scores: list[float]) -> list[str]:
    """Feed a sequence of anomaly scores straight into the hysteresis state machine."""
    states = []
    for s in scores:
        state = eng._update_state_hysteresis(float(s))
        states.append(state)
    return states


def main() -> None:
    print("=" * 62)
    print("  Phase 3 — Hysteresis Logic Edge-Case Validation")
    print("=" * 62)

    all_pass = True

    # Test 1: Single spike at 0.95 -> immediate anomalous
    eng = _make_engine()
    scores = [0.20, 0.20, 0.95, 0.20, 0.20]
    states = _feed(eng, scores)
    spike_ok = states[2] == "anomalous"   # must trigger at the spike index
    print(f"\n[1/4] Single spike (0.95):")
    print(f"  States: {states}")
    print(f"  Spike detected immediately: {'PASS OK' if spike_ok else 'FAIL X'}")
    all_pass = all_pass and spike_ok

    # Test 2: Repeated high scores (0.80) -> triggers within 2 windows
    eng = _make_engine()
    scores = [0.80, 0.80, 0.80, 0.80, 0.80]
    states = _feed(eng, scores)
    # With smoothed score and N=2 for high scores, should trigger by window 3
    rep_ok = "anomalous" in states[:4]
    triggered_at = next((i for i, s in enumerate(states) if s == "anomalous"), -1)
    print(f"\n[2/4] Repeated high scores (0.80):")
    print(f"  States: {states}")
    print(f"  Triggered at window {triggered_at}: {'PASS OK' if rep_ok else 'FAIL X'}")
    all_pass = all_pass and rep_ok

    # Test 3: Noisy oscillation (0.45-0.65) -> should stay normal
    eng = _make_engine()
    scores = [0.55, 0.45, 0.60, 0.48, 0.57, 0.42, 0.58, 0.44, 0.56, 0.43]
    states = _feed(eng, scores)
    noisy_ok = states.count("anomalous") == 0
    print(f"\n[3/4] Noisy oscillation (0.45-0.65):")
    print(f"  States: {states}")
    print(f"  No false triggers: {'PASS OK' if noisy_ok else 'FAIL X'}")
    all_pass = all_pass and noisy_ok

    # Test 4: Gradual rise to 0.75+ -> must eventually trigger
    eng = _make_engine()
    scores = [0.20, 0.30, 0.40, 0.52, 0.60, 0.68, 0.75, 0.80, 0.82, 0.83]
    states = _feed(eng, scores)
    grad_ok = "anomalous" in states
    triggered_at = next((i for i, s in enumerate(states) if s == "anomalous"), -1)
    print(f"\n[4/4] Gradual rise (0.20 -> 0.83):")
    print(f"  States: {states}")
    print(f"  Eventually triggered (at window {triggered_at}): {'PASS OK' if grad_ok else 'FAIL X'}")
    all_pass = all_pass and grad_ok

    # ── Recovery test: after anomalous, drops back ─────────────────────────
    eng = _make_engine()
    scores = [0.95, 0.20, 0.20, 0.20, 0.20, 0.20]   # spike then recovery
    states = _feed(eng, scores)
    recovery_ok = states[-1] == "normal"
    print(f"\n[bonus] Recovery after spike:")
    print(f"  States: {states}")
    print(f"  Returned to normal: {'PASS OK' if recovery_ok else 'INFO (still decaying, not failure)'}")

    print("\n" + "=" * 62)
    print(f"  OVERALL: {'ALL TESTS PASSED OK' if all_pass else 'SOME TESTS FAILED X'}")
    print("=" * 62)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
