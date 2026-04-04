"""
evaluation/scenario_runner.py — Phase 6: Scenario-based system validation.

Drives SensorSimulator through 4 scenarios, feeds each output through
InferenceEngine, logs state transitions, anomaly scores, and RUL per step.

Scenarios:
  1. Normal steady-state
  2. Gradual degradation (bearing_wear)
  3. Sudden failure (dry_run → instant high score)
  4. Sensor dropout recovery

Output: evaluation/reports/scenario_results.json

Run: python evaluation/scenario_runner.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.inference_engine import InferenceEngine
from sensor_sim import SensorSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scenario_runner")

REPORTS_DIR = _ROOT / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_SCENARIOS = [
    {
        "name":  "normal_steady_state",
        "steps": 80,
        "modes": [("normal", 0)],
        "expect_final_state": "normal",
    },
    {
        "name":  "gradual_degradation",
        "steps": 100,
        "modes": [("normal", 0), ("bearing_wear", 30)],
        "expect_final_state": "anomalous",
    },
    {
        "name":  "sudden_failure",
        "steps": 80,
        "modes": [("normal", 0), ("dry_run", 60)],
        "expect_state_after_failure": "anomalous",
    },
    {
        "name":  "sensor_dropout_recovery",
        "steps": 80,
        "modes": [("normal", 0)],
        "inject_dropout": True,
        "dropout_start": 30,
        "dropout_end":   50,
        "expect_final_state": "normal",
    },
]


def _run_scenario(eng: InferenceEngine, scenario: dict) -> dict:
    sim  = SensorSimulator()
    log_entries: list[dict] = []

    mode_schedule = sorted(scenario.get("modes", []), key=lambda x: x[1])
    inject_dropout = scenario.get("inject_dropout", False)
    dropout_start  = scenario.get("dropout_start", -1)
    dropout_end    = scenario.get("dropout_end",   -1)

    eng._state       = "normal"
    eng._anom_streak = 0
    eng._normal_streak = 0
    eng._state_score   = 0.0

    current_mode = mode_schedule[0][0] if mode_schedule else "normal"
    sim.set_mode(current_mode)

    for step in range(scenario["steps"]):
        # Switch mode on schedule
        for mode, start_step in mode_schedule:
            if step == start_step:
                sim.set_mode(mode)
                current_mode = mode

        reading = sim.get_reading(step=step)

        # Inject sensor dropout
        if inject_dropout and dropout_start <= step < dropout_end:
            reading["vibration_rms"]      = None
            reading["discharge_pressure"] = None

        result = eng.infer(reading)

        entry = {
            "step":          step,
            "mode":          current_mode,
            "anomaly_score": round(float(result.get("anomaly_score", 0.0)), 4),
            "health_index":  round(float(result.get("health_index",  1.0)), 4),
            "state":         result.get("state", "unknown"),
            "rul_hours":     round(float(result.get("rul_hours", 0.0)), 1),
            "fault_class":   result.get("fault_class", "normal"),
        }
        log_entries.append(entry)

    final_state     = log_entries[-1]["state"]
    state_changes   = sum(
        1 for i in range(1, len(log_entries))
        if log_entries[i]["state"] != log_entries[i - 1]["state"]
    )
    anomaly_scores  = [e["anomaly_score"] for e in log_entries]
    mean_score      = sum(anomaly_scores) / len(anomaly_scores)

    # Expectation checks
    expect_final = scenario.get("expect_final_state")
    passed = True
    if expect_final:
        passed = passed and (final_state == expect_final)

    expect_after_fail = scenario.get("expect_state_after_failure")
    if expect_after_fail:
        # Check state at least once after failure point
        fail_step = max(m[1] for m in scenario.get("modes", [(None, 0)]))
        post_states = [e["state"] for e in log_entries if e["step"] > fail_step + 10]
        passed = passed and (expect_after_fail in post_states)

    return {
        "scenario":       scenario["name"],
        "steps":          scenario["steps"],
        "final_state":    final_state,
        "state_changes":  state_changes,
        "mean_anomaly_score": round(mean_score, 4),
        "max_anomaly_score":  round(max(anomaly_scores), 4),
        "expected_final":     expect_final,
        "passed":             passed,
        "log":                log_entries,
    }


def run_all_scenarios() -> dict:
    log.info("=" * 60)
    log.info("  Phase 6 — Scenario Runner")
    log.info("=" * 60)

    eng = InferenceEngine(models_dir=_ROOT / "models")

    all_results = []
    all_pass    = True

    for scenario in _SCENARIOS:
        log.info(f"\n  Running: {scenario['name']} ({scenario['steps']} steps) ...")
        result = _run_scenario(eng, scenario)
        all_results.append(result)

        status = "PASS OK" if result["passed"] else "FAIL X"
        log.info(
            f"  {status}  final_state={result['final_state']}  "
            f"changes={result['state_changes']}  "
            f"mean_score={result['mean_anomaly_score']:.4f}  "
            f"max_score={result['max_anomaly_score']:.4f}"
        )
        if not result["passed"]:
            all_pass = False

    summary = {
        "total_scenarios": len(all_results),
        "passed":    sum(1 for r in all_results if r["passed"]),
        "failed":    sum(1 for r in all_results if not r["passed"]),
        "all_pass":  all_pass,
        "scenarios": all_results,
    }

    out_path = REPORTS_DIR / "scenario_results.json"

    # Save without log blowing up the file (optional: save logs separately)
    summary_clean = {
        k: v for k, v in summary.items() if k != "scenarios"
    }
    summary_clean["scenarios"] = [
        {k: v for k, v in r.items() if k != "log"}
        for r in all_results
    ]
    with open(out_path, "w") as fh:
        json.dump(summary_clean, fh, indent=2)
    log.info(f"\n  Report saved → {out_path}")

    log.info("\n" + "=" * 60)
    log.info(f"  OVERALL: {'ALL SCENARIOS PASSED OK' if all_pass else 'SOME FAILED X'}")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    result = run_all_scenarios()
    sys.exit(0 if result["all_pass"] else 1)
