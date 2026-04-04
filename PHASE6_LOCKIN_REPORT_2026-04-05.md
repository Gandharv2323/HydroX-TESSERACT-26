# Phase 6 Lock-In Report

Date: 2026-04-05
Scope: Stabilization and validation of existing HydroX pipeline without adding new feature modules.

## 1. Implemented Changes

### 1.1 RUL Prediction Enhancement
- Added tail-case augmentation in training for low-RUL windows in training/train_all.py.
- Retained finite-sample conformal quantile correction in pipeline/rul_lstm.py.
- Added conditional conformal bins by predicted degradation level in pipeline/rul_lstm.py.
- Added calibration-split affine bias correction (slope/intercept) in pipeline/rul_lstm.py.

### 1.2 Fusion Layer Rebalancing
- Extended fusion model inputs to include hysteresis signal branch:
  - X = [IF score, RF fault prob, hysteresis signal, PCA(latent), RUL]
- Kept latent compression at PCA(8).
- Persisted branch-level contribution metrics in models/fusion_meta.pkl.

### 1.3 Distribution Calibration
- Persisted IF domain profiles in isolation_forest.pkl:
  - synthetic profile always
  - real profile when real-domain windows are present
- Inference now prefers real-domain IF profile when available.

### 1.4 Verification/Runtime Robustness
- Added per-stage latency telemetry in pipeline/inference_engine.py.
- Fixed fusion call contract mismatch by passing RUL and hysteresis branch signals.
- Fixed scenario runner to load model artifacts and handle sensor-error rows robustly.
- Fixed Windows-console-safe output in validation/validate_hysteresis.py.

## 2. Final Validation Runs (Latest)

### 2.1 validate_if.py
- Result: PASS
- Key outputs:
  - Normal mean anomaly: 0.3833
  - Fault mean anomaly: 1.0000
  - Stress ordering: PASS (noise < drift and step >= drift)

### 2.2 validate_rul_coverage.py
- Result: FAIL
- Key outputs:
  - n = 800 windows
  - MAE = 56.19 h
  - Avg width = 191.6 h (PASS width target <= 220 h)
  - Empirical coverage = 82.4% (FAIL vs target band 88%-92%)

### 2.3 validate_hysteresis.py
- Result: PASS
- Single spike immediate trigger: PASS
- Repeated high spike trigger <=2 windows: PASS
- Noisy oscillation false-trigger suppression: PASS
- Gradual rise eventual trigger: PASS

### 2.4 validate_latency.py
- Result: PASS
- avg = 2.13 ms
- p95 = 3.31 ms
- max = 4.02 ms
- Meets strict real-time constraints and expected <=5 ms/window profile.

### 2.5 validate_fusion.py
- Result: PASS
- Dominant share = 0.3717 (< 0.60)
- Branch L1 shares:
  - IF: 0.1573
  - RF: 0.2466
  - Hysteresis: 0.0301
  - Latent(PCA): 0.3717
  - RUL: 0.1943

### 2.6 scenario_runner.py
- Result: PASS (all scenarios)
- normal_steady_state: PASS
- gradual_degradation: PASS
- sudden_failure: PASS
- sensor_dropout_recovery: PASS

## 3. Lock-In Status vs Requested Outcomes

- IF calibration and stress behavior: PASS
- Hysteresis behavior correctness: PASS
- Latency objective (<=5 ms/window): PASS
- Fusion balance with PCA and measurable branch influence: PASS
- RUL conformal coverage >=90%: NOT MET (82.4%)

## 4. Known Limitations

- The RUL model still under-covers on held-out synthetic validation despite:
  - finite-sample quantile correction
  - conditional conformal bins
  - bias correction
- Current intervals are sharp enough (width target met) but not wide enough for 90% empirical coverage.

## 5. Evidence Archive

Latest archived evidence bundle:
- validation/archive/2026-04-05_00-20-59

Contained artifacts:
- out_if.txt
- out_rul.txt
- out_hysteresis.txt
- out_latency.txt
- out_fusion.txt
- out_scenario.txt
- threshold.json
- pipeline_metrics.json
- scenario_results.json

## 6. Recommendation for Final Lock Closure

To close the remaining gap, run a targeted RUL-only calibration cycle:
1. Increase conformal calibration set size (hold out >=25% train windows).
2. Use conservative conformal quantile in low-RUL bin only.
3. Re-evaluate until empirical coverage reaches 88%-92% while keeping width <=220h.

No additional architecture modules are required; only calibration retuning is pending.
