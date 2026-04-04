# Phase 6 Targeted RUL Coverage Improvement Report

Date: 2026-04-05
Scope: RUL-only improvement cycle after Phase 6 lock-in validation.

## Objective
Improve empirical RUL coverage from the previously observed 82.4% toward the 88-92% target band without destabilizing IF, Hysteresis, Latency, or Fusion.

## Implemented Changes

### 1) Residual Gap Analysis
Added `validation/phase6_rul_gap_analysis.py` to quantify:
- coverage rate
- residual distribution
- 95th percentile residual tail
- out-of-interval count
- slope magnitude of outlier windows

### 2) Tail Window Augmentation
Added `training/phase6_rul_targeted_finetune.py` with:
- tail-focused oversampling
- synthetic perturbation for rapid-degradation trajectories
- separate calibration split after training block

### 3) RUL Predictor Enhancements
Updated `pipeline/rul_lstm.py` with:
- `fine_tune_tail(...)`
- `recalibrate_conformal(...)`
- affine calibration parameters
- conditional conformal bins
- conformal scale tuning

### 4) Evidence Artifacts
Generated:
- `out_rul_gap_analysis_before.txt`
- `out_rul_gap_analysis_after.txt`
- `out_rul_targeted_finetune.txt`
- `out_rul.txt`
- `out_if.txt`
- `out_hysteresis.txt`
- `out_latency.txt`
- `out_fusion.txt`
- `out_scenario.txt`

## Measured Outcomes

### Baseline (before targeted RUL cycle)
From `validation/phase6_rul_gap_analysis.py`:
- coverage: 0.82125
- outside count: 143
- residual mean: 56.80 h
- residual p95: 214.92 h

### After targeted RUL cycle
From `validation/phase6_rul_gap_analysis.py`:
- coverage: 0.86375
- outside count: 109
- residual mean: 55.53 h
- residual p95: 218.91 h

### Validator Result
From `validation/validate_rul_coverage.py`:
- empirical coverage: 86.8%
- average interval width: 218.8 h
- result: FAIL against the 88-92% target band, but improved versus baseline

## Integration Checks
The following subsystems remained stable and passed validation after the RUL-only changes:
- IF validation: PASS
- Hysteresis validation: PASS
- Latency validation: PASS
- Fusion validation: PASS
- Scenario runner: PASS

## Comparison Summary
- Coverage improved: +4.25 percentage points from baseline analysis
- Out-of-interval windows reduced: 143 -> 109
- Latency, IF, Hysteresis, Fusion remained stable
- Final RUL still short of the 88-92% acceptance band

## Residual Gap
The model still under-covers by roughly 1.2 to 3.2 percentage points depending on the validation path used.
The likely remaining issue is calibration width vs. tail miss-rate tradeoff, not a regression in the other subsystems.

## Archived Evidence
Latest bundle:
- `validation/archive/2026-04-05_01-22-20`

## Recommendation
Continue with a narrower RUL-only calibration pass:
- increase calibration-set size if possible
- tune scale multiplier directly on a held-out validation split
- preserve the already-passing IF/Hysteresis/Fusion/Latency paths
