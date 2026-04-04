# HydroX System Hardening Log

Date: 2026-04-04
Scope: Foundation stabilization, data/signal integrity hardening, model reliability controls, and evaluation unification.

## Phase 0 - Stabilization

### 0.1 Evaluator-model mismatch (84 vs 227)

Implemented:
- Added unified representation builder in `pipeline/representation.py`.
- Refactored evaluator to use the same hybrid feature pipeline as training/inference in `evaluation/report.py`.
- Added strict assertion:
  - `assert X_hybrid.shape[-1] == HYBRID_DIM == 227`

Validation:
- `python evaluation/report.py` now runs without IF/RF dimension errors.
- Current report includes IF, RF, and RUL metrics under `synthetic` in `evaluation/reports/evaluation_report.json`.

### 0.2 Synthetic leakage mitigation

Implemented:
- Added `shuffle` control to synthetic generator (`training/generate_data.py`).
- Training uses non-shuffled temporal generation path.
- Added temporal split checks:
  - `validate_temporal_split(...)` in `evaluation/pipeline.py`
  - monotonic timeline guard in `training/train_all.py`.
- Reworked synthetic sequence ordering to interleave class windows in timeline (avoid class-block split artifacts).

Validation:
- Training evaluation stage now executes and writes `pipeline_metrics.json`.
- Temporal overlap checks pass during training run.

## Phase 1 - Data Integrity and Signal Quality

### 1.1 Missing-data mask integration

Implemented:
- Sequence models now consume sequence+mask channels (7 -> 14) in training:
  - `X_wins_seq = concat([X_wins, mask_wins], axis=2)` in `training/train_all.py`.
- Runtime tracks mask windows and feeds them into shared encoder path:
  - `self._mask_buffer` + `build_hybrid_feature_vector(..., mask_window=...)` in `pipeline/inference_engine.py`.

Validation:
- Full training and runtime smoke tests pass with 14-channel LSTM/shared inputs.

### 1.2 Max gap policy

Implemented:
- Added hard max-gap constants in loader:
  - `MAX_GAP_SECONDS = 5`, `MAX_GAP_STEPS = 10` in `data_pipeline/loader.py`.
- Bounded forward-fill to max gap and drop unresolved rows after fill.
- Runtime gap enforcement:
  - missing streak tracker in `pipeline/inference_engine.py`
  - violation raised when streak exceeds 10 steps.

Validation:
- Runtime now reports sensor-error violations for prolonged missing data.

### 1.3 FFT leakage fix

Implemented:
- Applied Hann windowing before FFT magnitude and phase extraction in `pipeline/features.py`.

Validation:
- Feature extraction path compiles and training/evaluation completes with updated FFT pipeline.

## Phase 2 - Model Reliability and Generalization

### 2.1 Unknown fault handling

Implemented:
- Open-set fallback in RF inference:
  - if `max(probability) < 0.60` then label `unknown`, class id `-1`.
- Implemented in `pipeline/fault_classifier.py`.

Validation:
- Runtime outputs include unknown-threshold metadata and can emit unknown class.

### 2.2 Temporal stability (hysteresis)

Implemented:
- Hysteresis and persistence logic in `pipeline/inference_engine.py`:
  - enter anomaly threshold: 0.60
  - exit anomaly threshold: 0.40
  - persistence: 3 windows

Validation:
- Runtime smoke output includes hysteresis telemetry and stable state transitions.

### 2.3 Isolation Forest config consistency

Implemented:
- Removed hardcoded contamination in training path.
- IF contamination now loaded from `config.json` and logged.

Validation:
- Training logs show `Using IF contamination from config: 0.0657`.

## Phase 3 - RUL Calibration and Alignment

### 3.1 RUL conformal calibration

Implemented:
- Added conformal residual quantile (`q90`) computation during LSTM fit in `pipeline/rul_lstm.py`.
- Persisted `conformal_q90` in model checkpoint.
- Inference uncertainty now returns:
  - `conformal_low`, `conformal_high`, `conformal_q90_h`.

Validation:
- Runtime output includes conformal interval fields.

### 3.2 Representation alignment for RUL path

Implemented:
- Shared representation stack already contains a joint RUL head.
- LSTM training/input path now includes missingness channels (14 input channels), aligning signal semantics across tasks.

Validation:
- `main_train.py` completes successfully with 14-channel sequence model.

## Phase 4 - Evaluation Integrity

### 4.1 Unified evaluation pipeline

Implemented:
- Evaluator now uses unified representation module (`pipeline/representation.py`).
- Evaluator enforces dimension 227.
- Metrics reported for IF/RF/RUL in one aligned run.

Validation:
- `evaluation/reports/evaluation_report.json` now contains all three tasks under `synthetic`.

### 4.2 Synthetic vs real separation

Implemented:
- Evaluator accepts optional real CSV via env var `HYDROX_REAL_EVAL_CSV` and writes separate `real` section when provided.
- Synthetic and real are reported independently.

## Phase 5 - Fault-to-Component Mapping

Implemented:
- Added component mapping in runtime output (`pipeline/inference_engine.py`):
  - `bearing_fault -> motor_bearing`
  - `cavitation -> pump_inlet_hydraulics`
  - `dry_run -> seal_and_fluid_path`
  - `misalignment -> shaft_coupling`
  - `unknown -> unknown_component`

## Phase 6 - Monitoring Hooks

Implemented in runtime telemetry (`pipeline/inference_engine.py`):
- contradiction counts and recent contradiction events
- missing-data frequency counters per sensor
- rolling latency sample count and p95 latency

## Verification Commands Executed

- `python -m py_compile ...` on all modified modules
- `python main_train.py` (multiple runs after hardening updates)
- `python evaluation/report.py`
- runtime smoke check via inline `python -c` execution for inference path

## Current Known Notes

1. Synthetic metrics remain very strong; this is expected and still optimistic vs field data.
2. Real-data evaluation path requires a provided CSV (`HYDROX_REAL_EVAL_CSV`) to populate `real` metrics.
3. Shared model load currently triggers a PyTorch future-warning about `torch.load` default `weights_only=False`; non-blocking but should be hardened later.
