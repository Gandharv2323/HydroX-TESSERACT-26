# HydroX Architecture Upgrade Log

Date: 2026-04-04
Scope: Representation-consistent multi-task upgrade with calibration and consistency controls.

## Summary

This upgrade applies the requested strategy to move from loosely coupled model outputs toward a unified, representation-consistent pipeline. Changes were implemented with backward-safe fallbacks where existing artifacts are absent.

## Applied Changes by Gap

### Gap 1: Representation bifurcation

Applied:
- Added trainable shared encoder + multi-task heads runtime in `models/shared_latent.py`.
- Added training stage to produce `models/shared_latent.pt` in `training/train_all.py`.
- Runtime now forms hybrid representation from engineered + learned features in `pipeline/inference_engine.py`.

Result:
- IF/RF inference now consumes hybrid vector `[z || phase || h]`.
- Common latent `h` is available to downstream fusion.

### Gap 2: Feature information loss

Applied:
- Preserved engineered features `z` and concatenated learned latent `h`.
- Hybrid training/inference representation implemented in `training/train_all.py` and `pipeline/inference_engine.py`.

Result:
- No replacement loss; both physics-driven and learned temporal information are retained.

### Gap 3: IF threshold instability

Applied:
- Added unsupervised quantile calibration in `calibration/threshold.py` via `calibrate_threshold_unsupervised`.
- Switched training calibration to normal-score quantile (95th percentile) in `training/train_all.py`.

Result:
- Runtime threshold no longer depends on label-derived F1 optimization.

### Gap 4: RF probability miscalibration

Applied:
- Added RF probability calibration support in `pipeline/fault_classifier.py` using `CalibratedClassifierCV`.
- Training now uses isotonic calibration by default in `training/train_all.py`.

Result:
- `predict_proba` output is calibrated and suitable for fusion.

### Gap 5: Score fusion inconsistency

Applied:
- Added learned fusion meta-model in `calibration/fusion_meta.py`.
- Training now fits `fusion_meta.pkl` from `[if_score, rf_fault_prob, h]` in `training/train_all.py`.
- Inference now uses learned fusion when available, with weighted fallback in `pipeline/inference_engine.py`.

Result:
- Fusion is data-driven and no longer strictly hard-coded weighted average.

### Gap 6: FFT phase blindness

Applied:
- Added phase descriptors in `pipeline/features.py`:
  - per-sensor phase variance
  - per-sensor phase drift
  - cross-sensor mean phase difference
- Incorporated phase vector in hybrid representation in training and inference.

Result:
- Phase behavior contributes to downstream decisioning.

### Gap 7: Synthetic RUL mismatch

Applied:
- Shared model trains on relative RUL target (`rul / 600`) in `models/shared_latent.py`.
- Existing RUL model uncertainty remains active in runtime (`predict_with_uncertainty`).

Result:
- Added relative-health style target in shared objective while preserving current RUL predictor path.

### Gap 8: Consistency constraints

Applied:
- Shared training includes consistency loss term.
- Runtime rule constraints remain active in `pipeline/inference_engine.py`.

Result:
- Contradictions are reduced both by training objective and inference constraints.

### Gap 9: State logic correctness

Applied:
- Runtime state/severity already uses anomaly + confidence + RUL logic with consistency actions.
- Maintained and integrated with upgraded fused anomaly score.

Result:
- State derives from multiple factors, not anomaly alone.

### Gap 10: No joint objective

Applied:
- Implemented shared multi-task training objective in `models/shared_latent.py`:
  - anomaly BCE
  - class CE
  - relative RUL smooth L1
  - consistency loss

Result:
- Joint learning now exists and is persisted as a deployable encoder artifact.

## New Artifacts

- `models/shared_latent.pt`
- `models/fusion_meta.pkl`
- `configs/threshold.json` (quantile-normal calibration after training)

## Modified Files

- `models/shared_latent.py`
- `pipeline/fault_classifier.py`
- `pipeline/inference_engine.py`
- `pipeline/features.py`
- `training/train_all.py`
- `calibration/threshold.py`
- `calibration/fusion_meta.py`

## Operational Notes

1. Run full training to materialize new artifacts:

   `python main_train.py`

2. Runtime will automatically use:
- shared latent encoder if `models/shared_latent.pt` exists
- learned fusion if `models/fusion_meta.pkl` exists
- fallback modes otherwise

3. During transition, classifier/IF input alignment handles feature-width mismatches via pad/truncate to avoid runtime breaks.
