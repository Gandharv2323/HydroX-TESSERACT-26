# Smart Pump Health Digital Twin

Organization: Trimiti Innovations  
Project: Hackathon P-09  
Version: 2.x (updated pipeline)

## Overview

HydroX is a real-time digital twin for industrial pump monitoring. It combines:

- Sensor simulation across multiple operating/fault modes
- Real-time anomaly detection and fault classification
- Health scoring (subsystem + overall)
- RUL estimation
- FastAPI REST + WebSocket streaming for dashboard/Unity clients

The repository now contains both:

- Legacy detector path (`ml_model.py`) used in the main state payload
- Advanced production-style pipeline (`pipeline/`) with sliding-window features, Isolation Forest, 5-class Random Forest, and LSTM-based RUL

---

## Current Architecture

### Runtime flow (`main.py`)

1. Generate or replay sensor readings (`sensor_sim.py` / `replay.json`)
2. Run legacy anomaly detector (`ml_model.py`)
3. Compute health, NPSH, and pump-curve metrics (`health_engine.py`)
4. Run advanced inference pipeline (`pipeline/inference_engine.py`)
5. Broadcast unified state over WebSocket and serve via REST endpoints

### Advanced ML pipeline (`pipeline/inference_engine.py`)

`sensor_dict -> validation -> sliding window (50x7) -> feature extraction (84 dims) -> IF anomaly -> RF fault class -> LSTM RUL -> decision JSON`

Feature vector rationale:

- The 84-dimensional vector is built from 7 sensors using time-domain and FFT-derived descriptors.
- It captures both stationary statistics and non-stationary transients, allowing detection of persistent drift and impulsive degradation.

Returned advanced fields include:

- `anomaly_score` (1 = worst, 0 = healthy in advanced pipeline output)
- `fault_class` (`normal`, `bearing_fault`, `cavitation`, `dry_run`, `misalignment`)
- `RUL` (hours)
- `confidence`
- `state` (`normal`, `anomalous`, `buffering`, `sensor_error`)

---

## Key Components

- `main.py`: FastAPI server, broadcast loop, API routes, dashboard serving
- `main_train.py`: One-command training entrypoint
- `main_infer.py`: Real-time inference/latency demo across scenarios
- `training/generate_data.py`: Balanced synthetic labeled data generation
- `training/train_all.py`: Trains IF, RF (5-class), and LSTM; saves artifacts
- `pipeline/features.py`: Time + frequency-domain feature extraction (84 features)
- `pipeline/fault_classifier.py`: 5-class Random Forest classifier
- `pipeline/rul_lstm.py`: LSTM-based RUL predictor
- `evaluation/report.py`: Evaluation report generation (IF/RF/LSTM)
- `models/`: Saved model artifacts

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train advanced models (creates `models/` artifacts)

```bash
python main_train.py
```

This generates:

- `models/isolation_forest.pkl`
- `models/fault_classifier.pkl`
- `models/rul_lstm.pt`

### 3) Start backend server

```bash
python main.py
```

API docs: `http://localhost:8000/docs`  
WebSocket: `ws://localhost:8000/ws`

### 4) Run standalone inference benchmark/demo (optional)

```bash
python main_infer.py
```

### 5) Generate evaluation report (optional)

```bash
python evaluation/report.py
```

---

## API Summary

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Dashboard UI (`dashboard.html`) |
| GET | `/ping` | Liveness + mode + connected client count |
| GET | `/state` | Full current state snapshot |
| GET | `/poll` | Same as `/state` |
| POST | `/scenario` | Set simulator mode (`normal`, `cavitation`, `bearing_wear`, `dry_run`) |
| POST | `/reset` | Reset simulator to `normal` |
| PATCH | `/config` | Update `broadcast_hz` and `noise_pct` at runtime |
| GET | `/unity` | Structured Unity-friendly payload |
| POST | `/throttle` | Set flow throttle factor (`0.0` to `1.5`) |
| GET | `/history` | Last 60 state snapshots |
| WS | `/ws` | Real-time full-state stream |
| POST | `/api/chat` | AI diagnostic response (Anthropic if key set, otherwise mock) |

---

## Training and Evaluation Notes

- Isolation Forest is trained unsupervised on normal-class features.
  It does not explicitly learn a manifold; it models normal operating regions implicitly through recursive random partitioning, where anomalous points require fewer splits to isolate.
- The anomaly score is based on ensemble path-length behavior and is normalized for runtime decisioning.
- Random Forest is a supervised 5-class classifier on extracted window features.
  Class probabilities are posterior estimates derived from tree-vote distributions across the ensemble.
- LSTM predicts RUL from sequential sensor windows.
  The architecture uses channel-wise sensor gating and temporal attention to emphasize diagnostically salient signals and timesteps.
- `evaluation/report.py` reports:
  - IF: ROC-AUC, average precision
  - RF: accuracy, macro-F1, confusion matrix
  - LSTM: MAE/RMSE (hours)

### Fault Signature Notes (precision)

- Cavitation is treated as broadband spectral excitation with elevated mid/high-frequency energy from bubble-collapse dynamics; it is not modeled as ideal white noise.
- Misalignment is characterized by strong 1x and 2x shaft-frequency harmonics, with possible higher-order harmonics as severity increases.

### Why Hybrid Instead of Monolithic

The system decomposes the problem into orthogonal tasks:

- Unsupervised anomaly detection (Isolation Forest)
- Supervised fault typing (Random Forest)
- Temporal prognostics (LSTM RUL)

This modular design improves interpretability, reduces dependence on fully labeled fault histories, and allows independent tuning and validation of each subsystem.

---

## Project Structure

```text
.
|- main.py
|- main_train.py
|- main_infer.py
|- sensor_sim.py
|- health_engine.py
|- ml_model.py
|- pipeline/
|  |- inference_engine.py
|  |- features.py
|  |- buffer.py
|  |- fault_classifier.py
|  '- rul_lstm.py
|- training/
|  |- generate_data.py
|  '- train_all.py
|- evaluation/
|  |- report.py
|  '- ablation_study.py
|- models/
|- dashboard.html
|- config.json
'- requirements.txt
```

---

## Deployment Notes

- Run `main_train.py` at least once before relying on advanced pipeline outputs.
- If `models/` artifacts are missing, `main.py` continues with fallback behavior where applicable.
- `REPLAY_MODE=1` can be used with `replay.json` for deterministic demonstrations.
