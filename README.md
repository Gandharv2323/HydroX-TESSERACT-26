# Smart Pump Health Digital Twin
**Trimiti Innovations — Hackathon P-09**

Real-time pump health monitoring system with ML-based anomaly detection, subsystem health scoring, and WebSocket streaming for Unity dashboards.

---

## Quick Start (5 Steps)

**Step 1** — Install dependencies
```bash
pip install -r requirements.txt
```

**Step 2** — Start the backend (model trains automatically on first run)
```bash
python main.py
```
> First run trains an IsolationForest on 1200 synthetic samples and saves `pump_model.pkl`.  
> If `pump-sensor-data.csv` is present, Kaggle data is used instead.

**Step 3** — Verify the API is live
```
http://localhost:8000/docs
```

**Step 4** — Connect your Unity WebSocket client
```
ws://localhost:8000/ws
```
> Broadcasts full state JSON at 2 Hz. CORS is fully open for Unity WebRequest.

**Step 5** — Switch fault modes during the demo
```bash
# Trigger cavitation fault
curl -X POST http://localhost:8000/scenario \
     -H "Content-Type: application/json" \
     -d '{"mode": "cavitation"}'

# Reset to normal
curl -X POST http://localhost:8000/reset
```

Available modes: `normal` | `cavitation` | `bearing_wear` | `dry_run`

---

## Optional

**Generate offline replay file** (90 frames: normal → bearing wear → recovery)
```bash
python replay_gen.py
```
Then start server in replay mode:
```bash
$env:REPLAY_MODE=1; python main.py      # PowerShell
REPLAY_MODE=1 python main.py            # bash
```

**Train on real Kaggle data**  
Place `pump-sensor-data.csv` in the `pump_twin/` folder and restart the server.  
Download from: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

---

## REST API Reference

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`         | Liveness probe + connected clients   |
| GET    | `/state`    | Current full state snapshot          |
| GET    | `/poll`     | Identical to `/state` (Unity HTTP)   |
| POST   | `/scenario` | Switch scenario mode                 |
| POST   | `/reset`    | Reset to normal mode                 |
| WS     | `/ws`       | WebSocket — 2 Hz broadcast stream    |

---

## Full State JSON Schema

```json
{
  "timestamp": 1711633200.123,
  "mode": "normal",
  "step": 42,
  "sensors": {
    "vibration_rms": 2.13,
    "vibration_peak": 4.81,
    "discharge_pressure": 4.19,
    "suction_pressure": 1.82,
    "flow_rate": 119.7,
    "motor_current": 18.4,
    "fluid_temp": 42.1
  },
  "anomaly": {
    "anomaly_score": 0.87,
    "is_anomaly": false,
    "failure_mode": "none",
    "confidence": 0.74,
    "recommended_action": "Normal operation — no action required"
  },
  "health": {
    "overall_health": 85.3,
    "subsystem_health": {
      "bearing_front": 88.1,
      "bearing_rear": 85.5,
      "seal": 84.2,
      "impeller": 86.7,
      "casing": 87.0
    },
    "status": "healthy",
    "rul_hours": 113.3
  }
}
```

---

## Tuning for the Demo

All threshold and weight values are in `config.json` — no code changes needed.

| Key | Effect |
|-----|--------|
| `broadcast_hz` | WebSocket update rate (default 2) |
| `health_thresholds.healthy` | Above this → green (default 70) |
| `health_thresholds.warning` | Below this → red / critical (default 40) |
| `subsystem_weights` | Relative importance of each subsystem |
| `isolation_forest.contamination` | Expected anomaly fraction in training data |

---

*Built for Hackathon P-09 by Trimiti Innovations.*
