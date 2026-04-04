# Unity Integration Guide

## Transport Options

- REST endpoint: `GET /unity`
- WebSocket endpoint: `ws://<host>:8000/ws`

## Payload Contract (recommended)

```json
{
  "anomaly_score": 0.42,
  "fault_class": "bearing_fault",
  "RUL": 132.5,
  "confidence": 0.88,
  "probabilities": {
    "normal": 0.02,
    "bearing_fault": 0.88,
    "cavitation": 0.05,
    "dry_run": 0.03,
    "misalignment": 0.02
  }
}
```

## Suggested Mapping

- `cavitation` -> bubble particle system intensity by anomaly score
- `misalignment` -> shaft rotation offset proportional to severity/confidence
- `bearing_fault` -> vibration amplitude modulation
- `dry_run` -> heat/emissive color ramp + warning indicator

## Notes

- Keep a client-side smoothing window (3-5 frames) for visual stability.
- Treat `state=buffering` as a warmup period before rendering alerts.
- If `sensor_error=true`, display degraded-data warning instead of fault animation.
