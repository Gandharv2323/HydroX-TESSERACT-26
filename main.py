"""
main.py — FastAPI + WebSocket server for Smart Pump Health Digital Twin.
Team Trimiti Innovations — Hackathon P-09.

Run:  python main.py
      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dataset_loader import PumpDatasetLoader
from health_engine import HealthEngine
from ml_model import PumpAnomalyDetector
from sensor_sim import SensorSimulator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR      = Path(__file__).parent
_CONFIG_PATH   = _BASE_DIR / "config.json"
_MODEL_PATH    = _BASE_DIR / "pump_model.pkl"
_CSV_PATH      = _BASE_DIR / "pump-sensor-data.csv"
_REPLAY_PATH   = _BASE_DIR / "replay.json"

with open(_CONFIG_PATH) as _fh:
    _cfg = json.load(_fh)

PORT         = int(_cfg["server_port"])
BROADCAST_HZ = float(_cfg["broadcast_hz"])
REPLAY_MODE  = os.getenv("REPLAY_MODE", "0") == "1"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
simulator = SensorSimulator()
detector  = PumpAnomalyDetector()
engine    = HealthEngine()
loader    = PumpDatasetLoader()

_current_state: dict[str, Any] = {}
_clients:       set[WebSocket]  = set()
_step_counter:  int             = 0

# Replay
_replay_frames: list[dict] = []
_replay_index:  int        = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state() -> dict:
    global _step_counter
    sensors  = simulator.get_reading(step=_step_counter)
    _step_counter += 1
    anomaly  = detector.predict(sensors)
    health   = engine.compute(sensors, anomaly)
    return {
        "timestamp": time.time(),
        "mode":      simulator.mode,
        "step":      _step_counter,
        "sensors":   sensors,
        "anomaly":   anomaly,
        "health":    health,
    }


def _next_replay_frame() -> dict:
    global _replay_index
    frame = _replay_frames[_replay_index % len(_replay_frames)]
    _replay_index += 1
    frame["timestamp"] = time.time()
    return frame


# ---------------------------------------------------------------------------
# Background broadcast loop
# ---------------------------------------------------------------------------

async def _broadcast_loop() -> None:
    while True:
        interval = 1.0 / BROADCAST_HZ  # re-read each tick so PATCH /config takes effect
        try:
            if REPLAY_MODE and _replay_frames:
                state = _next_replay_frame()
            else:
                state = _build_state()
            _current_state.update(state)

            dead: list[WebSocket] = []
            for ws in list(_clients):
                try:
                    await ws.send_json(state)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _clients.discard(ws)

        except Exception as exc:
            print(f"[broadcast_loop] {exc}")

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _replay_frames

    # ------ Model init -------------------------------------------------------
    if _MODEL_PATH.exists():
        detector.load(_MODEL_PATH)
    else:
        if _CSV_PATH.exists():
            X_normal, X_fault = loader.load_kaggle(str(_CSV_PATH))
            contamination = round(len(X_fault) / (len(X_normal) + len(X_fault)), 4)
            detector.train(
                X_normal,
                X_fault=X_fault,
                contamination=contamination,
                feature_names=loader.get_feature_names("kaggle"),
                save_path=_MODEL_PATH,
            )
            print("[startup] Model trained on real pump data (Kaggle CSV).")
        else:
            X_normal, X_fault = loader.generate_synthetic()
            detector.train(
                X_normal,
                X_fault=X_fault,
                feature_names=loader.get_feature_names("synthetic"),
                save_path=_MODEL_PATH,
            )
            print("[startup] Model trained on synthetic data.")

    # ------ Warm up state ----------------------------------------------------
    _current_state.update(_build_state())

    # ------ Replay preload ---------------------------------------------------
    if REPLAY_MODE:
        if _REPLAY_PATH.exists():
            with open(_REPLAY_PATH) as fh:
                _replay_frames = json.load(fh)
            print(f"[startup] Replay mode — loaded {len(_replay_frames)} frames.")
        else:
            print("[startup] REPLAY_MODE=1 but replay.json not found; run replay_gen.py first.")

    task = asyncio.create_task(_broadcast_loop())

    print("=" * 55)
    print("  Smart Pump Digital Twin backend ready")
    print(f"  WebSocket : ws://localhost:{PORT}/ws")
    print(f"  REST API  : http://localhost:{PORT}/docs")
    print("=" * 55)

    yield

    task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Pump Digital Twin — Trimiti Innovations P-09",
    description="Real-time pump health monitoring with ML-based anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    mode: str


class ConfigRequest(BaseModel):
    broadcast_hz: Optional[float] = None
    noise_pct:    Optional[float] = None


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Dashboard UI")
async def root() -> HTMLResponse:
    html_path = _BASE_DIR / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/ping", summary="Liveness probe")
async def ping() -> dict:
    return {
        "status":            "ok",
        "mode":              simulator.mode,
        "clients_connected": len(_clients),
    }


@app.get("/state", summary="Current full state snapshot")
async def get_state() -> dict:
    return _current_state


@app.get("/poll", summary="Unity HTTP fallback — identical to /state")
async def poll() -> dict:
    return _current_state


@app.post("/scenario", summary="Switch fault scenario")
async def set_scenario(body: ScenarioRequest) -> dict:
    global _step_counter
    valid = {"normal", "cavitation", "bearing_wear", "dry_run"}
    if body.mode not in valid:
        return {"success": False, "error": f"Invalid mode. Choose from {valid}"}
    simulator.set_mode(body.mode)
    _step_counter = 0
    return {"success": True, "mode": body.mode}


@app.post("/reset", summary="Reset to normal mode")
async def reset() -> dict:
    global _step_counter
    simulator.reset()
    _step_counter = 0
    return {"success": True, "mode": "normal"}


@app.patch("/config", summary="Update runtime config (Hz, noise_pct)")
async def update_config(body: ConfigRequest) -> dict:
    global BROADCAST_HZ
    changes: dict[str, float] = {}
    if body.broadcast_hz is not None:
        BROADCAST_HZ = float(max(0.1, min(20.0, body.broadcast_hz)))
        changes["broadcast_hz"] = BROADCAST_HZ
    if body.noise_pct is not None:
        npt = float(max(0.005, min(0.20, body.noise_pct)))
        simulator._noise_pct = npt
        changes["noise_pct"] = simulator._noise_pct
    return {"success": True, "applied": changes}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        # Send current state immediately on connect
        if _current_state:
            await ws.send_json(_current_state)
        # Keep alive — broadcast loop does the pushing
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
