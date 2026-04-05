"""
main.py  FastAPI + WebSocket server for Smart Pump Health Digital Twin.
Team Trimiti Innovations  Hackathon P-09.

Run:  python main.py
      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Suppress uvicorn's "Invalid HTTP request received" — caused by browser
# HTTP/2 probes / partial WebSocket upgrades; these are harmless.
import logging as _logging

class _InvalidHTTPFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "Invalid HTTP request received" not in record.getMessage()

_logging.getLogger("uvicorn.error").addFilter(_InvalidHTTPFilter())
_logging.getLogger("uvicorn.access").addFilter(_InvalidHTTPFilter())

from dataset_loader import PumpDatasetLoader
from health_engine import HealthEngine
from ml_model import PumpAnomalyDetector
from model_bundle import extract_bundle
from sensor_sim import SensorSimulator
from pipeline.inference_engine import InferenceEngine
from evaluation.pipeline import time_aware_split_indices

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR      = Path(__file__).parent
_CONFIG_PATH   = _BASE_DIR / "config.json"
_MODEL_PATH    = _BASE_DIR / "pump_model.pkl"
_CSV_PATH      = _BASE_DIR / "pump-sensor-data.csv"
_REPLAY_PATH   = _BASE_DIR / "replay.json"
_MODEL_BUNDLE_PATH = Path(os.getenv("HYDROX_MODEL_BUNDLE", _BASE_DIR / "hydrox_model_bundle.zip"))

with open(_CONFIG_PATH) as _fh:
    _cfg = json.load(_fh)

PORT         = int(_cfg["server_port"])
BROADCAST_HZ = float(_cfg["broadcast_hz"])
REPLAY_MODE  = os.getenv("REPLAY_MODE", "0") == "1"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
simulator        = SensorSimulator()
detector         = PumpAnomalyDetector()
engine           = HealthEngine()
loader           = PumpDatasetLoader()
_adv_engine      = InferenceEngine(models_dir=_BASE_DIR / "models")

_current_state: dict[str, Any] = {}
_clients:       set[WebSocket]  = set()
_step_counter:  int             = 0
_history:       deque[dict]     = deque(maxlen=60)

# Replay
_replay_frames: list[dict] = []
_replay_index:  int        = 0

# Data-source mode: "sim" (live simulator) or "test" (held-out test split)
_data_source_mode: str = "sim"


class TestDatasetPlayer:
    """
    Replays the held-out test split (last 15% of synthetic data) frame-by-frame,
    extracting raw sensor readings so they flow through the full ML pipeline
    (IF → RF → LSTM) exactly like live simulator data.
    Loop back to the start when exhausted.
    """

    SENSOR_KEYS = [
        "vibration_rms", "vibration_peak", "discharge_pressure",
        "suction_pressure", "flow_rate", "motor_current", "fluid_temp",
    ]

    def __init__(self) -> None:
        self._frames: list[dict] = []
        self._labels: list[int]  = []
        self._rul:    list[float] = []
        self._idx:    int         = 0
        self._loaded: bool        = False
        self._total:  int         = 0

    def load(self) -> int:
        """Generate synthetic data and extract the test split windows."""
        import numpy as np
        from training.generate_data import generate
        from pipeline.buffer import SENSORS as _BUF_SENSORS

        try:
            X_wins, y_cls, y_rul, X_feats, t_idx = generate(
                n_per_class=300, shuffle=False, return_time_index=True
            )
        except TypeError:
            # Older generate() without return_time_index
            X_wins, y_cls, y_rul, X_feats = generate(n_per_class=300, shuffle=False)

        n = len(X_wins)
        _, _, test_idx = time_aware_split_indices(n)

        LABEL_MAP = {0: "normal", 1: "bearing_fault", 2: "cavitation",
                     3: "dry_run", 4: "misalignment"}

        # X_wins shape: (N, WINDOW_SIZE, 7)  — last row of each window = latest reading
        for i in test_idx:
            row = X_wins[i, -1, :].tolist()           # latest timestep
            sensor_dict = dict(zip(_BUF_SENSORS, row))
            sensor_dict["shaft_rpm"]      = float(sensor_dict.get("shaft_rpm", 1440.0))
            sensor_dict["vibration_peak"] = float(sensor_dict.get("vibration_peak",
                                                    sensor_dict.get("vibration_rms", 2.1) * 2.3))
            self._frames.append(sensor_dict)
            self._labels.append(int(y_cls[i]))
            self._rul.append(float(y_rul[i]))

        self._total  = len(self._frames)
        self._loaded = True
        self._idx    = 0
        return self._total

    def next_sensors(self) -> tuple[dict, int, float]:
        """Return (sensor_dict, true_label_id, true_rul)."""
        if not self._loaded or self._total == 0:
            raise RuntimeError("TestDatasetPlayer not loaded — call load() first")
        sensors = dict(self._frames[self._idx])
        label   = self._labels[self._idx]
        rul     = self._rul[self._idx]
        self._idx = (self._idx + 1) % self._total
        return sensors, label, rul

    @property
    def progress(self) -> dict:
        return {
            "current": self._idx,
            "total":   self._total,
            "pct":     round(100.0 * self._idx / max(1, self._total), 1),
        }


_test_player = TestDatasetPlayer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state(test_override: dict | None = None) -> dict:
    global _step_counter
    if test_override is not None:
        sensors = test_override
    else:
        sensors = simulator.get_reading(step=_step_counter)
    _step_counter += 1
    anomaly  = detector.predict(sensors)
    health   = engine.compute(sensors, anomaly)

    # Advanced ML pipeline (buffer → features → IF → 5-class RF → LSTM)
    adv = _adv_engine.infer(sensors)

    # UPGRADE 2 — pump curve
    op_pt   = engine.pump_curve.operating_point(sensors.get("flow_rate", 120.0))
    dev     = engine.pump_curve.deviation_from_curve(sensors)
    pump_curve = {
        "head_m":           op_pt["head_m"],
        "efficiency_pct":   op_pt["efficiency_pct"],
        "duty_status":      op_pt["duty_status"],
        "deviation_pct":    dev["deviation_pct"],
        "within_tolerance": dev["within_tolerance"],
    }

    mode_val = simulator.mode if test_override is None else "test_dataset"
    return {
        "timestamp":        time.time(),
        "mode":             mode_val,
        "data_source":      _data_source_mode,
        "step":             _step_counter,
        "throttle_factor":  simulator.throttle_factor,
        "sensors":          sensors,
        "anomaly":          anomaly,
        "health":           health,
        "pump_curve":       pump_curve,
        "advanced_ml":      adv,
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
        interval = 1.0 / BROADCAST_HZ
        try:
            if _data_source_mode == "test" and _test_player._loaded:
                sensors, true_label, true_rul = _test_player.next_sensors()
                state = _build_state(test_override=sensors)
                # Annotate with ground-truth for comparison in dashboard
                state["ground_truth"] = {
                    "fault_class_id": true_label,
                    "fault_label":    ["normal", "bearing_fault", "cavitation",
                                       "dry_run", "misalignment"][true_label],
                    "rul_hours":      round(true_rul, 1),
                    "progress":       _test_player.progress,
                }
            elif REPLAY_MODE and _replay_frames:
                state = _next_replay_frame()
            else:
                state = _build_state()

            _current_state.update(state)
            _history.append(dict(state))

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

    # ------ Advanced ML pipeline load ----------------------------------------
    _models_dir = _BASE_DIR / "models"
    _required_adv = [
        _models_dir / "isolation_forest.pkl",
        _models_dir / "fault_classifier.pkl",
        _models_dir / "rul_lstm.pt",
        _models_dir / "shared_latent.pt",
        _models_dir / "fusion_meta.pkl",
    ]

    if (not _models_dir.exists() or any(not p.exists() for p in _required_adv)) and _MODEL_BUNDLE_PATH.exists():
        try:
            extracted = extract_bundle(_MODEL_BUNDLE_PATH, _BASE_DIR, force=False)
            print(f"[startup] Extracted {len(extracted)} files from {_MODEL_BUNDLE_PATH.name}.")
        except Exception as exc:
            print(f"[startup] Model bundle extract failed: {exc}")

    if _models_dir.exists():
        try:
            _adv_engine.load()
            print("[startup] Advanced ML pipeline loaded (IF + RF-5class + LSTM).")
        except Exception as exc:
            print(f"[startup] Advanced ML pipeline not loaded: {exc}")
    else:
        print("[startup] models/ dir not found — run main_train.py first.")

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
            print(f"[startup] Replay mode  loaded {len(_replay_frames)} frames.")
        else:
            print("[startup] REPLAY_MODE=1 but replay.json not found; run replay_gen.py first.")

    # ------ Test dataset player preload (lazy, fast) -------------------------
    try:
        n_test = _test_player.load()
        print(f"[startup] Test dataset player ready: {n_test} test frames.")
    except Exception as exc:
        print(f"[startup] Test dataset player unavailable: {exc}")

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
    title="Smart Pump Digital Twin  Trimiti Innovations P-09",
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


class ThrottleRequest(BaseModel):      # UPGRADE 4
    flow_pct: float


class ChatRequest(BaseModel):
    prompt: str
    context: dict = {}


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


@app.get("/dataflow", response_class=HTMLResponse, summary="Real-time telemetry pipeline visualization")
async def dataflow_page() -> HTMLResponse:
    html_path = _BASE_DIR / "dataflow.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/ping", summary="Liveness probe")
async def ping() -> dict:
    return {
        "status":            "ok",
        "mode":              simulator.mode,
        "data_source":       _data_source_mode,
        "test_progress":     _test_player.progress if _test_player._loaded else None,
        "clients_connected": len(_clients),
    }


@app.get("/state", summary="Current full state snapshot")
async def get_state() -> dict:
    return _current_state


@app.get("/poll", summary="Unity HTTP fallback  identical to /state")
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


class DataSourceRequest(BaseModel):
    source: str  # "sim" | "test"


@app.post("/data-source", summary="Switch between live simulator and test dataset replay")
async def set_data_source(body: DataSourceRequest) -> dict:
    global _data_source_mode, _step_counter
    if body.source not in {"sim", "test"}:
        return {"success": False, "error": "source must be 'sim' or 'test'"}
    _data_source_mode = body.source
    if body.source == "test":
        # Reset test player to start of test split
        _test_player._idx = 0
        if not _test_player._loaded:
            try:
                n = _test_player.load()
                return {"success": True, "source": "test",
                        "test_frames": n, "message": "Test dataset loaded and ready"}
            except Exception as exc:
                _data_source_mode = "sim"
                return {"success": False, "error": str(exc)}
        return {"success": True, "source": "test",
                "test_frames": _test_player._total,
                "message": f"Replaying {_test_player._total} test frames"}
    else:
        _step_counter = 0
        return {"success": True, "source": "sim", "message": "Switched to live simulator"}


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

@app.get("/unity", summary="Structured output for Unity Digital Twin")
async def unity_poll() -> dict:
    """
    Clean structured JSON designed for Unity WebSocket / HTTP consumers.
    Maps directly to C# structs in the Unity side.
    Format:
    {
        fault:        string  (fault label from 5-class RF)
        severity:     string  (normal | warning | critical)
        anomaly_score: float  (0=healthy, 1=severe)
        RUL:          float   (hours to failure, LSTM-predicted)
        confidence:   float   (0-1)
        rpm:          float
        flow:         float
        vibration:    float
        temp:         float
        probabilities: dict
    }
    """
    s   = _current_state
    adv = s.get("advanced_ml", {})
    sensors = s.get("sensors", {})
    health  = s.get("health", {})
    return {
        "fault":         adv.get("fault_class", "unknown"),
        "severity":      health.get("status", "unknown"),
        "anomaly_score": adv.get("anomaly_score") or s.get("anomaly", {}).get("anomaly_score", 0.0),
        "RUL":           adv.get("RUL") or health.get("rul_hours", 0.0),
        "confidence":    adv.get("confidence", 0.0),
        "probabilities": adv.get("probabilities", {}),
        "rpm":           sensors.get("shaft_rpm", 0.0),
        "flow":          sensors.get("flow_rate", 0.0),
        "vibration":     sensors.get("vibration_rms", 0.0),
        "temp":          sensors.get("fluid_temp", 0.0),
        "npsh_status":   health.get("npsh", {}).get("status", "unknown"),
        "step":          s.get("step", 0),
        "timestamp":     s.get("timestamp", 0.0),
    }



@app.post("/throttle", summary="Set flow throttle factor (0.0–1.5, 1.0 = rated)")
async def set_throttle(body: ThrottleRequest) -> dict:
    """UPGRADE 4 — multiplies flow_rate baseline by flow_pct."""
    factor = float(max(0.0, min(1.5, body.flow_pct)))
    simulator.throttle_factor = factor
    expected_flow = round(
        simulator._baselines.get("flow_rate", 120.0) * factor, 1
    )
    return {"success": True, "throttle_factor": factor, "expected_flow": expected_flow}


@app.get("/history", summary="Last 60 state snapshots")
async def get_history() -> list:
    """UPGRADE 7 — returns rolling buffer of last 60 broadcast states."""
    return list(_history)


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
        # Keep alive  broadcast loop does the pushing
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


import httpx

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    # Simple Mock fallback if no key provided
    if not anthropic_key:
        prompt_lower = request.prompt.lower()
        mode = request.context.get("mode", "unknown")
        vib = request.context.get("vibration", 0.0)
        
        reply = f"I am operating in mock mode (no ANTHROPIC_API_KEY). Based on the current mode '{mode}' and vibration of {vib:.2f} mm/s, "
        if 'cavitation' in prompt_lower or mode == 'cavitation':
            reply += "cavitation is highly probable due to low suction pressure. I recommend reducing pump speed or inspecting the inlet valve."
        elif 'wear' in prompt_lower or mode == 'bearing_wear':
            reply += "bearing wear is detected due to high peak vibration. Schedule maintenance to replace the bearings soon."
        elif 'dry' in prompt_lower or mode == 'dry_run':
            reply += "dry run detected! Shut down the pump immediately to prevent seal damage."
        elif 'hello' in prompt_lower or 'hi' in prompt_lower:
            reply += "Hello! I'm the HydroX AI assistant. Ask me to analyze the pump health!"
        elif 'health' in prompt_lower or 'diagnose' in prompt_lower:
            reply += "The pump looks stable but monitor the vibration trends."
        else:
            reply += "please provide more specific queries about faults or performance."
            
        return {"reply": reply}

    # Real Claude API integration
    try:
        async with httpx.AsyncClient() as client:
            system_prompt = "You are the AI assistant for HydroX, an industrial pump digital twin. Analyze the provided prompt and context, and give a short, professional, highly actionable recommendation (2-3 sentences max)."
            user_content = f"Context: {request.context}\n\nUser Prompt: {request.prompt}"
            
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 200,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}]
                },
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["content"][0]["text"]
            return {"reply": reply}
    except Exception as e:
        return {"reply": f"API Error: {str(e)}"}
