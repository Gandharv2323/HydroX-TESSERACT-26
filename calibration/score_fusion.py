from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def load_fusion_config(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "if": {"method": "identity"},
            "rf": {"method": "identity"},
            "fusion": {"w_if": 0.6, "w_rf": 0.4},
        }
    return json.loads(p.read_text(encoding="utf-8"))


def calibrate_if_score(raw_if_anomaly: float, cfg: dict[str, Any]) -> float:
    section = cfg.get("if", {})
    method = str(section.get("method", "identity")).lower()
    x = float(np.clip(raw_if_anomaly, 0.0, 1.0))

    if method == "platt":
        a = float(section.get("a", 1.0))
        b = float(section.get("b", 0.0))
        return float(np.clip(_sigmoid(a * x + b), 0.0, 1.0))

    if method == "minmax":
        lo = float(section.get("lo", 0.0))
        hi = float(section.get("hi", 1.0))
        return float(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    return x


def calibrate_rf_probability(raw_fault_prob: float, cfg: dict[str, Any]) -> float:
    section = cfg.get("rf", {})
    method = str(section.get("method", "identity")).lower()
    x = float(np.clip(raw_fault_prob, 0.0, 1.0))

    if method == "platt":
        a = float(section.get("a", 1.0))
        b = float(section.get("b", 0.0))
        return float(np.clip(_sigmoid(a * x + b), 0.0, 1.0))

    if method == "temperature":
        t = max(1e-6, float(section.get("t", 1.0)))
        logit = np.log(x + 1e-12) - np.log(1.0 - x + 1e-12)
        return float(np.clip(_sigmoid(float(logit / t)), 0.0, 1.0))

    return x


def fuse_scores(cal_if: float, cal_rf: float, cfg: dict[str, Any]) -> float:
    fusion = cfg.get("fusion", {})
    w_if = float(fusion.get("w_if", 0.6))
    w_rf = float(fusion.get("w_rf", 0.4))
    s = w_if + w_rf
    if s <= 0:
        w_if, w_rf = 0.6, 0.4
        s = 1.0
    w_if, w_rf = w_if / s, w_rf / s
    return float(np.clip(w_if * cal_if + w_rf * cal_rf, 0.0, 1.0))
