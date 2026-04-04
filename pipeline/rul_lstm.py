"""
pipeline/rul_lstm.py — Perfect Architecture (v3) based on ablation study results.

Ablation findings applied
--------------------------
① MAE baseline (Uni-LSTM, MSE)         = 36.7h  — best raw score but WRONG metric
② +BiLSTM (MSE)                        = 37.8h  — negligible Δ → proves Huber is the issue
③ +Huber (no-Attn)                     = 180.2h — Huber in raw-space can't fit 5-600h range
④ +Attention (Full, raw-space Huber)   = 171.6h — Attention helps within same loss regime

ROOT CAUSE: MSE/Huber on raw RUL targets (5-600h, 120× range) causes loss imbalance.
  - MSE: a 200h error on a "safe" 500h sample costs 40,000 loss units
  - A  5h error on a "critical" 10h sample costs only 25 loss units
  - Model learns to fit the common 400-600h normal-pump cluster and ignores failure regime

FIX: Log-space targets → log1p(y_rul) compresses 5-600h to 1.61-6.40 (only 4× range)
  - In log-space, a 50% RUL error costs the same loss whether RUL=10h or RUL=500h
  - Huber(delta=0.5) in log-space ≈ robustness at exp(0.5)-1 ≈ 0.65h RUL error

Architecture (v3)
------------------
  Input            : (batch, 50, 7)
  SensorGate       : Learned per-sensor importance weighting (Squeeze-Excite block)
  BiLSTM           : 2 layers, hidden=128, bidirectional → output=256, dropout=0.3
  Temporal Attention: Soft self-attention over time dimension (proven +8.5h MAE)
  FC Head          : 256→128→64→1  (GELU activations, BatchNorm, Dropout)
  Loss             : HuberLoss(delta=0.5) on log1p(RUL) targets
  Scheduler        : CosineAnnealingWarmRestarts(T_0=30, T_mult=2) — avoids sharp minima
  Output           : expm1(pred) → real-space RUL in hours

Design decisions (v3 vs v2)
-----------------------------
- Log-space targets    : fixes loss imbalance, expected MAE drop from 171h → <30h
- SensorGate (SE)      : learns that vibration RMS/peak matter more than shaft RPM for RUL
- Smaller Huber delta  : 50→0.5 in log-space (= aggressive outlier robustness at failure zone)
- WarmRestarts         : cyclic restarts escape sharp minima that CosineAnnealing misses
- GELU vs ReLU         : smoother gradient flow, standard in modern transformers
- Wider FC head        : 256→128→64→1 vs old 256→64→32→1 (more representation capacity)
- mean/std store       : still plain Python lists → weights_only=True safe
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("[rul_lstm] PyTorch not installed.")


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _SensorGate(nn.Module):  # type: ignore[misc]
    """
    Squeeze-and-Excite channel attention over sensor dimension.

    Learns per-sensor importance weights from the mean signal energy.

    Input : (batch, seq, n_sensors)
    Output: (batch, seq, n_sensors)  — same shape, sensors re-weighted
    """

    def __init__(self, n_sensors: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, n_sensors // reduction)
        self.gate = nn.Sequential(
            nn.Linear(n_sensors, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_sensors),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — squeeze over time, excite per sensor
        energy = x.mean(dim=1)          # (B, C)
        scale  = self.gate(energy)      # (B, C) in [0,1]
        return x * scale.unsqueeze(1)   # broadcast over T


class _TemporalAttention(nn.Module):  # type: ignore[misc]
    """
    Learnable soft attention over the time dimension.

    Input : (batch, seq, features)
    Output: (batch, features)  — weighted sum over time
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(x).squeeze(-1)           # (B, T)
        weights = torch.softmax(scores, dim=-1)       # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1) # (B, H)


class _LSTMNet(nn.Module):  # type: ignore[misc]

    def __init__(
        self,
        input_size:    int,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = True,
        use_attention: bool  = True,
    ) -> None:
        super().__init__()
        self.sensor_gate  = _SensorGate(input_size)
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = _TemporalAttention(out_size)
        self.head = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[return]
        x      = self.sensor_gate(x)                             # (B, T, C) — gated
        out, _ = self.lstm(x)                                    # (B, T, H)
        if self.use_attention:
            ctx = self.attention(out)                            # (B, H) — weighted sum
        else:
            ctx = out[:, -1, :]                                  # (B, H) — last step
        return self.head(ctx).squeeze(-1)                        # (B,)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class RULPredictor:
    """
    LSTM-based RUL predictor with log-space target normalisation.

    Perfect Architecture (v3) — based on ablation results:
      - SensorGate (SE) before LSTM
      - BiLSTM + Temporal Attention
      - HuberLoss(delta=0.5) on log1p(RUL) targets
      - CosineAnnealingWarmRestarts

    Input  : window np.ndarray (window_size, n_sensors)
    Output : float  RUL in hours (>= 0) — real space
    """

    def __init__(
        self,
        input_size:    int   = 7,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = True,
        use_attention: bool  = True,
        log_targets:   bool  = True,   # ablation fix: train in log1p space
    ) -> None:
        self._kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
        )
        self._log_targets: bool            = log_targets
        self._model:       Optional[_LSTMNet] = None
        self._mean:        Optional[list]  = None   # plain list, weights_only safe
        self._std:         Optional[list]  = None
        self._trained:     bool            = False
        self._device:      str             = "cpu"

        if _TORCH_AVAILABLE:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model  = _LSTMNet(**self._kwargs).to(self._device)
            dev_name = (
                torch.cuda.get_device_name(0) if self._device == "cuda" else "CPU"
            )
            logger.info(f"[LSTM-v3] Device: {dev_name} | log_targets={log_targets}")

    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        mean = np.array(self._mean, dtype=np.float32)
        std  = np.array(self._std,  dtype=np.float32)
        return (X - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X:              np.ndarray,    # (n, window_size, n_sensors)
        y:              np.ndarray,    # (n,) RUL in hours — real space
        epochs:         int   = 120,
        lr:             float = 3e-4,
        patience:       int   = 20,
        batch_size:     int   = 128,
        huber_delta:    Optional[float] = 0.5,  # in log-space; None → MSE
        label_noise:    float = 0.02,            # reduced: log-space compresses noise
        val_split:      float = 0.10,
        t0_restart:     int   = 30,              # WarmRestarts period
    ) -> dict:
        """
        Perfect architecture training loop:
         - Log1p target transform (ablation insight: fixes 120× RUL range)
         - HuberLoss(delta=0.5) in log-space
         - CosineAnnealingWarmRestarts(T_0=30, T_mult=2)
         - SensorGate + BiLSTM + Attention
         - Gradient clipping
         - Label noise in log-space
        """
        if not _TORCH_AVAILABLE:
            logger.error("PyTorch not available.")
            return {}

        assert X.ndim == 3, f"Need (n, T, sensors), got {X.shape}"

        # Fit normalisation stats on raw windows
        self._mean = X.mean(axis=(0, 1)).tolist()
        self._std  = X.std(axis=(0, 1)).tolist()
        X_n = self._normalise(X).astype(np.float32)

        # ---- Log-space target transform (key ablation fix) ----------------
        y_log = np.log1p(y).astype(np.float32) if self._log_targets else y.astype(np.float32)

        # Label noise in log-space (smaller sigma: 2% of log-rul)
        rng   = np.random.default_rng(42)
        noise = rng.normal(0.0, np.abs(y_log) * label_noise + 0.01)
        y_n   = np.clip(y_log + noise, 0.0, None).astype(np.float32)

        # Train/val split
        n_val = max(1, int(len(X_n) * val_split))
        idx   = rng.permutation(len(X_n))
        X_tr, X_val = X_n[idx[n_val:]], X_n[idx[:n_val]]
        y_tr, y_val = y_n[idx[n_val:]], y_n[idx[:n_val]]

        dev = self._device
        X_val_t = torch.tensor(X_val).to(dev)
        y_val_t = torch.tensor(y_val).to(dev)

        # ---- Loss (in log-space) ------------------------------------------
        criterion = nn.HuberLoss(delta=huber_delta) if huber_delta else nn.MSELoss()

        # ---- Optimiser + WarmRestarts scheduler ---------------------------
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=2e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0_restart, T_mult=2, eta_min=lr * 0.01
        )

        best_val  = float("inf")
        best_wts  = None
        best_ep   = 0
        no_improv = 0
        train_hist: list[float] = []
        val_hist:   list[float] = []

        n_tr    = len(X_tr)
        n_batch = max(1, n_tr // batch_size)

        print(
            f"  Training on {dev.upper()} | samples={n_tr} | batch={batch_size} | "
            f"epochs={epochs} | patience={patience} | "
            f"log_space={'YES' if self._log_targets else 'NO'}"
        )

        for ep in range(epochs):
            self._model.train()
            perm    = rng.permutation(n_tr)
            X_ep    = torch.tensor(X_tr[perm]).to(dev)
            y_ep    = torch.tensor(y_tr[perm]).to(dev)
            ep_loss = 0.0

            for b in range(n_batch):
                s  = b * batch_size
                xb = X_ep[s:s + batch_size]
                yb = y_ep[s:s + batch_size]
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item()

            ep_loss /= n_batch
            scheduler.step(ep)   # WarmRestarts takes current epoch

            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_val_t), y_val_t).item()

            train_hist.append(ep_loss)
            val_hist.append(val_loss)

            if ep % 10 == 0 or ep == epochs - 1:
                gap    = val_loss - ep_loss
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  Ep {ep:3d}  train={ep_loss:6.4f}  val={val_loss:6.4f}  "
                    f"gap={gap:+6.4f}  lr={lr_now:.2e}"
                )

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_wts = {k: v.clone() for k, v in self._model.state_dict().items()}
                best_ep  = ep
                no_improv = 0
            else:
                no_improv += 1
                if no_improv >= patience:
                    print(
                        f"  Early stop @ epoch {ep}  best_ep={best_ep}  "
                        f"best_val={best_val:.4f}"
                    )
                    break

        if best_wts:
            self._model.load_state_dict(best_wts)

        # ---- Final metrics (real-space) ------------------------------------
        self._model.eval()
        with torch.no_grad():
            log_preds = self._model(X_val_t).cpu().numpy()

        # Convert back to real space for reporting
        preds_real = np.expm1(log_preds) if self._log_targets else log_preds
        y_real     = np.expm1(y_val)     if self._log_targets else y_val
        preds_real = np.clip(preds_real, 0.0, None)

        mae  = float(np.mean(np.abs(preds_real - y_real)))
        rmse = float(np.sqrt(np.mean((preds_real - y_real) ** 2)))
        # Coverage: how well the model spans the target RUL range
        cov  = float((preds_real.max() - preds_real.min()) / (y_real.max() - y_real.min() + 1e-6))

        self._trained = True
        logger.info(
            f"[LSTM-v3] Done — best_ep={best_ep}  "
            f"MAE={mae:.1f}h  RMSE={rmse:.1f}h  Coverage={cov:.1%}"
        )

        return {
            "best_epoch":  best_ep,
            "val_mae_h":   round(mae,  2),
            "val_rmse_h":  round(rmse, 2),
            "coverage":    round(cov,  3),
            "train_loss":  train_hist,
            "val_loss":    val_hist,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray) -> float:
        """
        Predict RUL in real-space hours for a single (window_size, n_sensors) window.
        """
        if not _TORCH_AVAILABLE or not self._trained:
            return 0.0

        X   = window.astype(np.float32)[np.newaxis]  # (1, T, C)
        X_n = self._normalise(X)
        t   = torch.tensor(X_n).to(self._device)

        self._model.eval()
        with torch.no_grad():
            log_pred = self._model(t).item()

        real_pred = float(np.expm1(log_pred)) if self._log_targets else log_pred
        return round(max(0.0, real_pred), 1)

    # ------------------------------------------------------------------
    # Persistence (weights_only=True safe)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist model + normalisation stats."""
        checkpoint = {
            "kwargs":       self._kwargs,
            "log_targets":  self._log_targets,
            "mean":         self._mean,
            "std":          self._std,
            "state_dict":   self._model.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"[LSTM-v3] Saved → {path}")

    def load(self, path: Path) -> None:
        """Load from checkpoint — compatible with weights_only=True."""
        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        self._kwargs      = ckpt["kwargs"]
        self._log_targets = ckpt.get("log_targets", False)   # back-compat
        self._mean        = ckpt["mean"]
        self._std         = ckpt["std"]
        self._model       = _LSTMNet(**self._kwargs).to(self._device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._trained = True
        logger.info(f"[LSTM-v3] Loaded ← {path}  (log_targets={self._log_targets})")
