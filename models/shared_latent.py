from __future__ import annotations

"""Shared latent representation scaffold for migration from multi-head heterogeneity.

This module is intentionally additive and non-breaking. It provides a single
sequence encoder `h = g(X_window)` and three task heads (anomaly, fault,
RUL) to support gradual migration from post-hoc fusion to multi-task learning.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None


if nn is None:
    class SharedLatentModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for models.shared_latent.SharedLatentModel")

else:
    class SharedEncoder(nn.Module):
        def __init__(self, in_channels: int = 7, hidden_dim: int = 128) -> None:
            super().__init__()
            self.rnn = nn.GRU(
                input_size=in_channels,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )

        def forward(self, x):
            # x: (B, T, C)
            out, _ = self.rnn(x)
            h = out[:, -1, :]
            return self.proj(h)


    class SharedLatentModel(nn.Module):
        """Multi-task heads on top of shared latent state h."""

        def __init__(self, in_channels: int = 7, hidden_dim: int = 128, n_faults: int = 5) -> None:
            super().__init__()
            self.encoder = SharedEncoder(in_channels=in_channels, hidden_dim=hidden_dim)

            self.anomaly_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            self.fault_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_faults),
            )
            self.rul_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Softplus(),
            )

        def forward(self, x):
            h = self.encoder(x)
            return {
                "h": h,
                "anomaly_score": self.anomaly_head(h).squeeze(-1),
                "fault_logits": self.fault_head(h),
                "rul_hours": self.rul_head(h).squeeze(-1),
            }


    class SharedLatentRuntime:
        """Train/infer wrapper for shared latent representation.

        The model predicts three targets jointly:
        - anomaly score in [0,1]
        - fault class logits
        - relative RUL in [0,1]
        """

        def __init__(
            self,
            in_channels: int = 7,
            hidden_dim: int = 128,
            n_faults: int = 5,
            device: Optional[str] = None,
        ) -> None:
            self._in_channels = in_channels
            self._hidden_dim = hidden_dim
            self._n_faults = n_faults
            self._trained = False

            if torch is None:
                raise ImportError("PyTorch is required for SharedLatentRuntime")

            self._device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = SharedLatentModel(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                n_faults=n_faults,
            ).to(self._device)
            self._x_mean = np.zeros((1, 1, in_channels), dtype=np.float32)
            self._x_std = np.ones((1, 1, in_channels), dtype=np.float32)
            self._rul_scale = 600.0

        def is_trained(self) -> bool:
            return self._trained

        def _norm(self, x: np.ndarray) -> np.ndarray:
            return (x - self._x_mean) / self._x_std

        def fit(
            self,
            X_windows: np.ndarray,
            y_class: np.ndarray,
            y_rul: np.ndarray,
            epochs: int = 20,
            batch_size: int = 128,
            lr: float = 1e-3,
            lambda_anom: float = 1.0,
            lambda_cls: float = 1.0,
            lambda_rul: float = 1.0,
            lambda_consistency: float = 0.2,
            k_consistency: float = 1.0,
        ) -> dict[str, float]:
            X = np.asarray(X_windows, dtype=np.float32)
            y_c = np.asarray(y_class, dtype=np.int64)
            y_r = np.asarray(y_rul, dtype=np.float32)

            self._x_mean = np.mean(X, axis=(0, 1), keepdims=True).astype(np.float32)
            self._x_std = (np.std(X, axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
            Xn = self._norm(X)

            y_anom = (y_c != 0).astype(np.float32)
            y_rul_rel = np.clip(y_r / self._rul_scale, 0.0, 1.0).astype(np.float32)

            xt = torch.tensor(Xn, device=self._device)
            y_anom_t = torch.tensor(y_anom, device=self._device)
            y_cls_t = torch.tensor(y_c, device=self._device)
            y_rul_t = torch.tensor(y_rul_rel, device=self._device)

            n = len(Xn)
            opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)

            self._model.train()
            for _ in range(max(1, int(epochs))):
                perm = np.random.permutation(n)
                for i in range(0, n, int(batch_size)):
                    idx = perm[i : i + int(batch_size)]
                    xb = xt[idx]
                    ya = y_anom_t[idx]
                    yc = y_cls_t[idx]
                    yr = y_rul_t[idx]

                    out = self._model(xb)
                    pred_a = out["anomaly_score"]
                    pred_c = out["fault_logits"]
                    pred_r = out["rul_hours"]

                    loss_a = F.binary_cross_entropy(pred_a, ya)
                    loss_c = F.cross_entropy(pred_c, yc)
                    loss_r = F.smooth_l1_loss(pred_r, yr)
                    # Enforce anomaly/RUL consistency in relative space.
                    loss_cons = torch.relu(pred_r - k_consistency * (1.0 - pred_a)).mean()

                    loss = (
                        lambda_anom * loss_a
                        + lambda_cls * loss_c
                        + lambda_rul * loss_r
                        + lambda_consistency * loss_cons
                    )

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            self._model.eval()
            with torch.no_grad():
                out = self._model(xt)
                cls_pred = torch.argmax(out["fault_logits"], dim=1)
                cls_acc = (cls_pred == y_cls_t).float().mean().item()
                rul_mae_rel = torch.mean(torch.abs(out["rul_hours"] - y_rul_t)).item()
                anom_mae = torch.mean(torch.abs(out["anomaly_score"] - y_anom_t)).item()

            self._trained = True
            return {
                "class_acc": round(float(cls_acc), 4),
                "rul_mae_rel": round(float(rul_mae_rel), 4),
                "anomaly_mae": round(float(anom_mae), 4),
            }

        def encode_batch(self, X_windows: np.ndarray) -> np.ndarray:
            X = np.asarray(X_windows, dtype=np.float32)
            Xn = self._norm(X)
            xt = torch.tensor(Xn, device=self._device)
            self._model.eval()
            with torch.no_grad():
                h = self._model(xt)["h"].detach().cpu().numpy().astype(np.float32)
            return h

        def encode_window(self, window: np.ndarray) -> np.ndarray:
            return self.encode_batch(window[np.newaxis, ...])[0]

        def predict_heads(self, window: np.ndarray) -> dict[str, Any]:
            X = np.asarray(window, dtype=np.float32)[np.newaxis, ...]
            Xn = self._norm(X)
            xt = torch.tensor(Xn, device=self._device)
            self._model.eval()
            with torch.no_grad():
                out = self._model(xt)
                probs = torch.softmax(out["fault_logits"], dim=1).detach().cpu().numpy()[0]
                return {
                    "anomaly_score": float(out["anomaly_score"].detach().cpu().numpy()[0]),
                    "fault_probs": probs.astype(np.float32),
                    "rul_relative": float(out["rul_hours"].detach().cpu().numpy()[0]),
                    "h": out["h"].detach().cpu().numpy()[0].astype(np.float32),
                }

        def save(self, path: Path | str) -> None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "state_dict": self._model.state_dict(),
                "in_channels": self._in_channels,
                "hidden_dim": self._hidden_dim,
                "n_faults": self._n_faults,
                "x_mean": self._x_mean,
                "x_std": self._x_std,
                "rul_scale": self._rul_scale,
                "trained": self._trained,
            }
            torch.save(payload, p)

        @classmethod
        def load(cls, path: Path | str, device: Optional[str] = None) -> "SharedLatentRuntime":
            p = Path(path)
            payload = torch.load(p, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
            obj = cls(
                in_channels=int(payload.get("in_channels", 7)),
                hidden_dim=int(payload.get("hidden_dim", 128)),
                n_faults=int(payload.get("n_faults", 5)),
                device=device,
            )
            obj._model.load_state_dict(payload["state_dict"])
            obj._x_mean = np.asarray(payload.get("x_mean", obj._x_mean), dtype=np.float32)
            obj._x_std = np.asarray(payload.get("x_std", obj._x_std), dtype=np.float32)
            obj._rul_scale = float(payload.get("rul_scale", 600.0))
            obj._trained = bool(payload.get("trained", True))
            obj._model.eval()
            return obj
