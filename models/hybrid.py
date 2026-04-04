from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch optional at import time
    torch = None
    nn = None

from models.tcn import TCN


if nn is None:
    class HybridModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for models.hybrid.HybridModel")

else:
    class HybridModel(nn.Module):
        """Fuse sequence embedding with engineered feature vector.

        Inputs:
          seq_x:  (batch, 50, 7)
          feat_x: (batch, feat_dim)
        Outputs:
          logits: (batch, n_classes)
          severity: (batch, 1)
        """

        def __init__(self, feat_dim: int = 84, n_classes: int = 5, seq_emb_dim: int = 64):
            super().__init__()
            self.seq = TCN(in_channels=7, hidden_dim=seq_emb_dim)
            self.feat = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(seq_emb_dim + 64, 128),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.classifier = nn.Linear(128, n_classes)
            self.severity = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, seq_x, feat_x):
            seq_e = self.seq(seq_x)
            feat_e = self.feat(feat_x)
            z = self.fuse(torch.cat([seq_e, feat_e], dim=1))
            return {
                "logits": self.classifier(z),
                "severity": self.severity(z),
            }
