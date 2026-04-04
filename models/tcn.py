from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch optional at import time
    torch = None
    nn = None


if nn is None:
    class TCN:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for models.tcn.TCN")

else:
    class CausalConv1d(nn.Conv1d):
        def __init__(self, *args, **kwargs):
            self._left_padding = (kwargs.get("kernel_size", 3) - 1) * kwargs.get("dilation", 1)
            super().__init__(*args, **kwargs)

        def forward(self, x):
            x = nn.functional.pad(x, (self._left_padding, 0))
            return super().forward(x)


    class TCNBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.net(x)


    class TCN(nn.Module):
        """Dilated causal TCN with residual blocks.

        Input shape: (batch, 50, 7)
        Output: embedding tensor (batch, hidden_dim)
        """

        def __init__(self, in_channels: int = 7, hidden_dim: int = 64, depth: int = 4, kernel_size: int = 3):
            super().__init__()
            self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            blocks = []
            for i in range(depth):
                blocks.append(TCNBlock(hidden_dim, kernel_size=kernel_size, dilation=2**i, dropout=0.1))
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        def forward(self, x):
            # (B, T, C) -> (B, C, T)
            x = x.transpose(1, 2)
            x = self.proj(x)
            x = self.blocks(x)
            return self.head(x)
