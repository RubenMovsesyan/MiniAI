"""SPPF (Spatial Pyramid Pooling — Fast): 1x1 conv down, 3 sequential 5x5
stride-1 max pools, concat raw + all pooled streams, 1x1 conv blend."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or in_channels // 2
        self.cv1 = Conv(in_channels, hidden, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = Conv(hidden * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        y1 = self.pool(y0)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], dim=1))


if __name__ == "__main__":
    x = torch.randn(2, 64, 20, 20)
    y = SPPF(64, 64)(x)
    assert y.shape == (2, 64, 20, 20), f"bad shape {tuple(y.shape)}"
    print("ok")
