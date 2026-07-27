"""Conv: the conv+BN+SiLU unit shared by every other block in modules/."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


if __name__ == "__main__":
    x = torch.randn(2, 3, 8, 8)
    y = Conv(3, 16)(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape {tuple(y.shape)}"
    print("ok")
