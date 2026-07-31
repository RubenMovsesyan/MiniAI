"""Conv: the conv+BN+activation unit shared by every other block in modules/.
DWConv: depthwise separable conv (depthwise Conv + pointwise 1x1 Conv), built from it."""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU}


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> activation ("silu" or "relu")."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, activation: str = "silu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable conv: depthwise (groups=in_channels) Conv, then pointwise 1x1 Conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 activation: str = "silu"):
        super().__init__()
        self.depthwise = Conv(in_channels, in_channels, kernel_size=kernel_size,
                               stride=1, groups=in_channels, activation=activation)
        self.pointwise = Conv(in_channels, out_channels, kernel_size=1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


if __name__ == "__main__":
    x = torch.randn(2, 3, 8, 8)
    y = Conv(3, 16)(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape {tuple(y.shape)}"
    print("ok")

    y = Conv(3, 16, activation="relu")(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape (relu) {tuple(y.shape)}"
    print("ok (relu)")

    y = DWConv(3, 16)(x)
    assert y.shape == (2, 16, 8, 8), f"bad DWConv shape {tuple(y.shape)}"
    print("ok (DWConv)")
