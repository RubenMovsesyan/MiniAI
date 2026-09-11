"""Conv: the conv+BN+activation unit shared by every other block in modules/.
DWConv: depthwise separable conv (depthwise Conv + pointwise 1x1 Conv), built from it.
ConvBlock: reflection-pad + InstanceNorm + activation, for image-synthesis nets
(style transfer, autoencoders) where Conv's zero-padding and BatchNorm cause
border artifacts and cross-image statistic leakage, respectively."""

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


class ConvBlock(nn.Module):
    """Reflection-padded Conv2d -> optional InstanceNorm2d -> optional activation.

    Reflection padding (not zero-padding) avoids the dark border artifacts
    image-synthesis networks are prone to. InstanceNorm (not BatchNorm) normalises
    each image against its own statistics, so unrelated images sharing a training
    batch don't contaminate each other's style. `norm=False, activation=None` gives
    a bare conv, for an output layer that needs neither."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, norm: bool = True,
                 activation: str | None = "relu"):
        super().__init__()
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)} or None, got {activation!r}")
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else None
        self.act = _ACTIVATIONS[activation]() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DWConvBlock(nn.Module):
    """Depthwise-separable ConvBlock: a depthwise (groups=in_channels) conv
    followed by a pointwise 1x1 conv, each with ConvBlock's reflection padding +
    InstanceNorm + activation -- the mobile-friendly counterpart to DWConv, for
    image-synthesis nets that need InstanceNorm instead of BatchNorm (see
    ConvBlock). Same interface as ConvBlock; far fewer weights for the same
    in/out channel shape (e.g. 128->128, 3x3: ~147K for ConvBlock, ~17.5K here)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, norm: bool = True, activation: str | None = "relu"):
        super().__init__()
        self.depthwise = ConvBlock(in_channels, in_channels, kernel_size, stride,
                                    groups=in_channels, norm=norm, activation=activation)
        self.pointwise = ConvBlock(in_channels, out_channels, kernel_size=1,
                                    norm=norm, activation=activation)

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

    y = ConvBlock(3, 16)(x)
    assert y.shape == (2, 16, 8, 8), f"bad ConvBlock shape {tuple(y.shape)}"
    print("ok (ConvBlock)")

    y = ConvBlock(16, 16, stride=2)(y)
    assert y.shape == (2, 16, 4, 4), f"bad ConvBlock stride shape {tuple(y.shape)}"
    print("ok (ConvBlock stride)")

    y = ConvBlock(3, 16, norm=False, activation=None)(x)
    assert y.shape == (2, 16, 8, 8), f"bad ConvBlock (bare) shape {tuple(y.shape)}"
    print("ok (ConvBlock norm=False, activation=None)")

    y = ConvBlock(3, 16, activation=None)(x + 100)  # big offset -- InstanceNorm should erase it
    assert y.abs().mean() < 5, f"InstanceNorm doesn't look applied, mean={y.abs().mean():.2f}"
    print("ok (ConvBlock normalises per-instance)")

    z = torch.randn(2, 16, 8, 8)
    y = DWConvBlock(16, 32)(z)
    assert y.shape == (2, 32, 8, 8), f"bad DWConvBlock shape {tuple(y.shape)}"
    print("ok (DWConvBlock)")

    y = DWConvBlock(16, 32, stride=2)(z)
    assert y.shape == (2, 32, 4, 4), f"bad DWConvBlock stride shape {tuple(y.shape)}"
    print("ok (DWConvBlock stride)")

    full = sum(p.numel() for p in ConvBlock(128, 128, 3).parameters())
    dw = sum(p.numel() for p in DWConvBlock(128, 128, 3).parameters())
    assert dw < full, f"DWConvBlock ({dw:,}) should be smaller than ConvBlock ({full:,})"
    print(f"ok (DWConvBlock lighter: {dw:,} vs ConvBlock's {full:,} params, 128->128 3x3)")
