"""C3k2 building block (YOLOv8-style CSP block): ConvBNSiLU (conv+BN+SiLU),
C3k (configurable conv chain with optional residual add), C3k2 (two-path CSP
block built from C3k)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _broadcast(val, n: int, name: str) -> list:
    if isinstance(val, (list, tuple)):
        if len(val) != n:
            raise ValueError(f"{name}: list length {len(val)} != num_layers {n}")
        return list(val)
    return [val] * n


class ConvBNSiLU(nn.Module):
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


class C3k(nn.Module):
    """Chain of `ConvBNSiLU` layers with an optional residual add (input + output)."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2,
                 kernel_size: int | list[int] = 3, stride: int | list[int] = 1,
                 add: bool = True):
        super().__init__()
        kernel_sizes = _broadcast(kernel_size, num_layers, "kernel_size")
        strides = _broadcast(stride, num_layers, "stride")

        if add and in_channels != out_channels:
            raise ValueError(
                f"C3k: add=True requires in_channels == out_channels "
                f"({in_channels} != {out_channels})"
            )
        total_stride = 1
        for s in strides:
            total_stride *= s
        if add and total_stride != 1:
            raise ValueError(f"C3k: add=True requires combined stride == 1 (got {total_stride})")

        layers = []
        c_in = in_channels
        for k, s in zip(kernel_sizes, strides):
            layers.append(ConvBNSiLU(c_in, out_channels, kernel_size=k, stride=s))
            c_in = out_channels
        self.layers = nn.Sequential(*layers)
        self.add = add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return x + y if self.add else y


class C3k2(nn.Module):
    """Two-path CSP block: 1x1 conv split, one path through a stack of `C3k`
    blocks, both paths concatenated (dim=1) and blended by a final 1x1 conv."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 hidden_channels: int | None = None, add: bool = True, **c3k_kwargs):
        super().__init__()
        hidden = hidden_channels or out_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden, kernel_size=1)
        self.cv2 = ConvBNSiLU(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[C3k(hidden, hidden, add=add, **c3k_kwargs) for _ in range(num_blocks)]
        )
        self.cv3 = ConvBNSiLU(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cv1(x)
        b = self.blocks(self.cv2(x))
        return self.cv3(torch.cat([a, b], dim=1))


if __name__ == "__main__":
    x = torch.randn(2, 3, 16, 16)
    y = C3k2(3, 32)(x)
    assert y.shape == (2, 32, 16, 16), f"bad shape {tuple(y.shape)}"

    C3k(16, 16, add=True)  # ok: channels match, stride 1

    try:
        C3k(16, 32, add=True)
        raise AssertionError("expected ValueError for channel mismatch")
    except ValueError:
        pass

    try:
        C3k(16, 16, stride=2, add=True)
        raise AssertionError("expected ValueError for stride mismatch")
    except ValueError:
        pass

    print("ok")
