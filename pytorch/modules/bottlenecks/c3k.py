"""C3k: a configurable chain of convs with an optional residual add. Used as the
bottleneck path of a CSP wrapper — see modules/wrappers.py and modules/csp.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..conv import Conv


def _broadcast(val, n: int, name: str) -> list:
    if isinstance(val, (list, tuple)):
        if len(val) != n:
            raise ValueError(f"{name}: list length {len(val)} != num_layers {n}")
        return list(val)
    return [val] * n


class C3k(nn.Module):
    """Chain of `Conv` layers with an optional residual add (input + output)."""

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
            layers.append(Conv(c_in, out_channels, kernel_size=k, stride=s))
            c_in = out_channels
        self.layers = nn.Sequential(*layers)
        self.add = add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return x + y if self.add else y


if __name__ == "__main__":
    x = torch.randn(2, 16, 16, 16)
    y = C3k(16, 16)(x)
    assert y.shape == (2, 16, 16, 16), f"bad shape {tuple(y.shape)}"

    y = C3k(16, 32, add=False, num_layers=3, kernel_size=[1, 3, 1])(x)
    assert y.shape == (2, 32, 16, 16), f"bad shape (no add) {tuple(y.shape)}"

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
