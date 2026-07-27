"""Ready-made CSP blocks: a bottleneck layer type dropped into a wrapper from
modules/wrappers.py. `C3k2` is `C3k` in the `C3` wrapper, `C2PSA` is `PSA` in
the `C2` wrapper. For a mixed bottleneck path, use the wrappers directly."""

from __future__ import annotations

import torch

from .bottlenecks.c3k import C3k
from .bottlenecks.psa import PSA
from .wrappers import C2, C3


class C3k2(C3):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 hidden_channels: int | None = None, **c3k_kwargs):
        hidden = hidden_channels or out_channels // 2
        super().__init__(in_channels, out_channels,
                         [C3k(hidden, hidden, **c3k_kwargs) for _ in range(num_blocks)],
                         hidden_channels=hidden)


class C2PSA(C2):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 hidden_channels: int | None = None, **psa_kwargs):
        hidden = hidden_channels or out_channels // 2
        super().__init__(in_channels, out_channels,
                         [PSA(hidden, hidden, **psa_kwargs) for _ in range(num_blocks)],
                         hidden_channels=hidden)


if __name__ == "__main__":
    x = torch.randn(2, 3, 16, 16)

    y = C3k2(3, 32)(x)
    assert y.shape == (2, 32, 16, 16), f"bad shape {tuple(y.shape)}"

    y = C3k2(3, 32, num_blocks=3, add=False, kernel_size=1)(x)
    assert y.shape == (2, 32, 16, 16), f"bad C3k2 shape (kwargs) {tuple(y.shape)}"

    y = C2PSA(3, 32)(x)
    assert y.shape == (2, 32, 16, 16), f"bad C2PSA shape {tuple(y.shape)}"

    y = C2PSA(3, 32, num_blocks=2, dpu_aware=True, activation="hard_sigmoid")(x)
    assert y.shape == (2, 32, 16, 16), f"bad C2PSA shape (kwargs) {tuple(y.shape)}"

    print("ok")
