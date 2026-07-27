"""CSP wrappers: take any stack of layers as a bottleneck path, run a second
bypass path alongside it, concat the two and blend with a 1x1 conv.

`C3` puts a conv on the bypass path, `C2` does not — hence 3 convs vs 2. The
bottleneck path is whatever modules you hand in: one, many, mixed types. They
must map `hidden -> hidden` channels; see modules/csp.py for ready-made combos.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .conv import Conv


def _as_sequential(blocks: nn.Module | Sequence[nn.Module]) -> nn.Module:
    # ponytail: no check that blocks map hidden -> hidden; a mismatch shows up
    # as a normal torch shape error at the first forward.
    return blocks if isinstance(blocks, nn.Module) else nn.Sequential(*blocks)


class C3(nn.Module):
    """Three convs: bottleneck-path input, bypass path, output blend."""

    def __init__(self, in_channels: int, out_channels: int,
                 blocks: nn.Module | Sequence[nn.Module],
                 hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or out_channels // 2
        self.cv1 = Conv(in_channels, hidden, kernel_size=1)
        self.cv2 = Conv(in_channels, hidden, kernel_size=1)
        self.blocks = _as_sequential(blocks)
        self.cv3 = Conv(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class C2(nn.Module):
    """Two convs: bottleneck-path input and output blend. The bypass path is the
    raw input, so the output conv sees `in_channels + hidden`, not `hidden * 2`."""

    def __init__(self, in_channels: int, out_channels: int,
                 blocks: nn.Module | Sequence[nn.Module],
                 hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or out_channels // 2
        self.cv1 = Conv(in_channels, hidden, kernel_size=1)
        self.blocks = _as_sequential(blocks)
        self.cv2 = Conv(in_channels + hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(torch.cat([self.blocks(self.cv1(x)), x], dim=1))


if __name__ == "__main__":
    from .bottlenecks.c3k import C3k
    from .bottlenecks.psa import PSA

    x = torch.randn(2, 3, 16, 16)
    h = 32 // 2

    for wrapper in (C3, C2):
        name = wrapper.__name__

        y = wrapper(3, 32, C3k(h, h))(x)  # a bare module, not a list
        assert y.shape == (2, 32, 16, 16), f"{name} bad shape (single) {tuple(y.shape)}"

        y = wrapper(3, 32, [C3k(h, h) for _ in range(3)])(x)
        assert y.shape == (2, 32, 16, 16), f"{name} bad shape (3 blocks) {tuple(y.shape)}"

        y = wrapper(3, 32, [C3k(h, h), PSA(h, h), C3k(h, h)])(x)
        assert y.shape == (2, 32, 16, 16), f"{name} bad shape (mixed) {tuple(y.shape)}"

        y = wrapper(3, 8, [C3k(20, 20)], hidden_channels=20)(x)
        assert y.shape == (2, 8, 16, 16), f"{name} bad shape (hidden) {tuple(y.shape)}"

    print("ok")
