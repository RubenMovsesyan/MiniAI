"""PSA (Parallel Spatial Attention): 1x1 conv channel match, then a History
path (identity) and an Active path (Q/K/V self-attention) that meet back up
via elementwise add, followed by a final 1x1 conv. The Active path supports
a DPU-unaware (matmul) flow or a DPU-aware (elementwise, transpose-free) flow."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..conv import Conv

_ACTIVATIONS = {
    "softmax": lambda: nn.Softmax(dim=-1),
    "hard_sigmoid": nn.Hardsigmoid,
}


class PSA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None,
                 dpu_aware: bool = False, activation: str = "softmax"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"PSA: unknown activation {activation!r}, expected one of {list(_ACTIVATIONS)}")

        hidden = hidden_channels or out_channels
        self.hidden = hidden
        self.dpu_aware = dpu_aware

        self.cv_in = Conv(in_channels, hidden, kernel_size=1)
        self.cv_q = Conv(hidden, hidden, kernel_size=1)
        self.cv_k = Conv(hidden, hidden, kernel_size=1)
        self.cv_v = Conv(hidden, hidden, kernel_size=1)
        self.cv_fuse = Conv(hidden, hidden, kernel_size=1)
        self.cv_out = Conv(hidden, out_channels, kernel_size=1)
        self.act = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.cv_in(x)
        history = x0

        b, c, h, w = x0.shape
        q = self.cv_q(x0).reshape(b, c, h * w)
        k = self.cv_k(x0).reshape(b, c, h * w)
        v = self.cv_v(x0).reshape(b, c, h * w)

        if self.dpu_aware:
            attn = self.act(q * k)
            out = attn * v
        else:
            attn = self.act(torch.matmul(q, k.transpose(1, 2)) * (self.hidden ** -0.5))
            out = torch.matmul(attn, v)

        out = out.reshape(b, c, h, w)
        out = self.cv_fuse(out)
        return self.cv_out(history + out)


if __name__ == "__main__":
    x = torch.randn(2, 3, 8, 8)

    y = PSA(3, 16)(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape {tuple(y.shape)}"

    y = PSA(3, 16, dpu_aware=True)(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape (dpu_aware) {tuple(y.shape)}"

    y = PSA(3, 16, activation="hard_sigmoid")(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape (hard_sigmoid) {tuple(y.shape)}"

    y = PSA(3, 16, dpu_aware=True, activation="hard_sigmoid")(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape (dpu_aware+hard_sigmoid) {tuple(y.shape)}"

    y = PSA(3, 16, hidden_channels=32)(x)
    assert y.shape == (2, 16, 8, 8), f"bad shape (hidden_channels) {tuple(y.shape)}"

    y = PSA(3, 3)(x)
    assert y.shape == (2, 3, 8, 8), f"bad shape (in==out) {tuple(y.shape)}"

    try:
        PSA(3, 16, activation="bogus")
        raise AssertionError("expected ValueError for unknown activation")
    except ValueError:
        pass

    print("ok")
