"""TransformNet: the small feed-forward network trained (see train_style.py) to
apply one fixed style in a single forward pass -- the piece that actually runs on
a phone. VGG plays no part at inference; it only supplies the training signal.

Architecture is not finalised. The sketch discussed so far: a stem conv, two
stride-2 downsample convs, a handful of residual blocks, two upsample stages
(resize + conv, not transposed conv -- avoids checkerboarding and has better
mobile-runtime support), and an output conv squashed to [0,1]. Instance norm and
reflection padding throughout; depthwise-separable convs (modules.conv.DWConv)
are a candidate swap for the interior 3x3 convs once a plain version works.

Fully convolutional -- no assumption on input H, W, or aspect ratio.

Run from inside pytorch/:  python -m projects.artist.transform
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """conv-norm-relu-conv-norm + skip connection, the repeated interior block of
    TransformNet's bottleneck. Channel count in == channel count out."""

    def __init__(self, channels: int):
        super().__init__()
        # TODO: reflection-pad + Conv2d(channels, channels, 3) + InstanceNorm2d +
        # ReLU, twice; forward adds the block's input back onto its output.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformNet(nn.Module):
    """Maps a [0,1] RGB image to a [0,1] stylised image in one pass. Fully
    convolutional: runs at any input resolution/aspect ratio unchanged."""

    def __init__(self):
        super().__init__()
        # TODO: stem conv (9x9) -> two stride-2 downsample convs -> N x
        # ResidualBlock -> two upsample stages (nearest-upsample + conv) -> output
        # conv (9x9) -> squash to [0,1]. Channel widths, block count, and the
        # DWConv-vs-plain-conv choice are still open (see module docstring).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("artist/transform.py — skeleton, not implemented yet")
