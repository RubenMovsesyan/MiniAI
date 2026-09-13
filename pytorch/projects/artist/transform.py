"""TransformNet: the small feed-forward network trained (see train_style.py) to
apply one fixed style in a single forward pass -- the piece that actually runs on
a phone. VGG plays no part at inference; it only supplies the training signal.

Architecture (Johnson et al. 2016): a wide stem conv, two stride-2 downsample
convs, a stack of residual blocks, two upsample stages (nearest-upsample + conv,
not transposed conv -- avoids checkerboarding and has better mobile-runtime
support), and an output conv squashed to [0,1] via tanh. modules.conv.ConvBlock
supplies the reflection padding + InstanceNorm + activation used throughout.

`depthwise=True` swaps every *interior* 3x3 conv (downsample, residual, upsample)
for modules.conv.DWConvBlock -- far fewer weights, same InstanceNorm/reflection-pad
conventions. The 9x9 stem and output convs stay full ConvBlocks regardless: the
stem needs to jointly mix the 3 raw RGB channels (a depthwise conv over only 3
channels barely saves anything and loses cross-channel mixing at the one place it
matters most), and the output is a plain linear projection back to 3 channels.

Fully convolutional -- runs at any input H, W (both should be multiples of 4, the
network's total downsample factor) and any aspect ratio.

Run from inside pytorch/:  python -m projects.artist.transform
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.conv import ConvBlock, DWConvBlock


class ResidualBlock(nn.Module):
    """Two 3x3 conv blocks (ReLU after the first only) plus a skip connection.
    Channel count in == channel count out. `depthwise=True` uses DWConvBlock
    instead of the full ConvBlock for both."""

    def __init__(self, channels: int, depthwise: bool = False):
        super().__init__()
        block = DWConvBlock if depthwise else ConvBlock
        self.conv1 = block(channels, channels, kernel_size=3, stride=1)
        self.conv2 = block(channels, channels, kernel_size=3, stride=1, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class TransformNet(nn.Module):
    """Maps a [0,1] RGB image to a [0,1] stylised image in one forward pass.
    Fully convolutional: the same weights apply at any resolution or aspect ratio
    (H and W should be multiples of 4). `depthwise=True` trades some quality for
    a much smaller/faster network -- see the module docstring for which layers
    that touches."""

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128), num_res_blocks: int = 5,
                 depthwise: bool = False):
        super().__init__()
        c1, c2, c3 = channels
        block = DWConvBlock if depthwise else ConvBlock
        self.down = nn.Sequential(
            ConvBlock(3, c1, kernel_size=9, stride=1),     # stem: full conv, needs joint RGB mixing
            block(c1, c2, kernel_size=3, stride=2),
            block(c2, c3, kernel_size=3, stride=2),
        )
        self.res = nn.Sequential(*(ResidualBlock(c3, depthwise) for _ in range(num_res_blocks)))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            block(c3, c2, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            block(c2, c1, kernel_size=3, stride=1),
        )
        self.out = ConvBlock(c1, 3, kernel_size=9, stride=1, norm=False, activation=None)  # plain output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        x = self.out(x)
        return (torch.tanh(x) + 1) / 2


def crop_to_multiple(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Crop a CHW/NCHW tensor's H,W down to the nearest multiple of `k`.
    TransformNet needs H and W divisible by its total downsample factor (4, two
    stride-2 stages) for the upsample path to land back on the exact input size;
    a photo of arbitrary size needs this before being fed in (see stylize.py)."""
    h, w = x.shape[-2:]
    return x[..., : h - h % k, : w - w % k]


if __name__ == "__main__":
    net = TransformNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
    assert 1_000_000 < n_params < 2_500_000, f"unexpected param count {n_params:,}"

    x = torch.rand(2, 3, 64, 64)
    y = net(x)
    assert y.shape == x.shape, f"bad shape {tuple(y.shape)}"
    assert y.min() >= 0.0 and y.max() <= 1.0, f"output out of [0,1]: [{y.min():.3f}, {y.max():.3f}]"
    print("ok (shape preserved, output in [0,1])")

    # fully convolutional: non-square input, still a multiple of 4 -> no distortion
    x2 = torch.rand(1, 3, 64, 96)
    y2 = net(x2)
    assert y2.shape == x2.shape, f"bad non-square shape {tuple(y2.shape)}"
    print("ok (non-square input preserved)")

    # gradient reaches the network's weights (this is what train_style.py relies on --
    # the input photo itself needs no gradient, only TransformNet's parameters do)
    net.zero_grad()
    net(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    print("ok (grad reaches weights)")

    small = TransformNet(channels=(16, 32, 64), num_res_blocks=2)
    small_params = sum(p.numel() for p in small.parameters())
    y3 = small(x)
    assert y3.shape == x.shape
    assert small_params < n_params
    print(f"ok (smaller config: {small_params:,} params, shape preserved)")

    dw_net = TransformNet(depthwise=True)
    dw_params = sum(p.numel() for p in dw_net.parameters())
    y4 = dw_net(x)
    assert y4.shape == x.shape, f"bad depthwise shape {tuple(y4.shape)}"
    assert y4.min() >= 0.0 and y4.max() <= 1.0
    assert dw_params < n_params * 0.6, f"expected a big drop, got {dw_params:,} vs {n_params:,}"
    print(f"ok (depthwise=True: {dw_params:,} params vs {n_params:,} plain, shape/[0,1] preserved)")

    odd = torch.rand(1, 3, 65, 99)  # neither dim a multiple of 4
    cropped = crop_to_multiple(odd)
    assert cropped.shape[-2:] == (64, 96), f"bad crop {tuple(cropped.shape[-2:])}"
    assert torch.equal(cropped, odd[:, :, :64, :96])
    assert net(cropped).shape == cropped.shape
    print("ok (crop_to_multiple)")
