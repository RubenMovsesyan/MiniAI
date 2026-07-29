"""YOLO training entry point — you define the model here, the toolkit does the rest.

Run from inside pytorch/:  python -m projects.YOLO.yolo
"""

import torch
import torch.nn as nn

from modules.conv import Conv
from modules.csp import C2PSA, C3k2
from modules.sppf import SPPF
from training.trainer import Trainer

# from projects.YOLO.voc import voc_image_loader   # TODO: wire loader later

WIDTH = 0.5  # width multiple — scale all channels for cheaper compute
CHANNELS = [round(c * WIDTH) for c in (64, 128, 256, 512, 1024)]


# --- define your model here ---------------------------------------------------


class YOLO(nn.Module):
    """YOLO backbone + top-down FPN fuse. Stem -> 4 downsample stages -> SPPF -> C2PSA,
    then each stage's C3k2 output is fused back in deepest-to-shallowest with 2x nearest
    upsampling, combined via `combine` ("concat" or "add"), and re-fused with a C3k2 per
    level. Returns a single tensor at the shallowest tap's resolution (stride 4).

    Bottom-up (PAN) fusion and the detect head are still TODO.
    """

    def __init__(self, channels: list[int] = CHANNELS, in_channels: int = 3, combine: str = "concat"):
        super().__init__()
        if combine not in ("concat", "add"):
            raise ValueError(f"combine must be 'concat' or 'add', got {combine!r}")
        self.combine = combine

        # stem: 3 -> 32ch, /2
        self.stem = Conv(in_channels, channels[0], kernel_size=3, stride=2)

        # 4 feature extraction stages: Conv (/2, doubles channels) + C3k2
        self.stages = nn.ModuleList(
            nn.Sequential(Conv(c_in, c_out, kernel_size=3, stride=2), C3k2(c_out, c_out))
            for c_in, c_out in zip(channels, channels[1:])
        )

        self.sppf = SPPF(channels[-1], channels[-1])  # enlarge receptive field
        self.attn = C2PSA(channels[-1], channels[-1], num_blocks=1)  # self-attention on P5

        # top-down fuse: deepest tap first, each fuse block outputs the next (shallower)
        # tap's channel count so add-mode channels line up with zero extra projection convs
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        tap_channels = channels[1:]
        rev = list(reversed(tap_channels))
        fuse_out = rev[1:] + [rev[-1]]  # last step is terminal, stays at its own width
        mult = 2 if combine == "concat" else 1
        self.fuse = nn.ModuleList(
            C3k2(c_in * mult, c_out) for c_in, c_out in zip(rev, fuse_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        taps = []
        for stage in self.stages:
            x = stage(x)
            taps.append(x)

        x = self.attn(self.sppf(x))

        # --- FPN (top-down fuse) ---------------------------------------------------
        fpn_taps = []
        for i, (tap, fuse) in enumerate(zip(reversed(taps), self.fuse)):
            if i > 0:
                x = self.upsample(x)
            x = torch.cat([x, tap], dim=1) if self.combine == "concat" else x + tap
            x = fuse(x)
            fpn_taps.append(x)
        # --- end FPN -----------------------------------------------------------------

        return x


# TODO: rest of the YOLO net — neck (PAN-FPN) + detect head


# --- train --------------------------------------------------------------------
if __name__ == "__main__":
    model = YOLO()
    y = model(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, CHANNELS[1], 16, 16), f"bad shape {tuple(y.shape)}"  # /4
    print("ok")

    model_add = YOLO(combine="add")
    y = model_add(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, CHANNELS[1], 16, 16), f"bad shape (add) {tuple(y.shape)}"
    print("ok (add)")

    # train, test = yolo_loaders(batch=...)   # TODO
    tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
    # tr.fit(train, test, epochs=..., save_best="best.onnx")
    # tr.visualize(input_shape=(1, 3, 640, 640))
    raise SystemExit("YOLO scaffold — add the loader + head, then wire up fit()")
