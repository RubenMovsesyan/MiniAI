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
    """YOLO backbone. Stem -> 4 downsample stages -> SPPF -> C2PSA, returns P5.

    A class rather than an `nn.Sequential` because the PAN neck coming next needs
    the intermediate stage outputs (P3 @/8, P4 @/16), which a straight chain can't
    hand out. `forward` is where those taps will be collected.
    """

    def __init__(self, channels: list[int] = CHANNELS, in_channels: int = 3):
        super().__init__()
        # stem: 3 -> 32ch, /2
        self.stem = Conv(in_channels, channels[0], kernel_size=3, stride=2)

        # 4 feature extraction stages: Conv (/2, doubles channels) + C3k2
        self.stages = nn.ModuleList(
            nn.Sequential(Conv(c_in, c_out, kernel_size=3, stride=2), C3k2(c_out, c_out))
            for c_in, c_out in zip(channels, channels[1:])
        )

        self.sppf = SPPF(channels[-1], channels[-1])  # enlarge receptive field
        self.attn = C2PSA(channels[-1], channels[-1], num_blocks=1)  # self-attention on P5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            # ponytail: neck taps go here — P3 is stages[1] out, P4 is stages[2] out
        return self.attn(self.sppf(x))


# TODO: rest of the YOLO net — neck (PAN-FPN) + detect head


# --- train --------------------------------------------------------------------
if __name__ == "__main__":
    model = YOLO()
    y = model(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, CHANNELS[-1], 2, 2), f"bad shape {tuple(y.shape)}"  # /32
    print("ok")

    # train, test = yolo_loaders(batch=...)   # TODO
    tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
    # tr.fit(train, test, epochs=..., save_best="best.onnx")
    # tr.visualize(input_shape=(1, 3, 640, 640))
    raise SystemExit("YOLO scaffold — add the loader + head, then wire up fit()")
