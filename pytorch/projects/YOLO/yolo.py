"""YOLO training entry point — you define the model here, the toolkit does the rest.

Run from inside pytorch/:  python -m projects.YOLO.yolo
"""

import torch.nn as nn

from modules.conv import ConvBNSiLU
from modules.c3k2 import C3k2
from modules.sppf import SPPF
from training.trainer import Trainer

# from projects.YOLO.voc import voc_image_loader   # TODO: wire loader later

WIDTH = 0.5  # width multiple — scale all channels for cheaper compute
CHANNELS = [round(c * WIDTH) for c in (64, 128, 256, 512, 1024)]


# --- define your model here ---------------------------------------------------
# stem: 3 -> 64ch, /2
layers = [ConvBNSiLU(3, CHANNELS[0], kernel_size=3, stride=2)]

# 4 feature extraction layers: ConvBNSiLU (/2, doubles channels) + C3k2
for c_in, c_out in zip(CHANNELS, CHANNELS[1:]):
    layers.append(ConvBNSiLU(c_in, c_out, kernel_size=3, stride=2))
    layers.append(C3k2(c_out, c_out))

# enlarge receptive field after the backbone
layers.append(SPPF(CHANNELS[-1], CHANNELS[-1]))

model = nn.Sequential(*layers)
# TODO: rest of the YOLO net


# --- train --------------------------------------------------------------------
if __name__ == "__main__":
    # train, test = yolo_loaders(batch=...)   # TODO
    tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
    # tr.fit(train, test, epochs=..., save_best="best.onnx")
    # tr.visualize()
    raise SystemExit("YOLO scaffold — add the model + loader, then wire up fit()")
