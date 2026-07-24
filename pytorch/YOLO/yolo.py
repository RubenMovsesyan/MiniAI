"""YOLO training entry point — you define the model here, the toolkit does the rest.

Run from inside pytorch/:  python YOLO/yolo.py
"""

import _lib  # noqa: F401  # puts parent pytorch/ on sys.path — keep first

import torch.nn as nn

from trainer import Trainer

# from voc import voc_image_loader   # TODO: wire loader later

WIDTH = 0.5  # width multiple — scale all channels for cheaper compute
c1 = round(64 * WIDTH)
c2 = round(128 * WIDTH)


# --- define your model here ---------------------------------------------------
model = nn.Sequential(
    # stem: 3 -> 64ch, /2
    nn.Conv2d(3, c1, 3, 2, 1, bias=False),
    nn.BatchNorm2d(c1),
    nn.SiLU(),
    # 64 -> 128ch, /2
    nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
    nn.BatchNorm2d(c2),
    nn.SiLU(),
    # TODO: rest of the YOLO net
)


# --- train --------------------------------------------------------------------
if __name__ == "__main__":
    # train, test = yolo_loaders(batch=...)   # TODO
    tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
    # tr.fit(train, test, epochs=..., save_best="best.onnx")
    # tr.visualize()
    raise SystemExit("YOLO scaffold — add the model + loader, then wire up fit()")
