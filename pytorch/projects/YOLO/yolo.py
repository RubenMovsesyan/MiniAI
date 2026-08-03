"""YOLO training entry point — you define the model here, the toolkit does the rest.

Run from inside pytorch/:  python -m projects.YOLO.yolo
"""

import torch
import torch.nn as nn

from modules.conv import Conv, DWConv
from modules.csp import C2PSA, C3k2
from modules.sppf import SPPF
from training.trainer import Trainer

# from projects.YOLO.voc import voc_image_loader   # TODO: wire loader later

WIDTH = 0.5  # width multiple — scale all channels for cheaper compute
CHANNELS = [round(c * WIDTH) for c in (64, 128, 256, 512, 1024)]
NUM_CLASSES = 20  # Pascal VOC


# --- define your model here ---------------------------------------------------


class DetectHead(nn.Module):
    """Per-scale detect head: C3k2 refine + 3x3 Conv, then split into parallel
    classification/regression paths, each a 2x DWConv trunk feeding 1x1 Conv2d
    prediction layers. Output channels are box(4) + obj(1) + cls(num_classes),
    concatenated in that order."""

    def __init__(self, channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.c3k2 = C3k2(channels, channels)
        self.conv = Conv(channels, channels, kernel_size=3, stride=1)
        self.cls_dwconv1 = DWConv(channels, channels, kernel_size=3, activation="silu")
        self.cls_dwconv2 = DWConv(channels, channels, kernel_size=3, activation="silu")
        self.cls_pred = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.reg_dwconv1 = DWConv(channels, channels, kernel_size=3, activation="silu")
        self.reg_dwconv2 = DWConv(channels, channels, kernel_size=3, activation="silu")
        self.reg_pred = nn.Conv2d(channels, 4, kernel_size=1)  # box (l, t, r, b) — decode TBD
        self.obj_pred = nn.Conv2d(channels, 1, kernel_size=1)  # objectness logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.c3k2(x))
        reg = self.reg_dwconv2(self.reg_dwconv1(x))
        box = self.reg_pred(reg)
        obj = self.obj_pred(reg)
        cls = self.cls_pred(self.cls_dwconv2(self.cls_dwconv1(x)))
        return torch.cat([box, obj, cls], dim=1)


def flatten_and_concat(outputs: list[torch.Tensor]) -> torch.Tensor:
    """[B,C,Hi,Wi] per scale -> [B,C,sum(Hi*Wi)] anchor-point predictions, shallow -> deep."""
    b, c = outputs[0].shape[:2]
    return torch.cat([o.reshape(b, c, -1) for o in outputs], dim=2)


class YOLO(nn.Module):
    """YOLO backbone + PAN-FPN neck + detect heads. Stem -> 4 downsample stages -> SPPF ->
    C2PSA, top-down FPN fuse (upsample + concat/add + C3k2), bottom-up PAN fuse (strided
    Conv downsample + concat/add + C3k2). Each PAN tap is then run through its own
    `DetectHead` (C3k2 + 3x3 Conv), and the per-scale outputs are flattened and
    concatenated into a single [B, 4+1+num_classes, total_anchors] tensor.
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

        # bottom-up fuse (PAN): downsample via a strided 3x3 Conv (learned, keeps more
        # detail than plain pooling) instead of the parameterless upsample used above, so
        # it can also change channel count -> each level's fuse just targets its own tap's
        # width, no lookahead trick needed like the FPN pass
        pan_channels = list(reversed(fuse_out))  # shallow -> deep, matches fpn_taps reversed
        self.downsample = nn.ModuleList(
            Conv(c_in, c_out, kernel_size=3, stride=2)
            for c_in, c_out in zip(pan_channels, pan_channels[1:])
        )
        self.pan_fuse = nn.ModuleList(
            C3k2(c_out * mult, c_out) for c_out in pan_channels[1:]
        )

        # detect heads: one per PAN tap, same order (shallow -> deep)
        self.detect_heads = nn.ModuleList(DetectHead(c) for c in pan_channels)

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

        # --- PAN (bottom-up fuse) ----------------------------------------------------
        rev_fpn = list(reversed(fpn_taps))
        x = rev_fpn[0]
        pan_taps = [x]
        for tap, downsample, fuse in zip(rev_fpn[1:], self.downsample, self.pan_fuse):
            x = downsample(x)
            x = torch.cat([x, tap], dim=1) if self.combine == "concat" else x + tap
            x = fuse(x)
            pan_taps.append(x)
        # --- end PAN -------------------------------------------------------------------

        preds = [head(tap) for head, tap in zip(self.detect_heads, pan_taps)]
        return flatten_and_concat(preds)


# TODO: anchor-to-ground-truth target assignment still needed before real training
# (needs the VOC label loader — see projects/YOLO/voc.py)


# --- train --------------------------------------------------------------------
if __name__ == "__main__":
    EXPECTED_SHAPE = (1, 4 + 1 + NUM_CLASSES, 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2)

    model = YOLO()
    out = model(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == EXPECTED_SHAPE, f"bad shape {tuple(out.shape)}"
    print("ok")

    model_add = YOLO(combine="add")
    out = model_add(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == EXPECTED_SHAPE, f"bad shape (add) {tuple(out.shape)}"
    print("ok (add)")

    # train, test = yolo_loaders(batch=...)   # TODO
    tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
    # tr.fit(train, test, epochs=..., save_best="best.onnx")
    print("architecture ->", tr.visualize(input_shape=(1, 3, 640, 640)))
    raise SystemExit("YOLO scaffold — add the loader + head, then wire up fit()")
