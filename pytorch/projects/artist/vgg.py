"""VGG16 truncated to its conv stack, used as a *frozen* feature extractor.

Not a model we train — style transfer optimises an image, not this network. We
rebuild torchvision's VGG16 "features" (config D) in code, pour in the pretrained
ImageNet weights from the `weights/vgg16.tv_in1k` submodule, freeze them, and read
out named intermediate activations for the content / style losses.

Input is expected already VGG-normalised (see images.to_vgg) — normalisation
happens once in artist.py, not here.

Run from inside pytorch/:  python -m projects.artist.vgg
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

WEIGHTS_DIR = Path(__file__).parent / "weights" / "vgg16.tv_in1k"
WEIGHTS_SAFETENSORS = WEIGHTS_DIR / "model.safetensors"

_PULL_HINT = (
    f"{WEIGHTS_SAFETENSORS} is missing or still a Git-LFS pointer.\n"
    "Fetch the real weights:\n"
    "  git submodule update --init pytorch/projects/artist/weights/vgg16.tv_in1k\n"
    "  cd pytorch/projects/artist/weights/vgg16.tv_in1k && git lfs pull --include model.safetensors"
)

# torchvision VGG16 conv stack (config "D"): ints are conv out-channels, "M" a 2x2 max-pool.
_CFG: list[int | str] = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]

# Gatys et al. layer names -> index of that conv's ReLU output in the built Sequential.
# (Conv, ReLU, Conv, ReLU, MaxPool, ...) — matches the `features.N` keys in the weights.
LAYER_INDEX = {
    "conv1_1": 1,  "conv1_2": 3,
    "conv2_1": 6,  "conv2_2": 8,
    "conv3_1": 11, "conv3_2": 13, "conv3_3": 15,
    "conv4_1": 18, "conv4_2": 20, "conv4_3": 22,
    "conv5_1": 25, "conv5_2": 27, "conv5_3": 29,
}

# Gatys defaults: one deep layer for content, five "conv_x_1" layers for style.
CONTENT_LAYERS = ("conv4_2",)
STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")


def _build_features(pool: str = "max") -> nn.Sequential:
    """Assemble the conv/ReLU/pool stack from `_CFG`. Module index N lines up with
    the `features.N` weight keys. `pool` is "max" (pretrained default) or "avg"
    (Gatys' smoother variant — pools carry no weights, so either loads fine)."""
    if pool not in ("max", "avg"):
        raise ValueError(f"pool must be 'max' or 'avg', got {pool!r}")
    make_pool = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
    layers: list[nn.Module] = []
    in_ch = 3
    for c in _CFG:
        if c == "M":
            layers.append(make_pool(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_ch, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=False))
            in_ch = c
    return nn.Sequential(*layers)


def load_features_state_dict() -> dict[str, torch.Tensor]:
    """Read the submodule weights -> the `features.*` subset, keyed to match a
    `VGGFeatures` instance (classifier keys `head.*` / `pre_logits.*` dropped)."""
    if not WEIGHTS_SAFETENSORS.exists() or WEIGHTS_SAFETENSORS.stat().st_size < 1024:
        raise FileNotFoundError(_PULL_HINT)
    full = load_file(WEIGHTS_SAFETENSORS)
    return {k: v for k, v in full.items() if k.startswith("features.")}


class VGGFeatures(nn.Module):
    """Frozen VGG16 conv stack. `forward(x)` -> `{layer_name: activation}` for the
    requested layers. `x` must already be VGG-normalised NCHW."""

    def __init__(self, layers: tuple[str, ...] = CONTENT_LAYERS + STYLE_LAYERS,
                 pretrained: bool = True, pool: str = "max"):
        super().__init__()
        unknown = set(layers) - LAYER_INDEX.keys()
        if unknown:
            raise ValueError(f"unknown layer(s): {sorted(unknown)}; pick from {sorted(LAYER_INDEX)}")

        self.features = _build_features(pool)
        if pretrained:
            self.load_state_dict(load_features_state_dict(), strict=True)

        self.layers = tuple(layers)
        self._tap_names = {LAYER_INDEX[name]: name for name in self.layers}  # idx -> name
        self._last = max(self._tap_names)                                    # deepest tap

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._tap_names:
                out[self._tap_names[i]] = x
            if i == self._last:
                break
        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    vgg = VGGFeatures(pretrained=False)
    taps = vgg(torch.randn(1, 3, 64, 64))
    assert set(taps) == set(CONTENT_LAYERS + STYLE_LAYERS), sorted(taps)
    want = {"conv1_1": (1, 64, 64, 64), "conv2_1": (1, 128, 32, 32),
            "conv3_1": (1, 256, 16, 16), "conv4_1": (1, 512, 8, 8),
            "conv4_2": (1, 512, 8, 8), "conv5_1": (1, 512, 4, 4)}
    for name, shape in want.items():
        assert tuple(taps[name].shape) == shape, f"{name}: {tuple(taps[name].shape)} != {shape}"
    assert all(not p.requires_grad for p in vgg.parameters()), "params not frozen"
    print("ok (architecture, shapes, frozen)")

    x = torch.randn(1, 3, 48, 48, requires_grad=True)
    VGGFeatures(("conv4_2",), pretrained=False)(x)["conv4_2"].pow(2).mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "no grad to input"
    assert VGGFeatures(("conv1_1",), pretrained=False)._last == 1, "early-stop index wrong"
    print("ok (grad to image, early-stop)")

    if WEIGHTS_SAFETENSORS.exists() and WEIGHTS_SAFETENSORS.stat().st_size >= 1024:
        w = VGGFeatures(pretrained=True).features[0].weight
        assert 0.01 < w.std().item() < 1.0, f"suspicious first-conv std {w.std().item():.4f}"
        print(f"ok (pretrained weights loaded, features.0.weight std {w.std().item():.3f})")
    else:
        print("skip (pretrained): run `git lfs pull` in weights/vgg16.tv_in1k")
