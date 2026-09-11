"""Image I/O + VGG preprocessing for the style-transfer project.

    load_image  -> CHW float tensor in [0,1], longer side resized to `size`
    to_vgg      -> ImageNet mean/std normalisation (what VGG16 was trained on)
    from_vgg    -> inverse of to_vgg (pure; no clamp)
    save_image  -> write a CHW/1CHW tensor to disk as an image (clamps to [0,1])

The generated image is optimised in VGG-normalised space, so `to_vgg` is applied
once at init (on the content, the style, and the seed) and `from_vgg` only when an
image leaves the pipeline. Out-of-range pixels from the optimiser are clamped in
`save_image`, not mid-optimisation.

Run from inside pytorch/:  python -m projects.artist.images
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

# vgg16.tv_in1k pretrained_cfg (weights/vgg16.tv_in1k/config.json)
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)


def _stats(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std) as [3,1,1] tensors on `ref`'s device/dtype, ready to broadcast."""
    mean = ref.new_tensor(VGG_MEAN).view(3, 1, 1)
    std = ref.new_tensor(VGG_STD).view(3, 1, 1)
    return mean, std


def load_image(path: str | Path, size: int = 512) -> torch.Tensor:
    """Read an image file -> CHW float32 tensor in [0,1], on CPU, no batch dim.

    The longer side is scaled to exactly `size` (up or down) and the aspect ratio
    is preserved. EXIF orientation is applied; any mode is coerced to RGB.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")

    w, h = im.size
    scale = size / max(w, h)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

    x = np.asarray(im, dtype=np.float32) / 255.0  # HWC in [0,1]
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()  # CHW


def to_vgg(x: torch.Tensor) -> torch.Tensor:
    """[0,1] CHW or NCHW -> VGG-normalised: (x - mean) / std."""
    mean, std = _stats(x)
    return (x - mean) / std


def from_vgg(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_vgg`: x * std + mean. Pure — does not clamp to [0,1]."""
    mean, std = _stats(x)
    return x * std + mean


def save_image(x: torch.Tensor, path: str | Path) -> Path:
    """Write a CHW (or 1CHW) tensor to `path` as an image. Clamps to [0,1] first;
    the file format follows the extension. Returns the written path."""
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(f"expected a single image, got batch of {x.shape[0]}")
        x = x[0]
    if x.dim() != 3:
        raise ValueError(f"expected CHW or 1CHW, got shape {tuple(x.shape)}")

    x = x.detach().cpu().float().clamp(0.0, 1.0)
    arr = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()  # HWC

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p)
    return p


if __name__ == "__main__":
    import tempfile

    # to_vgg / from_vgg round-trip, CHW and NCHW
    for shape in [(3, 16, 16), (2, 3, 8, 8)]:
        x = torch.rand(shape)
        assert torch.allclose(from_vgg(to_vgg(x)), x, atol=1e-6), f"round-trip {shape}"
    print("ok (normalise round-trip)")

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.png"
        Image.fromarray(np.random.randint(0, 256, (60, 120, 3), dtype=np.uint8)).save(src)  # HxW = 60x120

        img = load_image(src, size=64)
        assert img.shape == (3, 32, 64), f"bad shape {tuple(img.shape)}"       # longer side (W) -> 64
        assert img.dtype == torch.float32 and 0.0 <= img.min() and img.max() <= 1.0
        print("ok (load_image)")

        out = save_image(img, Path(d) / "nested" / "out.png")
        assert out.exists()
        back = np.asarray(Image.open(out), dtype=np.float32) / 255.0
        back = torch.from_numpy(back).permute(2, 0, 1)
        assert torch.allclose(back, img, atol=1.5 / 255), "save/reload drift"
        print("ok (save_image)")

        save_image(from_vgg(to_vgg(img)) * 3 - 1, Path(d) / "clamped.png")  # out-of-range in -> no error
        print("ok (save_image clamps)")
