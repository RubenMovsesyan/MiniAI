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
import torch.nn.functional as F
from PIL import Image, ImageOps

# PIL's decompression-bomb guard exists to protect against untrusted input (a tiny
# file that decodes into a huge memory footprint). Everything here is a local file
# the user pointed at by path -- e.g. a high-res museum scan used as a style image
# can legitimately be >178M pixels -- so that guard is disabled at import time
# rather than worked around per call site.
Image.MAX_IMAGE_PIXELS = None

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


def resize_longer_side(x: torch.Tensor, size: int) -> torch.Tensor:
    """NCHW tensor -> longer side resized to `size`, aspect preserved -- the
    tensor-space, differentiable equivalent of load_image's own resize. Used to
    judge a loss at a different (typically smaller) scale than the image being
    optimised actually renders at, e.g. to decouple output resolution from
    brush-stroke scale (see artist.Config.eval_size)."""
    h, w = x.shape[-2:]
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)


def to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """[0,1] RGB -> YCbCr (ITU-R BT.601, full range): channel 0 is luminance
    (brightness -- carries almost all perceived texture), 1-2 are chrominance
    (colour), both centred at 0.5. CHW or NCHW."""
    r, g, b = x[..., 0:1, :, :], x[..., 1:2, :, :], x[..., 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=-3)


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_ycbcr`. Not guaranteed in [0,1] if Y and Cb/Cr came from
    different images (an out-of-gamut combination) -- callers that need a
    displayable image (e.g. `preserve_color`) clamp afterward."""
    y, cb, cr = x[..., 0:1, :, :], x[..., 1:2, :, :] - 0.5, x[..., 2:3, :, :] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=-3)


def preserve_color(stylized: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Recombine `stylized`'s luminance (brightness -- carries almost all of the
    perceived brush texture) with `content`'s chrominance (its actual colour),
    via YCbCr. Fixes a style-heavy global palette (e.g. Van Gogh's blue/yellow)
    getting imposed regardless of the content's real colour, while keeping the
    learned texture, which lives almost entirely in luminance, not colour.
    CHW or NCHW, [0,1], both inputs the same spatial shape; output clamped to
    [0,1]."""
    y = to_ycbcr(stylized)[..., 0:1, :, :]
    cbcr = to_ycbcr(content)[..., 1:3, :, :]
    return to_rgb(torch.cat([y, cbcr], dim=-3)).clamp(0.0, 1.0)


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

    x = torch.rand(1, 3, 60, 120, requires_grad=True)
    small = resize_longer_side(x, 64)
    assert small.shape == (1, 3, 32, 64), f"bad shape {tuple(small.shape)}"  # longer side (W) -> 64
    small.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "resize_longer_side must be differentiable"
    print("ok (resize_longer_side: shape, differentiable)")

    # preserve_color: recombining an image with itself is (near) identity
    content = torch.rand(1, 3, 16, 16)
    assert torch.allclose(preserve_color(content, content), content, atol=1e-4), \
        "self-recombination should round-trip to (almost) the same image"
    print("ok (preserve_color: self-recombination round-trips)")

    # a colour-tinted "stylized" image, same luminance pattern as content: recombining
    # should recover content's colour but keep the tint's brightness pattern. Kept
    # gentle (mid-range values, small shift) so the recombined RGB stays in [0,1] --
    # mismatched Y/chrominance CAN legitimately overflow gamut (that's why
    # preserve_color clamps for real use), which would make this exact check flaky.
    content = content * 0.6 + 0.2
    tinted = (content + torch.tensor([0.08, -0.05, 0.06]).view(1, 3, 1, 1)).clamp(0, 1)
    result = preserve_color(tinted, content)
    assert torch.allclose(to_ycbcr(result)[:, 0:1], to_ycbcr(tinted)[:, 0:1], atol=1e-4), \
        "result's luminance should match the stylized image's, not the content's"
    assert torch.allclose(to_ycbcr(result)[:, 1:3], to_ycbcr(content)[:, 1:3], atol=1e-4), \
        "result's colour should match the content image's, not the tinted/stylized one's"
    assert result.shape == content.shape and 0.0 <= result.min() and result.max() <= 1.0
    print("ok (preserve_color: keeps stylized luminance, restores content colour)")
