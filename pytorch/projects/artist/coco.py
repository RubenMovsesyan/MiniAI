"""COCO (or any folder of photos) image loading for feed-forward style-transfer
training. Every training photo doubles as its own content target (see
train_style.py) -- this loader just needs a large, varied stream of generic
photos, cropped to a consistent size so a batch can be stacked.

Point it at a folder of images directly (e.g. .../coco/images/train2017), not a
COCO root -- this loader is deliberately COCO-agnostic, no annotations involved.

Run from inside pytorch/:  python -m projects.artist.coco
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

_EXTENSIONS = (".jpg", ".jpeg", ".png")


class CocoImages(Dataset):
    """Lazily reads photos from `root` (or $COCO_DIR). Item -> CHW float32 tensor
    in [0,1], resized short-side-to-crop_size then randomly cropped to
    crop_size x crop_size (same tensor convention as images.load_image)."""

    def __init__(self, root: str | Path | None = None, crop_size: int = 256):
        self.crop_size = crop_size

        resolved = root or os.environ.get("COCO_DIR")
        if not resolved:
            raise ValueError("no dataset root given: pass root=, or set $COCO_DIR")
        self.root = Path(resolved).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"COCO root not found: {self.root}")

        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in _EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"no images ({', '.join(_EXTENSIONS)}) under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = Image.open(self.paths[i])
        im = ImageOps.exif_transpose(im).convert("RGB")

        w, h = im.size
        scale = self.crop_size / min(w, h)
        im = im.resize((max(self.crop_size, round(w * scale)), max(self.crop_size, round(h * scale))),
                        Image.Resampling.LANCZOS)

        w, h = im.size
        x0 = random.randint(0, w - self.crop_size)
        y0 = random.randint(0, h - self.crop_size)
        im = im.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))

        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC in [0,1]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def coco_loader(root: str | Path | None = None, crop_size: int = 256, batch: int = 8,
                 shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Every item is the same crop_size x crop_size shape, so the default
    collate (stack) is enough -- no pad_collate needed like voc.py's
    variable-size images. `drop_last=True` so every training batch is full-size."""
    return DataLoader(CocoImages(root, crop_size), batch_size=batch, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        nested = Path(d) / "nested"
        nested.mkdir()
        Image.fromarray(np.random.randint(0, 256, (300, 500, 3), dtype=np.uint8)).save(Path(d) / "wide.jpg")
        Image.fromarray(np.random.randint(0, 256, (500, 300, 3), dtype=np.uint8)).save(nested / "tall.png")
        Image.fromarray(np.random.randint(0, 256, (128, 128), dtype=np.uint8), mode="L").save(nested / "gray.jpg")
        (Path(d) / "notes.txt").write_text("not an image")

        ds = CocoImages(d, crop_size=64)
        assert len(ds) == 3, f"expected 3 images (recursive scan, non-images ignored), found {len(ds)}"
        for i in range(len(ds)):
            x = ds[i]
            assert x.shape == (3, 64, 64), f"bad shape {tuple(x.shape)}"
            assert x.dtype == torch.float32 and 0.0 <= x.min() and x.max() <= 1.0
        print("ok (recursive scan, ignores non-images, converts grayscale, crop shape/range)")

        dl = coco_loader(d, crop_size=64, batch=2, shuffle=True, num_workers=0)
        xb = next(iter(dl))
        assert tuple(xb.shape) == (2, 3, 64, 64), tuple(xb.shape)
        print("ok (coco_loader batches, drop_last)")

        try:
            CocoImages(str(Path(d) / "missing"))
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            print("ok (missing root raises)")

    saved = os.environ.pop("COCO_DIR", None)
    try:
        CocoImages()
        raise AssertionError("expected ValueError")
    except ValueError:
        print("ok (no root/env raises)")
    finally:
        if saved is not None:
            os.environ["COCO_DIR"] = saved

    real = os.environ.get("COCO_DIR")
    if real:
        ds = CocoImages(real, crop_size=256)
        print(f"$COCO_DIR: {len(ds)} images found, sample crop shape {tuple(ds[0].shape)}")
    else:
        print("skip (real dataset): set $COCO_DIR to also smoke-test against it")
