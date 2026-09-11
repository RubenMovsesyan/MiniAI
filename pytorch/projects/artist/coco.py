"""COCO (or any folder of photos) image loading for feed-forward style-transfer
training. Every training photo doubles as its own content target (see
train_style.py) -- this loader just needs a large, varied stream of generic
photos, cropped to a consistent size so a batch can be stacked.

The on-disk layout of the downloaded dataset isn't settled yet, so scanning and
cropping are left as TODOs; the interface (env var, tensor convention) is fixed.

Run from inside pytorch/:  python -m projects.artist.coco
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class CocoImages(Dataset):
    """Lazily reads photos from `root` (or $COCO_DIR). Item -> CHW float32 tensor
    in [0,1], cropped to `crop_size` x `crop_size` (same tensor convention as
    images.load_image)."""

    def __init__(self, root: str | Path | None = None, crop_size: int = 256):
        self.crop_size = crop_size
        self.root = root  # resolved against $COCO_DIR below
        self.paths: list[Path] = []
        # TODO: resolve self.root (arg, else $COCO_DIR env var, else raise -- no
        # hardcoded default path; the download location isn't settled), then scan
        # it for images. On-disk layout (flat folder vs COCO's train2017/*.jpg
        # split, recursive vs top-level) isn't known yet either.

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        # TODO: open self.paths[i] -> RGB -> resize short side to self.crop_size ->
        # random-crop crop_size x crop_size -> CHW float32 in [0,1].
        raise NotImplementedError


def coco_loader(root: str | Path | None = None, crop_size: int = 256,
                batch: int = 8, shuffle: bool = True) -> DataLoader:
    """Every item is the same crop_size x crop_size shape, so the default
    collate (stack) is enough -- no pad_collate needed like voc.py's
    variable-size images."""
    return DataLoader(CocoImages(root, crop_size), batch_size=batch, shuffle=shuffle)


if __name__ == "__main__":
    raise SystemExit("artist/coco.py — skeleton, not implemented yet")
