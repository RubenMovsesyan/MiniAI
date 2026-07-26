"""PASCAL VOC2012 image loading for YOLO.

JPEGImages -> torch tensors, CHW float in [0,1]. Images vary in size, so a batch
is zero-padded to the batch's max H,W (pad right/bottom; top-left origin kept so
annotation box coords stay valid). Annotation (XML) loading comes later — the
stem (e.g. "2007_000027") is the image<->annotation association.

Run from inside pytorch/:  python -m projects.YOLO.voc
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VOC_DIR = "/home/rubenmovsesyan/external_1/ml_datasets/pascal_voc/VOC2012"


class VocImages(Dataset):
    """Lazily reads JPEGImages from disk. Item -> (stem, CHW float tensor [0,1])."""

    def __init__(self, voc_dir: str | None = None):
        d = Path(voc_dir or os.environ.get("VOC_DIR", VOC_DIR)).expanduser()
        self.img_dir = d / "JPEGImages"
        self.stems = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        if not self.stems:
            raise FileNotFoundError(f"no .jpg under {self.img_dir}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int) -> tuple[str, torch.Tensor]:
        stem = self.stems[i]
        im = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        x = np.asarray(im, dtype=np.float32) / 255.0  # HWC
        return stem, torch.from_numpy(x).permute(2, 0, 1).contiguous()  # CHW


def pad_collate(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    """Zero-pad each image (right/bottom) to the batch's max H,W and stack."""
    stems, imgs = zip(*batch)
    h = max(t.shape[1] for t in imgs)
    w = max(t.shape[2] for t in imgs)
    out = imgs[0].new_zeros(len(imgs), 3, h, w)
    for k, t in enumerate(imgs):
        out[k, :, : t.shape[1], : t.shape[2]] = t
    return list(stems), out


def voc_image_loader(
    voc_dir: str | None = None, batch: int = 16, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        VocImages(voc_dir), batch_size=batch, shuffle=shuffle, collate_fn=pad_collate
    )


if __name__ == "__main__":
    dl = voc_image_loader(batch=4)
    stems, xb = next(iter(dl))
    print(f"images {len(dl.dataset)}  batch {tuple(xb.shape)}  stems {stems}")
    assert xb.shape[0] == 4 and xb.shape[1] == 3, "bad batch shape"
    assert xb.min() >= 0.0 and xb.max() <= 1.0, "not in [0,1]"
    print("ok")
