"""MNIST loading. IDX -> torch DataLoaders, NCHW (N,1,28,28) in [0,1], int64 labels."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_IDX_DTYPES = {0x08: ">u1", 0x09: ">i1", 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}
_IMAGES = {"train": "train-images.idx3-ubyte", "test": "t10k-images.idx3-ubyte"}
_LABELS = {"train": "train-labels.idx1-ubyte", "test": "t10k-labels.idx1-ubyte"}


def load_idx(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < 4 or raw[0] or raw[1]:
        raise ValueError(f"{path}: bad IDX magic")
    dtype, ndims = raw[2], raw[3]
    if dtype not in _IDX_DTYPES:
        raise ValueError(f"{path}: unknown IDX dtype 0x{dtype:02X}")
    dims = struct.unpack(f">{ndims}I", raw[4 : 4 + 4 * ndims])
    return np.frombuffer(raw[4 + 4 * ndims :], dtype=_IDX_DTYPES[dtype]).reshape(dims)


def _split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = load_idx(data_dir / _IMAGES[split]).astype(np.float32) / 255.0
    y = load_idx(data_dir / _LABELS[split]).astype(np.int64)
    return x.reshape(-1, 1, 28, 28), y


def mnist_loaders(data_dir: str | None = None, batch: int = 100) -> tuple[DataLoader, DataLoader]:
    d = Path(data_dir or os.environ.get("MNIST_DIR", "~/Downloads/ml_training")).expanduser()

    def loader(split: str, shuffle: bool) -> DataLoader:
        x, y = _split(d, split)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    return loader("train", True), loader("test", False)


if __name__ == "__main__":
    tr, te = mnist_loaders()
    xb, yb = next(iter(tr))
    print(f"train batches {len(tr)}  test batches {len(te)}  batch {tuple(xb.shape)} {tuple(yb.shape)}")
