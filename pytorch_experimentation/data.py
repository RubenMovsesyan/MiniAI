"""IDX file loading for the PyTorch track.

Mirrors src/io/idx.cu in NumPy: parse the IDX magic, read big-endian dims,
byte-swap multi-byte elements, normalize pixels to [0, 1]. Labels are kept as
int64 *class indices* (0..9) — NOT one-hot — because torch.nn.CrossEntropyLoss
takes integer class targets, not one-hot vectors. (The C++ engine one-hots
because its loss expects that; here we don't.)

Pure loading only — no model, no training. Run `python data.py` for a self-check.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# IDX dtype byte -> (numpy dtype, element size). Same set as IdxType in idx.cuh.
_IDX_DTYPES = {
    0x08: (">u1", 1),  # unsigned byte
    0x09: (">i1", 1),  # signed byte
    0x0B: (">i2", 2),  # short
    0x0C: (">i4", 4),  # int
    0x0D: (">f4", 4),  # float
    0x0E: (">f8", 8),  # double
}

# MNIST filenames as they sit in the data dir (dotted variant).
IMAGES = {"train": "train-images.idx3-ubyte", "test": "t10k-images.idx3-ubyte"}
LABELS = {"train": "train-labels.idx1-ubyte", "test": "t10k-labels.idx1-ubyte"}


def load_idx(path: str | Path) -> np.ndarray:
    """Load one IDX file into a NumPy array with its native shape."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 4 or raw[0] != 0x00 or raw[1] != 0x00:
        raise ValueError(f"{path}: bad IDX magic (expected 0x00 0x00 prefix)")
    dtype_byte, ndims = raw[2], raw[3]
    if dtype_byte not in _IDX_DTYPES:
        raise ValueError(f"{path}: unknown IDX dtype 0x{dtype_byte:02X}")
    np_dtype, _ = _IDX_DTYPES[dtype_byte]

    # ndims big-endian u32 dims follow the 4 magic bytes.
    dims = struct.unpack(f">{ndims}I", raw[4 : 4 + 4 * ndims])
    header = 4 + 4 * ndims
    # `>`-prefixed dtype makes frombuffer byte-swap big-endian elements for us.
    data = np.frombuffer(raw[header:], dtype=np_dtype)
    return data.reshape(dims)


def load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split -> (X: (N, 28, 28) float32 in [0,1], y: (N,) int64 in 0..9).

    Images stay 2-D — the model flattens them in forward()."""
    data_dir = Path(data_dir).expanduser()
    images = load_idx(data_dir / IMAGES[split])          # (N, 28, 28) uint8
    labels = load_idx(data_dir / LABELS[split])          # (N,) uint8
    x = images.astype(np.float32) / 255.0
    x = x.reshape(-1, 1, 28, 28)
    y = labels.astype(np.int64)
    return x, y


def make_loaders(data_dir: str | Path, batch_size: int):
    """Build train/test torch DataLoaders. Imported lazily so `python data.py`
    (the self-check) runs without torch installed."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def loader(split: str, shuffle: bool) -> DataLoader:
        x, y = load_split(data_dir, split)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader("train", True), loader("test", False)


def _self_check() -> None:
    import os

    data_dir = os.environ.get("MNIST_DIR", "~/Downloads/ml_training")
    xtr, ytr = load_split(data_dir, "train")
    xte, yte = load_split(data_dir, "test")

    assert xtr.shape == (60000, 28, 28), xtr.shape
    assert xte.shape == (10000, 28, 28), xte.shape
    assert ytr.shape == (60000,) and yte.shape == (10000,)
    assert xtr.min() >= 0.0 and xtr.max() <= 1.0, (xtr.min(), xtr.max())
    assert ytr.min() == 0 and ytr.max() == 9, (ytr.min(), ytr.max())
    assert xtr.dtype == np.float32 and ytr.dtype == np.int64
    print(f"OK  train {xtr.shape} test {xte.shape}  "
          f"pixels [{xtr.min():.1f},{xtr.max():.1f}]  labels {ytr.min()}..{ytr.max()}")


if __name__ == "__main__":
    _self_check()
