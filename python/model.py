"""Model definition + run config for the MNIST PyTorch track."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

CONV_CHANNELS = [16, 16]     # output channels per conv layer (chained; each 3x3 + ReLU + MaxPool2d)
HIDDEN_LAYERS = [128, 64, 32]   # width of each dense hidden layer

@dataclass
class Config:
    data_dir: str
    device: str
    epochs: int
    batch: int
    lr: float
    seed: int

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()  # reads python/.env if present
        return cls(
            data_dir=os.environ.get("MNIST_DIR", "~/Downloads/ml_training"),
            device=os.environ.get("DEVICE", "cuda"),
            epochs=int(os.environ.get("EPOCHS", 10)),
            batch=int(os.environ.get("BATCH", 100)),
            lr=float(os.environ.get("LR", 0.01)),
            seed=int(os.environ.get("SEED", 42)),
        )


class Net(nn.Module):
    """conv stack (CONV_CHANNELS) -> dense stack (HIDDEN_LAYERS) -> 10 logits.
    Layer counts and sizes are controlled by the two lists at the top of the file."""

    def __init__(self, cfg: Config):
        super().__init__()

        # Chain a conv layer per entry in CONV_CHANNELS (each 3x3, ReLU + pool in forward).
        self.convs = nn.ModuleList()
        in_ch = 1
        for out_ch in CONV_CHANNELS:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            self.convs.append(conv)
            in_ch = out_ch

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p = 0.5)
        self.flatten = nn.Flatten()

        # Figure out the flattened size by running a dummy image through the conv
        # stack — works for any number of convs / pools, no hand-computing.
        with torch.no_grad():
            prev_layer = self._conv_features(torch.zeros(1, 1, 28, 28)).shape[1]

        self.linears = nn.ModuleList()
        for hidden in HIDDEN_LAYERS:
            lin = nn.Linear(prev_layer, hidden)
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            self.linears.append(lin)
            prev_layer = hidden

        self.output = nn.Linear(prev_layer, 10)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_features(x)
        for idx, layer in enumerate(self.linears):
            if idx == len(self.linears) - 1:
                x = self.dropout(x)
            x = F.relu(layer(x))
        return self.output(x)
