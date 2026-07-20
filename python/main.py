"""MNIST PyTorch track — entry point. Train, report stats, then visualize.

    python main.py

The pieces live in their own modules:
  model.py       Config + Net
  train.py       the training loop + evaluation (the meat)
  data.py        IDX -> torch loaders
  visualizer.py  correct/incorrect prediction viewer
"""

from __future__ import annotations

from model import Config
from train import train
from visualizer import show


def main() -> None:
    cfg = Config.from_env()
    model = train(cfg)      # trains, prints per-epoch loss + accuracy
    show(model, cfg)        # opens the correct/incorrect viewer


if __name__ == "__main__":
    main()
