"""Feed-forward style-transfer TRAINING: fit one TransformNet to one fixed style
image, using a large stream of generic photos (see coco.py) and the same frozen
VGG16 + losses that drive the single-image optimiser in artist.py. The trained
TransformNet -- not VGG -- is what ends up on a phone.

Run from inside pytorch/:
    python -m projects.artist.train_style <style> [--coco-dir DIR] [--epochs N] ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from projects.artist.coco import coco_loader
from projects.artist.images import load_image
from projects.artist.transform import TransformNet
from projects.artist.vgg import CONTENT_LAYERS, STYLE_LAYERS


@dataclass
class Config:
    coco_dir: str | None = None              # falls back to $COCO_DIR (see coco.py)
    style_size: int = 256
    crop_size: int = 256
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-3
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 0.0
    content_layers: tuple[str, ...] = CONTENT_LAYERS
    style_layers: tuple[str, ...] = STYLE_LAYERS
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500               # steps
    preview_path: str | None = None           # a fixed photo re-stylised for progress checks
    preview_every: int = 200                  # steps
    device: str = "cuda"
    log_every: int = 50


class StyleTrainer:
    """Owns the frozen VGG, the fixed style targets, the TransformNet being
    trained, and the optimisation loop over COCO batches."""

    def __init__(self, style: torch.Tensor, cfg: Config = Config()):
        self.cfg = cfg
        self.device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
        # TODO: build self.vgg (VGGFeatures over content_layers+style_layers,
        # .to(self.device)); precompute self.style_grams from `style` once
        # (no_grad, detached -- same idea as StyleTransfer.__init__ in artist.py);
        # build self.net = TransformNet().to(self.device); build
        # self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr).

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """One gradient step over a batch of COCO photos. Returns the four raw
        (pre-weight) loss values."""
        # TODO: forward `batch` through self.net; run both the stylised output
        # and the original (no_grad) batch through self.vgg; content_loss against
        # each photo's own (no_grad) features, style_loss against self.style_grams
        # (the one fixed target, same for every batch), tv_loss on the output;
        # backward, self.opt.step().
        raise NotImplementedError

    def fit(self, loader: DataLoader) -> None:
        """Loop train_step over `loader` for cfg.epochs, logging every
        cfg.log_every steps, checkpointing self.net every cfg.checkpoint_every,
        and (if cfg.preview_path is set) writing a re-stylised preview image
        every cfg.preview_every."""
        raise NotImplementedError


def _parse(argv: list[str]) -> tuple[str, Config]:
    p = argparse.ArgumentParser(prog="train_style", description="train a feed-forward style network")
    p.add_argument("style")
    p.add_argument("--coco-dir", default=Config.coco_dir, dest="coco_dir")
    p.add_argument("--style-size", type=int, default=Config.style_size, dest="style_size")
    p.add_argument("--crop-size", type=int, default=Config.crop_size, dest="crop_size")
    p.add_argument("--batch-size", type=int, default=Config.batch_size, dest="batch_size")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--content-weight", type=float, default=Config.content_weight, dest="content_weight")
    p.add_argument("--style-weight", type=float, default=Config.style_weight, dest="style_weight")
    p.add_argument("--tv-weight", type=float, default=Config.tv_weight, dest="tv_weight")
    p.add_argument("--checkpoint-dir", default=Config.checkpoint_dir, dest="checkpoint_dir")
    p.add_argument("--preview", default=Config.preview_path, dest="preview_path")
    p.add_argument("--device", default=Config.device)
    a = p.parse_args(argv)
    cfg = Config(coco_dir=a.coco_dir, style_size=a.style_size, crop_size=a.crop_size,
                 batch_size=a.batch_size, epochs=a.epochs, lr=a.lr,
                 content_weight=a.content_weight, style_weight=a.style_weight,
                 tv_weight=a.tv_weight, checkpoint_dir=a.checkpoint_dir,
                 preview_path=a.preview_path, device=a.device)
    return a.style, cfg


def main(argv: list[str]) -> None:
    style_path, cfg = _parse(argv)
    style = load_image(style_path, cfg.style_size)
    loader = coco_loader(cfg.coco_dir, crop_size=cfg.crop_size, batch=cfg.batch_size)

    trainer = StyleTrainer(style, cfg)
    trainer.fit(loader)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        raise SystemExit("artist/train_style.py — skeleton, not implemented yet")
