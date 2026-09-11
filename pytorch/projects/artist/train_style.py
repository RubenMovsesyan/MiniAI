"""Feed-forward style-transfer TRAINING: fit one TransformNet to one fixed style
image, using a large stream of generic photos (see coco.py) and the same frozen
VGG16 + losses that drive the single-image optimiser in artist.py. The trained
TransformNet -- not VGG -- is what ends up on a phone.

Each training photo is fed to TransformNet AND, unmodified, to VGG -- its own
activations are its own content target. The style target (Gram matrices) is
computed once from the style image, before training starts, and never changes.

Run from inside pytorch/:
    python -m projects.artist.train_style <style> [--coco-dir DIR] [--epochs N] ...
    python -m projects.artist.train_style            # no args -> offline self-check
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from projects.artist.coco import coco_loader
from projects.artist.images import load_image, save_image, to_vgg
from projects.artist.losses import content_loss, gram_matrices, style_loss, tv_loss
from projects.artist.transform import TransformNet
from projects.artist.vgg import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures


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
    style_layer_weights: dict[str, float] | None = None
    depthwise: bool = False                  # TransformNet(depthwise=...) -- see transform.py
    pretrained: bool = True                  # False -> random VGG (self-check only)
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500              # steps
    preview_path: str | None = None          # a fixed photo re-stylised for progress checks
    preview_size: int = 512
    preview_every: int = 200                 # steps
    device: str = "cuda"
    log_every: int = 50


class StyleTrainer:
    """Owns the frozen VGG, the fixed style targets, the TransformNet being
    trained, and the optimisation loop over COCO batches."""

    def __init__(self, style: torch.Tensor, cfg: Config = Config()):
        self.cfg = cfg
        self.device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"

        taps = tuple(dict.fromkeys(cfg.content_layers + cfg.style_layers))
        self.vgg = VGGFeatures(taps, pretrained=cfg.pretrained).to(self.device)

        style = to_vgg(style.unsqueeze(0)).to(self.device)
        with torch.no_grad():
            sf = self.vgg(style)
        self.style_grams = {l: g.detach() for l, g in gram_matrices(sf, cfg.style_layers).items()}

        self.net = TransformNet(depthwise=cfg.depthwise).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        self._step = 0
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._preview: torch.Tensor | None = None
        if cfg.preview_path:
            self._preview = load_image(cfg.preview_path, cfg.preview_size).unsqueeze(0).to(self.device)
            (Path(cfg.checkpoint_dir) / "previews").mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """One gradient step over a batch of COCO photos. Returns the four raw
        (pre-weight) loss values."""
        batch = batch.to(self.device)
        out = self.net(batch)  # [0,1], grad-tracked back into self.net's weights

        with torch.no_grad():
            feats_in = self.vgg(to_vgg(batch))  # each photo's own (fixed) content target
        feats_out = self.vgg(to_vgg(out))

        c = content_loss(feats_out, feats_in, self.cfg.content_layers)
        s = style_loss(feats_out, self.style_grams, self.cfg.style_layers, self.cfg.style_layer_weights)
        tv = tv_loss(out)
        total = self.cfg.content_weight * c + self.cfg.style_weight * s + self.cfg.tv_weight * tv

        self.opt.zero_grad()
        total.backward()
        self.opt.step()

        return {"content": c.item(), "style": s.item(), "tv": tv.item(), "total": total.item()}

    def _checkpoint(self, name: str) -> Path:
        path = Path(self.cfg.checkpoint_dir) / name
        torch.save(self.net.state_dict(), path)
        return path

    def _write_preview(self) -> Path:
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._preview)
        self.net.train()
        path = Path(self.cfg.checkpoint_dir) / "previews" / f"step_{self._step:06d}.png"
        return save_image(out[0], path)

    def fit(self, loader: DataLoader) -> None:
        """Loop train_step over `loader` for cfg.epochs, logging every
        cfg.log_every steps, checkpointing every cfg.checkpoint_every, and (if
        cfg.preview_path is set) writing a re-stylised preview every
        cfg.preview_every."""
        cfg = self.cfg
        total_steps = len(loader) * cfg.epochs
        print(f"training: {len(loader.dataset):,} images, {len(loader):,} batches/epoch, "
              f"{cfg.epochs} epoch(s), {total_steps:,} steps total, depthwise={cfg.depthwise}")

        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            for batch in loader:
                self._step += 1
                m = self.train_step(batch)

                if self._step % cfg.log_every == 0 or self._step == total_steps:
                    rate = self._step * loader.batch_size / (time.time() - t0)
                    print(f"epoch {epoch}/{cfg.epochs}  step {self._step}/{total_steps}  "
                          f"total {m['total']:.2f}  content {m['content']:.4f}  "
                          f"style {m['style']:.3e}  tv {m['tv']:.4f}  ({rate:.1f} img/s)")

                if self._step % cfg.checkpoint_every == 0:
                    print(f"  checkpoint -> {self._checkpoint('latest.pt')}")

                if self._preview is not None and self._step % cfg.preview_every == 0:
                    print(f"  preview -> {self._write_preview()}")

        final = self._checkpoint("final.pt")
        print(f"done. final weights -> {final}")


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
    p.add_argument("--depthwise", action="store_true", default=Config.depthwise)
    p.add_argument("--checkpoint-dir", default=Config.checkpoint_dir, dest="checkpoint_dir")
    p.add_argument("--checkpoint-every", type=int, default=Config.checkpoint_every, dest="checkpoint_every")
    p.add_argument("--preview", default=Config.preview_path, dest="preview_path")
    p.add_argument("--preview-size", type=int, default=Config.preview_size, dest="preview_size")
    p.add_argument("--preview-every", type=int, default=Config.preview_every, dest="preview_every")
    p.add_argument("--log-every", type=int, default=Config.log_every, dest="log_every")
    p.add_argument("--device", default=Config.device)
    a = p.parse_args(argv)
    cfg = Config(coco_dir=a.coco_dir, style_size=a.style_size, crop_size=a.crop_size,
                 batch_size=a.batch_size, epochs=a.epochs, lr=a.lr,
                 content_weight=a.content_weight, style_weight=a.style_weight,
                 tv_weight=a.tv_weight, depthwise=a.depthwise,
                 checkpoint_dir=a.checkpoint_dir, checkpoint_every=a.checkpoint_every,
                 preview_path=a.preview_path, preview_size=a.preview_size,
                 preview_every=a.preview_every, log_every=a.log_every, device=a.device)
    return a.style, cfg


def main(argv: list[str]) -> None:
    style_path, cfg = _parse(argv)
    style = load_image(style_path, cfg.style_size)
    loader = coco_loader(cfg.coco_dir, crop_size=cfg.crop_size, batch=cfg.batch_size)

    trainer = StyleTrainer(style, cfg)
    print(f"device {trainer.device}  net params {sum(p.numel() for p in trainer.net.parameters()):,}  "
          f"lr {cfg.lr}  cw={cfg.content_weight} sw={cfg.style_weight:g} tv={cfg.tv_weight}")
    trainer.fit(loader)


def _selfcheck() -> None:
    import tempfile

    import numpy as np
    from PIL import Image

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        photos = Path(d) / "photos"
        photos.mkdir()
        for i in range(6):
            Image.fromarray(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)).save(photos / f"{i}.jpg")
        style_path = Path(d) / "style.jpg"
        Image.fromarray(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)).save(style_path)

        style = load_image(style_path, 64)
        loader = coco_loader(str(photos), crop_size=64, batch=2, num_workers=0)

        # train_step: sane (non-NaN, non-negative) losses, gradient reaches the net's weights
        trainer = StyleTrainer(style, Config(crop_size=64, device="cpu", pretrained=False))
        m = trainer.train_step(next(iter(loader)))
        assert all(not math.isnan(v) and v >= 0 for v in m.values()), f"bad losses: {m}"
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.net.parameters())
        print(f"ok (train_step: {m})")

        # fit: full loop, checkpointing, preview writing
        cfg = Config(crop_size=64, batch_size=2, epochs=1, device="cpu", pretrained=False,
                     checkpoint_dir=str(Path(d) / "ckpt"), checkpoint_every=2,
                     preview_path=str(style_path), preview_size=64, preview_every=2, log_every=1)
        trainer2 = StyleTrainer(style, cfg)
        trainer2.fit(loader)

        assert trainer2._step == len(loader), f"expected {len(loader)} steps, got {trainer2._step}"
        assert (Path(cfg.checkpoint_dir) / "final.pt").exists(), "final checkpoint missing"
        assert (Path(cfg.checkpoint_dir) / "latest.pt").exists(), "periodic checkpoint missing"
        previews = list((Path(cfg.checkpoint_dir) / "previews").glob("*.png"))
        assert previews, "no preview written"
        print(f"ok (fit: {trainer2._step} steps, checkpoints + {len(previews)} preview(s) written)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _selfcheck()
