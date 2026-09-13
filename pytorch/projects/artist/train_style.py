"""Feed-forward style-transfer TRAINING: fit one TransformNet to one or more
fixed style images (their common style, not any one of them), using a large
stream of generic photos (see coco.py) and the same frozen VGG16 + losses that
drive the single-image optimiser in artist.py. The trained TransformNet -- not
VGG -- is what ends up on a phone.

Each training photo is fed to TransformNet AND, unmodified, to VGG -- its own
activations are its own content target. The style targets (Gram matrices, one
per style image, averaged) are computed once before training starts and never
change. `eval_size` mirrors artist.py's multi-scale mechanism: content/style/
color are all judged at one or more scales of the training crop (summed),
while TransformNet's own output resolution is whatever the crop size is -- a
coarse scale sets big stroke placement, a finer scale forces real texture at
that finer scale too, instead of one scale doing both jobs. Note: unlike
artist.py, `crop_size` must be >= every value in `eval_size` -- there IS no
separate "output resolution" to borrow real detail from here, only the
training crop itself.

`--color-weight` pulls the network's output colour toward its own source
photo's -- judged at every eval_size scale, same as content/style (a high
weight judged only at the native resolution let it correct each pixel's
colour almost independently of its neighbours, breaking coherent brush
strokes into a bubbly/cellular texture; see artist.py's `Config.color_weight`
docstring and README.md's colour-imposition writeup for the full story).

`--resume <checkpoint.pt>` warm-starts TransformNet's weights from an earlier
run (e.g. to re-tune the content/style balance without training from scratch) --
only the weights carry over, the optimizer (Adam's momentum) always starts fresh.

If training blows up (a non-finite loss or gradient -- seen in practice with an
overly aggressive style_weight), fit() stops immediately rather than running the
remaining epochs, and final.pt is the last state before the blow-up, not garbage.

Run from inside pytorch/:
    python -m projects.artist.train_style <style> [<style2> ...] [--coco-dir DIR] ...
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
from projects.artist.images import load_image, resize_longer_side, save_image, to_vgg
from projects.artist.losses import color_loss, content_loss, gram_matrices, style_loss, tv_loss
from projects.artist.transform import TransformNet
from projects.artist.vgg import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures


@dataclass
class Config:
    coco_dir: str | None = None              # falls back to $COCO_DIR (see coco.py)
    style_size: int = 256
    crop_size: int = 256                     # must be >= max(eval_size) -- see module docstring
    eval_size: tuple[int, ...] = ()          # () = judge once, at crop_size (current behaviour).
                                              # One or more scales sums their content/style loss,
                                              # judged on TransformNet's own output -- see train_step.
    eval_size_weights: tuple[float, ...] | None = None  # per-scale weight (default 1.0 each)
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-3
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 0.0
    color_weight: float = 0.0                # 0 = off; pulls the output's colour toward its own
                                              # source photo's -- see losses.color_loss and
                                              # artist.Config.color_weight (same mechanism, validated
                                              # there first since a single-image run is much cheaper)
    content_layers: tuple[str, ...] = CONTENT_LAYERS
    style_layers: tuple[str, ...] = STYLE_LAYERS
    style_layer_weights: dict[str, float] | None = None
    depthwise: bool = False                  # TransformNet(depthwise=...) -- see transform.py
    pretrained: bool = True                  # False -> random VGG (self-check only)
    resume: str | None = None                # warm-start from a saved checkpoint's weights only (fresh
                                              # optimizer, fresh step counter) -- for re-tuning, e.g. a
                                              # different style_weight, not for resuming an interrupted run
    continue_from: str | None = None         # fully resume an interrupted run: weights + optimizer state
                                              # + step counter, all restored, so training picks back up as
                                              # if it had never stopped (needs a checkpoint saved by THIS
                                              # code -- see _checkpoint -- not an old bare state_dict file)
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

    def __init__(self, styles: list[torch.Tensor], cfg: Config = Config()):
        if not styles:
            raise ValueError("styles must be a non-empty list of images")
        if cfg.eval_size and max(cfg.eval_size) > cfg.crop_size:
            raise ValueError(f"eval_size {cfg.eval_size} exceeds crop_size {cfg.crop_size} -- "
                              "there's no extra real detail to judge beyond what the crop contains")
        self.cfg = cfg
        self.device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"

        taps = tuple(dict.fromkeys(cfg.content_layers + cfg.style_layers))
        self.vgg = VGGFeatures(taps, pretrained=cfg.pretrained).to(self.device)

        # one or more scales of the training crop get judged (summed) each step --
        # see train_step. Style targets are fixed for the whole run, so build one
        # per scale here, once: each style image's Gram matrices computed
        # independently, then averaged per layer into ONE target per scale -- the
        # "common style" across all of them (same idea as artist.StyleTransfer).
        self._scales = cfg.eval_size if cfg.eval_size else (cfg.crop_size,)
        self.style_grams: list[dict[str, torch.Tensor]] = []
        for scale in self._scales:
            per_image_grams = []
            for style in styles:
                s = self._at_scale(to_vgg(style.unsqueeze(0)).to(self.device), scale)
                with torch.no_grad():
                    sf = self.vgg(s)
                per_image_grams.append(gram_matrices(sf, cfg.style_layers))
            self.style_grams.append({
                l: torch.stack([g[l] for g in per_image_grams], dim=0).mean(0).detach()
                for l in cfg.style_layers
            })

        if cfg.resume and cfg.continue_from:
            raise ValueError("pass either --resume or --continue-from, not both")

        self.net = TransformNet(depthwise=cfg.depthwise).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self._step = 0

        if cfg.resume:
            ckpt = torch.load(cfg.resume, map_location=self.device)
            # checkpoints saved by _checkpoint() are {"net":..., "opt":..., "step":...};
            # older files (before continue_from existed) are a bare state_dict -- accept both
            state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
            self.net.load_state_dict(state)
            print(f"resumed weights from {cfg.resume} (optimizer state starts fresh)")

        if cfg.continue_from:
            ckpt = torch.load(cfg.continue_from, map_location=self.device)
            if not (isinstance(ckpt, dict) and "net" in ckpt and "opt" in ckpt and "step" in ckpt):
                raise ValueError(f"{cfg.continue_from} isn't a full training checkpoint (no optimizer "
                                  "state/step count) -- use --resume instead to warm-start weights only")
            self.net.load_state_dict(ckpt["net"])
            self.opt.load_state_dict(ckpt["opt"])
            self._step = ckpt["step"]
            print(f"continuing from {cfg.continue_from} at step {self._step:,} "
                  "(weights + optimizer state + step counter all restored)")

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._preview: torch.Tensor | None = None
        if cfg.preview_path:
            self._preview = load_image(cfg.preview_path, cfg.preview_size).unsqueeze(0).to(self.device)
            (Path(cfg.checkpoint_dir) / "previews").mkdir(parents=True, exist_ok=True)

        self.step_ok = True  # set by train_step; False means the optimizer step was skipped

    def _at_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Resize to `scale` (longer side), or return `x` unchanged if it's
        already that size -- skips a needless identity resize."""
        return x if max(x.shape[-2:]) == scale else resize_longer_side(x, scale)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """One gradient step over a batch of COCO photos. Returns the four raw
        (pre-weight) loss values.

        Content/style/color are all judged at every scale in self._scales
        (summed) -- each photo is its own content AND colour target at every
        scale, freshly, since (unlike the style images) the content changes
        every batch and can't be precomputed. color_loss reuses `batch_s`/
        `out_s` (already resized for content/style) rather than resizing a
        second time. Judging color_loss at only the native resolution (the
        original approach) let it correct each pixel's colour almost
        independently of its neighbours at a high color_weight, breaking
        coherent brush strokes into a bubbly/cellular texture -- the same
        artifact, and the same fix, as artist.py's StyleTransfer._closure.
        tv_loss stays on the network's raw, native-resolution output,
        unaffected by eval_size (matching artist.py, where tv_loss also
        isn't judged per-scale).

        If the loss, or its gradients, come out non-finite (nan/inf -- a training
        blow-up), the optimiser step is skipped entirely and `self.step_ok` is set
        to False. Because the check happens before `opt.step()`, self.net's
        weights are left exactly as they were at the end of the last *successful*
        step -- there's nothing to roll back, the current state already is "the
        last good value" (see fit())."""
        batch = batch.to(self.device)
        out = self.net(batch)  # [0,1], grad-tracked back into self.net's weights

        c = s = color = out.new_zeros(())
        for i, (scale, style_gram) in enumerate(zip(self._scales, self.style_grams)):
            w = 1.0 if self.cfg.eval_size_weights is None else self.cfg.eval_size_weights[i]
            batch_s, out_s = self._at_scale(batch, scale), self._at_scale(out, scale)
            with torch.no_grad():
                feats_in = self.vgg(to_vgg(batch_s))  # each photo's own (fixed) content target
            feats_out = self.vgg(to_vgg(out_s))
            c = c + w * content_loss(feats_out, feats_in, self.cfg.content_layers)
            s = s + w * style_loss(feats_out, style_gram, self.cfg.style_layers, self.cfg.style_layer_weights)
            color = color + w * color_loss(out_s, batch_s)

        tv = tv_loss(out)
        total = (self.cfg.content_weight * c + self.cfg.style_weight * s
                 + self.cfg.tv_weight * tv + self.cfg.color_weight * color)

        m = {"content": c.item(), "style": s.item(), "tv": tv.item(),
             "color": color.item(), "total": total.item()}
        self.step_ok = all(math.isfinite(v) for v in m.values())
        if not self.step_ok:
            return m  # blown up before backward() even ran -- nothing to step, nothing to undo

        self.opt.zero_grad()
        total.backward()
        self.step_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in self.net.parameters())
        if not self.step_ok:
            self.opt.zero_grad()  # drop the bad gradients so they can't leak into a later step
            return m

        self.opt.step()
        return m

    def _checkpoint(self, name: str) -> Path:
        """Save weights + optimizer state + step counter, so --continue-from can
        resume this exact run later (see __init__) -- not just its weights."""
        path = Path(self.cfg.checkpoint_dir) / name
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "step": self._step}, path)
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
        cfg.preview_every.

        Stops early -- before completing all epochs -- if a step's loss or
        gradients come out non-finite, and saves final.pt from the weights as of
        the last successful step (train_step never applies a corrupting update,
        so there's no separate "last good" state to track: self.net already is
        it).

        `cfg.epochs` always means "epochs this call will run" -- if self._step
        is already nonzero (continuing an earlier session via --continue-from),
        it keeps counting up from there rather than restarting at 0, but the
        loop below still runs exactly cfg.epochs more epochs; pass however many
        epochs you want left, not the original total."""
        cfg = self.cfg
        step_at_start = self._step
        total_steps = step_at_start + len(loader) * cfg.epochs
        session_note = f" (continuing from step {step_at_start:,})" if step_at_start else ""
        print(f"training: {len(loader.dataset):,} images, {len(loader):,} batches/epoch, "
              f"{cfg.epochs} epoch(s), {total_steps - step_at_start:,} steps this session{session_note}, "
              f"depthwise={cfg.depthwise}")

        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            for batch in loader:
                self._step += 1
                m = self.train_step(batch)

                if not self.step_ok:
                    print(f"non-finite loss/gradients at step {self._step} ({m}) -- "
                          f"stopping early, using the weights from step {self._step - 1} as final")
                    final = self._checkpoint("final.pt")
                    print(f"final weights -> {final}")
                    return

                if self._step % cfg.log_every == 0 or self._step == total_steps:
                    rate = (self._step - step_at_start) * loader.batch_size / (time.time() - t0)
                    print(f"epoch {epoch}/{cfg.epochs}  step {self._step:,}/{total_steps:,}  "
                          f"total {m['total']:.2f}  content {m['content']:.4f}  "
                          f"style {m['style']:.3e}  tv {m['tv']:.4f}  color {m['color']:.4f}  "
                          f"({rate:.1f} img/s)")

                if self._step % cfg.checkpoint_every == 0:
                    print(f"  checkpoint -> {self._checkpoint('latest.pt')}")

                if self._preview is not None and self._step % cfg.preview_every == 0:
                    print(f"  preview -> {self._write_preview()}")

        final = self._checkpoint("final.pt")
        print(f"done. final weights -> {final}")


def _parse_layer_weights(tokens: list[str] | None) -> dict[str, float] | None:
    """['conv1_1=4.0', 'conv2_1=3.0'] -> {'conv1_1': 4.0, 'conv2_1': 3.0}."""
    if not tokens:
        return None
    weights = {}
    for tok in tokens:
        layer, sep, value = tok.partition("=")
        if not sep:
            raise ValueError(f"expected LAYER=WEIGHT, got {tok!r}")
        weights[layer] = float(value)
    return weights


def _parse(argv: list[str]) -> tuple[list[str], Config]:
    p = argparse.ArgumentParser(prog="train_style", description="train a feed-forward style network")
    p.add_argument("style", nargs="+", help="one or more style images; their common style is used")
    p.add_argument("--coco-dir", default=Config.coco_dir, dest="coco_dir")
    p.add_argument("--style-size", type=int, default=Config.style_size, dest="style_size")
    p.add_argument("--crop-size", type=int, default=Config.crop_size, dest="crop_size")
    p.add_argument("--eval-size", type=int, nargs="+", default=list(Config.eval_size), dest="eval_size",
                   help="judge content/style at this size (or sum of several); must be <= --crop-size")
    p.add_argument("--eval-size-weights", type=float, nargs="+", default=Config.eval_size_weights,
                   dest="eval_size_weights", help="per-scale weight, matched by position to --eval-size")
    p.add_argument("--batch-size", type=int, default=Config.batch_size, dest="batch_size")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--content-weight", type=float, default=Config.content_weight, dest="content_weight")
    p.add_argument("--style-weight", type=float, default=Config.style_weight, dest="style_weight")
    p.add_argument("--tv-weight", type=float, default=Config.tv_weight, dest="tv_weight")
    p.add_argument("--color-weight", type=float, default=Config.color_weight, dest="color_weight",
                   help="pulls the output's colour toward its own source photo's (soft alternative "
                        "to stylize.py's --preserve-color) -- see artist.py's --color-weight")
    p.add_argument("--style-layer-weights", type=str, nargs="+", default=None, dest="style_layer_weights",
                   metavar="LAYER=WEIGHT", help="e.g. conv1_1=4.0 conv2_1=3.0 conv4_1=0.5 conv5_1=0.25")
    p.add_argument("--depthwise", action="store_true", default=Config.depthwise)
    p.add_argument("--resume", default=Config.resume,
                   help="warm-start from a checkpoint's WEIGHTS ONLY (fresh optimizer/step counter) -- "
                        "for re-tuning, e.g. a different --style-weight")
    p.add_argument("--continue-from", default=Config.continue_from, dest="continue_from",
                   help="fully resume an interrupted run -- weights + optimizer state + step counter, "
                        "all restored (needs a checkpoint saved by this script, not an old bare weights file)")
    p.add_argument("--checkpoint-dir", default=Config.checkpoint_dir, dest="checkpoint_dir")
    p.add_argument("--checkpoint-every", type=int, default=Config.checkpoint_every, dest="checkpoint_every")
    p.add_argument("--preview", default=Config.preview_path, dest="preview_path")
    p.add_argument("--preview-size", type=int, default=Config.preview_size, dest="preview_size")
    p.add_argument("--preview-every", type=int, default=Config.preview_every, dest="preview_every")
    p.add_argument("--log-every", type=int, default=Config.log_every, dest="log_every")
    p.add_argument("--device", default=Config.device)
    a = p.parse_args(argv)
    eval_size_weights = tuple(a.eval_size_weights) if a.eval_size_weights is not None else None
    cfg = Config(coco_dir=a.coco_dir, style_size=a.style_size, crop_size=a.crop_size,
                 eval_size=tuple(a.eval_size), eval_size_weights=eval_size_weights,
                 batch_size=a.batch_size, epochs=a.epochs, lr=a.lr,
                 content_weight=a.content_weight, style_weight=a.style_weight,
                 tv_weight=a.tv_weight, color_weight=a.color_weight,
                 style_layer_weights=_parse_layer_weights(a.style_layer_weights),
                 depthwise=a.depthwise, resume=a.resume, continue_from=a.continue_from,
                 checkpoint_dir=a.checkpoint_dir, checkpoint_every=a.checkpoint_every,
                 preview_path=a.preview_path, preview_size=a.preview_size,
                 preview_every=a.preview_every, log_every=a.log_every, device=a.device)
    return a.style, cfg


def main(argv: list[str]) -> None:
    style_paths, cfg = _parse(argv)
    styles = [load_image(p, cfg.style_size) for p in style_paths]
    loader = coco_loader(cfg.coco_dir, crop_size=cfg.crop_size, batch=cfg.batch_size)

    trainer = StyleTrainer(styles, cfg)
    eval_str = f"  eval_size={cfg.eval_size}" if cfg.eval_size else ""
    if cfg.eval_size_weights:
        eval_str += f" weights={cfg.eval_size_weights}"
    print(f"device {trainer.device}  net params {sum(p.numel() for p in trainer.net.parameters()):,}  "
          f"style images {len(styles)} ({', '.join(style_paths)})  "
          f"lr {cfg.lr}  cw={cfg.content_weight} sw={cfg.style_weight:g} tv={cfg.tv_weight} "
          f"color={cfg.color_weight}{eval_str}"
          f"{'  resume=' + cfg.resume if cfg.resume else ''}")
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
        style2_path = Path(d) / "style2.jpg"
        Image.fromarray(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)).save(style2_path)

        style = load_image(style_path, 64)
        style2 = load_image(style2_path, 64)
        loader = coco_loader(str(photos), crop_size=64, batch=2, num_workers=0)

        # train_step: sane (non-NaN, non-negative) losses, gradient reaches the net's weights
        trainer = StyleTrainer([style], Config(crop_size=64, device="cpu", pretrained=False))
        m = trainer.train_step(next(iter(loader)))
        assert all(not math.isnan(v) and v >= 0 for v in m.values()), f"bad losses: {m}"
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.net.parameters())
        print(f"ok (train_step: {m})")

        # multiple style images: runs end-to-end, one Gram target per scale (here, one scale)
        multi = StyleTrainer([style, style2], Config(crop_size=64, device="cpu", pretrained=False))
        m_multi = multi.train_step(next(iter(loader)))
        assert all(not math.isnan(v) and v >= 0 for v in m_multi.values()), f"bad losses: {m_multi}"
        print(f"ok (multi-style train_step: {m_multi})")

        # color_weight: isolate the mechanism from content/style (which, with random
        # noise photos and an untrained VGG, would otherwise dominate unpredictably)
        # by zeroing every other weight -- color_loss is then the ONLY gradient
        # signal, so repeatedly training on ONE fixed batch (a well-posed, stationary
        # target -- like overfitting one image) must drive it down; with every weight
        # at 0 there's no gradient at all and colour drift just sits at the random init
        one_batch = next(iter(loader))
        torch.manual_seed(0)
        no_grad_trainer = StyleTrainer([style], Config(crop_size=64, content_weight=0.0,
                                                        style_weight=0.0, device="cpu", pretrained=False))
        torch.manual_seed(0)
        colored_trainer = StyleTrainer([style], Config(crop_size=64, content_weight=0.0, style_weight=0.0,
                                                        color_weight=1000.0, device="cpu", pretrained=False))
        for _ in range(10):
            m_flat = no_grad_trainer.train_step(one_batch)
            m_colored = colored_trainer.train_step(one_batch)
        assert m_colored["color"] < m_flat["color"], \
            f"training on color_loss alone should reduce colour drift: {m_colored['color']} >= {m_flat['color']}"
        print(f"ok (color_weight reduces colour drift over a few steps: "
              f"{m_flat['color']:.4f} (untrained) -> {m_colored['color']:.4f} (trained on it))")

        # color_weight + multi-scale eval_size: color_loss should be judged at EACH
        # scale (summed), not just the native crop resolution -- judging it only at
        # native resolution let a high color_weight correct pixels near-
        # independently of their neighbours (see artist.py's own bubble-texture fix)
        ms_color_cfg = Config(crop_size=64, eval_size=(32, 64), color_weight=1000.0,
                               device="cpu", pretrained=False)
        ms_color_trainer = StyleTrainer([style], ms_color_cfg)
        m_ms_color = ms_color_trainer.train_step(one_batch)
        assert not math.isnan(m_ms_color["color"]) and m_ms_color["color"] >= 0, \
            f"bad multi-scale color loss: {m_ms_color['color']}"
        print(f"ok (color_weight judged at each eval_size scale: {m_ms_color})")

        # multi-scale eval_size: two scales, summed, each with its own style-Gram target;
        # the larger scale (== crop_size) should skip resizing (no-op _at_scale)
        ms_cfg = Config(crop_size=64, eval_size=(32, 64), device="cpu", pretrained=False)
        ms_trainer = StyleTrainer([style], ms_cfg)
        assert len(ms_trainer.style_grams) == 2, "two scales -> two style-Gram target dicts"
        m_ms = ms_trainer.train_step(next(iter(loader)))
        assert all(not math.isnan(v) and v >= 0 for v in m_ms.values()), f"bad losses: {m_ms}"
        print(f"ok (multi-scale eval_size=(32,64) train_step: {m_ms})")

        # eval_size must not exceed crop_size -- there's no extra real detail beyond the crop
        try:
            StyleTrainer([style], Config(crop_size=32, eval_size=(64,), device="cpu", pretrained=False))
            raise AssertionError("expected ValueError for eval_size > crop_size")
        except ValueError:
            print("ok (eval_size > crop_size raises)")

        # fit: full loop, checkpointing, preview writing
        cfg = Config(crop_size=64, batch_size=2, epochs=1, device="cpu", pretrained=False,
                     checkpoint_dir=str(Path(d) / "ckpt"), checkpoint_every=2,
                     preview_path=str(style_path), preview_size=64, preview_every=2, log_every=1)
        trainer2 = StyleTrainer([style], cfg)
        trainer2.fit(loader)

        assert trainer2._step == len(loader), f"expected {len(loader)} steps, got {trainer2._step}"
        assert (Path(cfg.checkpoint_dir) / "final.pt").exists(), "final checkpoint missing"
        assert (Path(cfg.checkpoint_dir) / "latest.pt").exists(), "periodic checkpoint missing"
        previews = list((Path(cfg.checkpoint_dir) / "previews").glob("*.png"))
        assert previews, "no preview written"
        print(f"ok (fit: {trainer2._step} steps, checkpoints + {len(previews)} preview(s) written)")

        # resume: weights come back exactly, with a fresh optimizer, and training continues
        resumed = StyleTrainer([style], Config(crop_size=64, device="cpu", pretrained=False,
                                                resume=str(Path(cfg.checkpoint_dir) / "final.pt")))
        for p1, p2 in zip(resumed.net.parameters(), trainer2.net.parameters()):
            assert torch.equal(p1, p2), "resume did not restore the exact trained weights"
        assert resumed._step == 0, "resume should not restore the step counter"
        resumed.train_step(next(iter(loader)))  # doesn't crash, i.e. optimizer/net are usable
        print("ok (resume: weights restored exactly, step counter fresh, training continues)")

        # continue_from: weights AND optimizer state AND step counter all come back,
        # so a paused-and-resumed run picks up exactly where it left off
        cont_cfg = Config(crop_size=64, batch_size=2, epochs=1, device="cpu", pretrained=False,
                           checkpoint_dir=str(Path(d) / "ckpt2"), log_every=1,
                           continue_from=str(Path(cfg.checkpoint_dir) / "final.pt"))
        continued = StyleTrainer([style], cont_cfg)
        for p1, p2 in zip(continued.net.parameters(), trainer2.net.parameters()):
            assert torch.equal(p1, p2), "continue_from did not restore the exact weights"
        assert continued._step == trainer2._step, \
            f"continue_from should restore the step counter: {continued._step} != {trainer2._step}"
        assert continued.opt.state_dict()["state"].keys() == trainer2.opt.state_dict()["state"].keys(), \
            "continue_from should restore optimizer state (Adam's per-parameter momentum/variance)"
        step_before = continued._step
        continued.fit(loader)
        assert continued._step == step_before + len(loader), \
            "fit() should keep counting up from the restored step, not restart at 0"
        print(f"ok (continue_from: weights + optimizer state + step {step_before} all restored, "
              f"training continues to step {continued._step})")

        # continue_from refuses an old-style bare-weights file (no optimizer state to restore)
        bare_weights = Path(d) / "bare.pt"
        torch.save(trainer2.net.state_dict(), bare_weights)
        try:
            StyleTrainer([style], Config(crop_size=64, device="cpu", pretrained=False,
                                          continue_from=str(bare_weights)))
            raise AssertionError("expected ValueError for a bare-weights file")
        except ValueError:
            print("ok (continue_from rejects a bare-weights file, suggests --resume)")

        # resume and continue_from are mutually exclusive
        try:
            StyleTrainer([style], Config(crop_size=64, device="cpu", pretrained=False,
                                          resume=str(bare_weights), continue_from=str(bare_weights)))
            raise AssertionError("expected ValueError when both resume and continue_from are set")
        except ValueError:
            print("ok (resume + continue_from together raises)")

        # non-finite loss (simulated blow-up): step must be skipped, weights untouched
        trainer3 = StyleTrainer([style], Config(crop_size=64, device="cpu", pretrained=False))
        params = list(trainer3.net.parameters())
        with torch.no_grad():
            params[0].fill_(float("nan"))  # forces a non-finite forward pass
        untouched_before = params[-1].clone()
        trainer3.train_step(next(iter(loader)))
        assert not trainer3.step_ok, "expected step_ok=False on a non-finite loss"
        assert torch.equal(params[-1], untouched_before), "optimizer step must be skipped, not just flagged"
        print("ok (non-finite loss: step skipped, other weights untouched)")

        # fit() must stop immediately on a non-finite step (rather than run all epochs) and
        # still write final.pt -- force it via a wrapped train_step so this is deterministic
        # rather than depending on manufacturing a real numeric blow-up in a tiny synthetic run
        cfg3 = Config(crop_size=64, batch_size=2, epochs=5, device="cpu", pretrained=False,
                      checkpoint_dir=str(Path(d) / "ckpt_nan"), log_every=1)
        trainer4 = StyleTrainer([style], cfg3)
        real_train_step = trainer4.train_step

        def blow_up_on_second_call(batch):
            m = real_train_step(batch)
            if trainer4._step == 2:
                trainer4.step_ok = False
            return m

        trainer4.train_step = blow_up_on_second_call
        trainer4.fit(loader)
        assert trainer4._step == 2, f"expected fit() to stop right after the failing step, got {trainer4._step}"
        assert (Path(cfg3.checkpoint_dir) / "final.pt").exists(), "final.pt missing after early stop"
        print("ok (fit: stops early on a non-finite step, still writes final.pt)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _selfcheck()
