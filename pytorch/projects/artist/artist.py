"""AI Artist — restyle a content image with the texture of a style image.

Gatys-style neural style transfer: freeze VGG16, then optimise the *pixels* of a
generated image so its deep features match the content image and its Gram matrices
match the style image. Nothing here trains network weights, so it does not use
`training.Trainer` — the optimisation loop is `StyleTransfer.run`.

Run from inside pytorch/:
    python -m projects.artist.artist <content> <style> [-o out.png] [--steps N] ...
    python -m projects.artist.artist            # no args -> offline self-check
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

from projects.artist.images import from_vgg, load_image, save_image, to_vgg
from projects.artist.losses import content_loss, gram_matrices, style_loss, tv_loss
from projects.artist.vgg import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures


@dataclass
class Config:
    image_size: int = 512
    steps: int = 20                          # lbfgs: outer steps (each = lbfgs_iter inner); adam: ~500
    optimizer: str = "lbfgs"                 # "lbfgs" | "adam"
    lr: float = 1.0                          # lbfgs ~1.0, adam ~0.02
    lbfgs_iter: int = 20                     # inner iterations per lbfgs step (max_iter)
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 0.0
    init: str = "content"                    # "content" | "noise"
    pool: str = "max"                        # "max" | "avg", passed to VGGFeatures
    content_layers: tuple[str, ...] = CONTENT_LAYERS
    style_layers: tuple[str, ...] = STYLE_LAYERS
    style_layer_weights: dict[str, float] | None = None
    pretrained: bool = True                  # False -> random VGG (self-check only)
    device: str = "cuda"
    verbose: bool = True
    log_every: int = 10


class StyleTransfer:
    """Frozen VGG + fixed content/style targets + the optimise loop over one image."""

    def __init__(self, content: torch.Tensor, style: torch.Tensor, cfg: Config = Config()):
        if cfg.init not in ("content", "noise"):
            raise ValueError(f"init must be 'content' or 'noise', got {cfg.init!r}")
        if cfg.optimizer not in ("lbfgs", "adam"):
            raise ValueError(f"optimizer must be 'lbfgs' or 'adam', got {cfg.optimizer!r}")
        self.cfg = cfg
        self.device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"

        # one frozen feature extractor, taps for every layer either loss needs
        taps = tuple(dict.fromkeys(cfg.content_layers + cfg.style_layers))
        self.vgg = VGGFeatures(taps, pretrained=cfg.pretrained, pool=cfg.pool).to(self.device)

        # normalise the two inputs once, here — never inside the loop
        content = to_vgg(content.unsqueeze(0)).to(self.device)
        style = to_vgg(style.unsqueeze(0)).to(self.device)

        # fixed optimisation targets: content activations, style Gram matrices
        with torch.no_grad():
            cf, sf = self.vgg(content), self.vgg(style)
        self.content_targets = {l: cf[l].detach() for l in cfg.content_layers}
        self.style_grams = {l: g.detach() for l, g in gram_matrices(sf, cfg.style_layers).items()}

        # the only thing we optimise: the generated image's pixels (a leaf tensor)
        seed = content.clone() if cfg.init == "content" else torch.randn_like(content)
        self.img = seed.detach().clone().requires_grad_(True)

        if cfg.optimizer == "lbfgs":
            self.opt = torch.optim.LBFGS([self.img], lr=cfg.lr, max_iter=cfg.lbfgs_iter,
                                         line_search_fn="strong_wolfe")
        else:
            self.opt = torch.optim.Adam([self.img], lr=cfg.lr)

        self._i = 0
        self._last: dict[str, float] = {}
        self._best_loss = float("inf")
        self._best = self.img.detach().clone()

    # --- one optimiser step -------------------------------------------------

    def _closure(self) -> torch.Tensor:
        self.opt.zero_grad()
        feats = self.vgg(self.img)
        c = content_loss(feats, self.content_targets, self.cfg.content_layers)
        s = style_loss(feats, self.style_grams, self.cfg.style_layers, self.cfg.style_layer_weights)
        tv = tv_loss(self.img)
        total = self.cfg.content_weight * c + self.cfg.style_weight * s + self.cfg.tv_weight * tv
        total.backward()
        self._last = {"content": c.item(), "style": s.item(), "tv": tv.item(), "total": total.item()}
        return total

    def step(self) -> dict[str, float]:
        """One optimiser step (lbfgs: up to `lbfgs_iter` inner iterations; adam: one
        update). Returns the four raw (pre-weight) loss values from the last eval."""
        if self.cfg.optimizer == "lbfgs":
            self.opt.step(self._closure)
        else:
            self._closure()
            self.opt.step()
        self._i += 1
        if self._last["total"] < self._best_loss:            # keep the best iterate
            self._best_loss = self._last["total"]
            self._best = self.img.detach().clone()
        return self._last

    def run(self, steps: int | None = None) -> torch.Tensor:
        """Optimise for `steps` iterations; return the best image as CHW in [0,1]."""
        n = steps if steps is not None else self.cfg.steps
        for k in range(1, n + 1):
            m = self.step()
            if self.cfg.verbose and (k == 1 or k == n or k % self.cfg.log_every == 0):
                print(f"step {k:4d}/{n}  total {m['total']:11.2f}  "
                      f"content {m['content']:.4f}  style {m['style']:.3e}  tv {m['tv']:.4f}")
        return self.best_image

    # --- views ------------------------------------------------------------

    @property
    def image(self) -> torch.Tensor:
        """Current iterate, de-normalised to CHW (unclamped; save_image clamps)."""
        return from_vgg(self.img.detach())[0].cpu()

    @property
    def best_image(self) -> torch.Tensor:
        """Lowest-loss iterate seen, de-normalised to CHW."""
        return from_vgg(self._best)[0].cpu()


# --- CLI ---------------------------------------------------------------------


def _parse(argv: list[str]) -> tuple[str, str, str, Config]:
    p = argparse.ArgumentParser(prog="artist", description="neural style transfer")
    p.add_argument("content")
    p.add_argument("style")
    p.add_argument("-o", "--out", default="styled.png")
    p.add_argument("--size", type=int, default=Config.image_size, dest="image_size")
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--optimizer", choices=("lbfgs", "adam"), default=Config.optimizer)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--content-weight", type=float, default=Config.content_weight, dest="content_weight")
    p.add_argument("--style-weight", type=float, default=Config.style_weight, dest="style_weight")
    p.add_argument("--tv-weight", type=float, default=Config.tv_weight, dest="tv_weight")
    p.add_argument("--init", choices=("content", "noise"), default=Config.init)
    p.add_argument("--pool", choices=("max", "avg"), default=Config.pool)
    p.add_argument("--device", default=Config.device)
    a = p.parse_args(argv)
    cfg = Config(image_size=a.image_size, steps=a.steps, optimizer=a.optimizer, lr=a.lr,
                 content_weight=a.content_weight, style_weight=a.style_weight,
                 tv_weight=a.tv_weight, init=a.init, pool=a.pool, device=a.device)
    return a.content, a.style, a.out, cfg


def main(argv: list[str]) -> None:
    content_path, style_path, out_path, cfg = _parse(argv)
    content = load_image(content_path, cfg.image_size)
    style = load_image(style_path, cfg.image_size)

    art = StyleTransfer(content, style, cfg)
    print(f"device {art.device}  content {tuple(content.shape)}  style {tuple(style.shape)}  "
          f"{cfg.optimizer} lr={cfg.lr}  steps {cfg.steps}  "
          f"cw={cfg.content_weight} sw={cfg.style_weight:g} tv={cfg.tv_weight}")
    art.run()
    print("wrote", save_image(art.best_image, out_path), f"(best total {art._best_loss:.2f})")


def _selfcheck() -> None:
    torch.manual_seed(0)
    content, style = torch.rand(3, 64, 64), torch.rand(3, 64, 64)
    try:                                             # real weights -> a meaningful drop
        pretrained, drop = True, 0.5
        StyleTransfer(content, style, Config(image_size=64, steps=1, device="cpu"))
    except FileNotFoundError:                        # LFS not pulled -> exercise plumbing only
        pretrained, drop = False, 1.0
        print("  (no pretrained weights; run `git lfs pull` in weights/vgg16.tv_in1k)")

    cfg = Config(image_size=64, steps=6, lbfgs_iter=5, device="cpu",
                 pretrained=pretrained, verbose=False)
    art = StyleTransfer(content, style, cfg)
    first = art.step()["total"]
    for _ in range(6):
        last = art.step()["total"]
    assert 0 < last < first * drop, f"loss did not drop enough: {first:.2f} -> {last:.2f}"
    assert art.best_image.shape == (3, 64, 64), art.best_image.shape
    assert art._best_loss <= first
    grad_ok = art.img.grad is not None and art.img.grad.abs().sum() > 0
    assert grad_ok, "no gradient on the image"
    print(f"ok (self-check: total {first:.1f} -> {last:.1f}, best {art._best_loss:.1f}, grad flows)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _selfcheck()
