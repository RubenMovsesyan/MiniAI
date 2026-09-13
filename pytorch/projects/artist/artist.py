"""AI Artist — restyle a content image with the texture of one or more style images.

Gatys-style neural style transfer: freeze VGG16, then optimise the *pixels* of a
generated image so its deep features match the content image and its Gram matrices
match the style image(s). Nothing here trains network weights, so it does not use
`training.Trainer` — the optimisation loop is `StyleTransfer.run`.

Multiple style images -> their common style, not any one of them: each image's
Gram matrices are computed independently, then averaged per layer into a single
target before the loss ever runs (see StyleTransfer.__init__). style_loss itself
is unchanged -- it always compares against one target, just possibly a blended one.

Run from inside pytorch/:
    python -m projects.artist.artist <content> <style> [<style2> ...] [-o out.png] [--steps N] ...
    python -m projects.artist.artist            # no args -> offline self-check
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from projects.artist.images import from_vgg, load_image, resize_longer_side, save_image, to_vgg
from projects.artist.losses import (coherence_map, content_loss, gram_matrices, style_loss,
                                     style_loss_masked, tv_loss)
from projects.artist.vgg import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures


@dataclass
class Config:
    image_size: int = 512
    eval_size: tuple[int, ...] = ()          # () = judge losses once, at image_size (current behaviour).
                                              # One or more sizes decouples output resolution from
                                              # brush-stroke scale, and multiple sizes judge at ALL of
                                              # them at once (summed) -- e.g. (200, 1024): a coarse
                                              # scale sets big stroke placement, a fine scale forces
                                              # real sub-200px texture instead of a smoothed-up filter.
                                              # The optimised image always still renders at image_size.
    eval_size_weights: tuple[float, ...] | None = None  # per-scale weight (default 1.0 each)
    steps: int = 20                          # lbfgs: outer steps (each = lbfgs_iter inner); adam: ~500
    optimizer: str = "lbfgs"                 # "lbfgs" | "adam"
    lr: float = 1.0                          # lbfgs ~1.0, adam ~0.02
    lbfgs_iter: int = 20                     # inner iterations per lbfgs step (max_iter)
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 0.0
    coherence_mask_strength: float = 0.0     # 0 = off (plain style_loss everywhere); 1 = the generated
                                              # image's contribution to each style Gram is weighted by
                                              # the CONTENT's own structure-tensor orientation coherence
                                              # (see losses.coherence_map/style_loss_masked) -- damps
                                              # style's pull specifically where the content has no
                                              # dominant local edge direction to align a brush stroke to
                                              # (e.g. flat skin), instead of changing the loss everywhere
                                              # or changing which layer dominates. Values between 0 and 1
                                              # blend: mask = (1 - strength) + strength * coherence.
    coherence_blur_size: int = 32             # coherence_map's raw output is collapsed to this size
                                              # (longer side) before use, to smooth out its per-pixel
                                              # noise into a per-REGION signal -- see __init__.
    coherence_threshold: float | None = None  # None = off (use the smoothed coherence value directly).
                                              # Set a cutoff (e.g. 0.4) to binarise the smoothed map
                                              # instead -- above -> full style pull, below -> none --
                                              # since a "bad" region's mean coherence isn't dramatically
                                              # lower than a "good" one's (both sit well under 0.5), so
                                              # using the value directly as a linear mask leaves real
                                              # style pull in the bad region too. Applied AFTER blurring.
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

    def __init__(self, content: torch.Tensor, styles: list[torch.Tensor], cfg: Config = Config()):
        if cfg.init not in ("content", "noise"):
            raise ValueError(f"init must be 'content' or 'noise', got {cfg.init!r}")
        if cfg.optimizer not in ("lbfgs", "adam"):
            raise ValueError(f"optimizer must be 'lbfgs' or 'adam', got {cfg.optimizer!r}")
        if not styles:
            raise ValueError("styles must be a non-empty list of images")
        self.cfg = cfg
        self.device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"

        # one frozen feature extractor, taps for every layer either loss needs
        taps = tuple(dict.fromkeys(cfg.content_layers + cfg.style_layers))
        self.vgg = VGGFeatures(taps, pretrained=cfg.pretrained, pool=cfg.pool).to(self.device)

        # raw, [0,1] content reference for coherence_map (pixel-space, not
        # VGG-normalised) -- only built when actually used, keep it before
        # `content` below is overwritten
        content_raw = content.unsqueeze(0).to(self.device) if cfg.coherence_mask_strength else None

        # normalise the content input once, here — never inside the loop. Keep the
        # FULL-resolution version to seed self.img (the actual output canvas);
        # build content/style targets at one or more (possibly smaller) eval_size
        # scales instead, so brush-stroke scale and output resolution can be tuned
        # independently -- see StyleTransfer._closure, which re-derives the same
        # per-scale downsamples from self.img every step. One dict per scale.
        content = to_vgg(content.unsqueeze(0)).to(self.device)
        self._scales = cfg.eval_size if cfg.eval_size else (cfg.image_size,)

        self.content_targets: list[dict[str, torch.Tensor]] = []
        self.style_grams: list[dict[str, torch.Tensor]] = []
        self.style_masks: list[dict[str, torch.Tensor]] = []
        for scale in self._scales:
            content_s = self._at_scale(content, scale)
            with torch.no_grad():
                cf = self.vgg(content_s)
            self.content_targets.append({l: cf[l].detach() for l in cfg.content_layers})

            # coherence mask for style_loss_masked (see _closure): computed once from
            # the content's OWN coherence at this scale. coherence_map's per-pixel
            # output is itself locally noisy (varies from ~0 to ~1 within a few
            # pixels, even in "good" regions) -- used directly as a mask, that fine
            # texture imprints onto the Gram statistic instead of acting as a clean
            # per-REGION dial. Collapsing it down to coherence_blur_size first (then
            # letting the per-layer resize below blow it back up) throws away that
            # noise and keeps only the broad, region-scale trend. blend =
            # (1-strength) + strength*coherence, so strength=0 skips this entirely.
            if cfg.coherence_mask_strength:
                coh = coherence_map(self._at_scale(content_raw, scale)).detach()
                coh = resize_longer_side(coh, cfg.coherence_blur_size)
                if cfg.coherence_threshold is not None:
                    coh = (coh > cfg.coherence_threshold).float()
                self.style_masks.append({
                    l: (1 - cfg.coherence_mask_strength
                        + cfg.coherence_mask_strength
                        * F.interpolate(coh, size=cf[l].shape[-2:], mode="bilinear", align_corners=False))
                    for l in cfg.style_layers
                })

            # style target: each image's Gram matrices computed independently, then
            # averaged per layer into ONE target -- the "common style" across all of
            # them. Gram matrices are [1,C,C] regardless of the source image's H,W
            # (see gram_matrix's normalisation), so different-sized/aspect-ratio
            # style images average together with no extra resizing. style_loss
            # itself never knows there was more than one image.
            per_image_grams = []
            for s in styles:
                s_s = self._at_scale(to_vgg(s.unsqueeze(0)).to(self.device), scale)
                with torch.no_grad():
                    sf = self.vgg(s_s)
                per_image_grams.append(gram_matrices(sf, cfg.style_layers))
            self.style_grams.append({
                l: torch.stack([g[l] for g in per_image_grams], dim=0).mean(0).detach()
                for l in cfg.style_layers
            })

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

    def _at_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Resize to `scale` (longer side), or return `x` unchanged if it's
        already that size -- skips a needless identity resize."""
        return x if max(x.shape[-2:]) == scale else resize_longer_side(x, scale)

    # --- one optimiser step -------------------------------------------------

    def _closure(self) -> torch.Tensor:
        self.opt.zero_grad()
        # judge content/style at each eval_size scale (brush-stroke scale) but
        # self.img itself stays at full resolution -- gradients flow back through
        # each downsample, so neighbouring high-res pixels move together in large,
        # coherent strokes. Multiple scales sum: a coarse one sets big stroke
        # placement, a finer one forces real texture at that finer scale too,
        # instead of one scale doing both jobs.
        c = s = self.img.new_zeros(())
        for i, (scale, content_target, style_gram) in enumerate(
                zip(self._scales, self.content_targets, self.style_grams)):
            w = 1.0 if self.cfg.eval_size_weights is None else self.cfg.eval_size_weights[i]
            x = self._at_scale(self.img, scale)
            feats = self.vgg(x)
            c = c + w * content_loss(feats, content_target, self.cfg.content_layers)
            if self.cfg.coherence_mask_strength:
                s = s + w * style_loss_masked(feats, style_gram, self.cfg.style_layers,
                                               self.style_masks[i], self.cfg.style_layer_weights)
            else:
                s = s + w * style_loss(feats, style_gram, self.cfg.style_layers, self.cfg.style_layer_weights)
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
                      f"content {m['content']:.4f}  style {m['style']:.3e}  "
                      f"tv {m['tv']:.4f}")
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


def _parse(argv: list[str]) -> tuple[str, list[str], str, Config]:
    p = argparse.ArgumentParser(prog="artist", description="neural style transfer")
    p.add_argument("content")
    p.add_argument("style", nargs="+", help="one or more style images; their common style is used")
    p.add_argument("-o", "--out", default="styled.png")
    p.add_argument("--size", type=int, default=Config.image_size, dest="image_size")
    p.add_argument("--eval-size", type=int, nargs="+", default=list(Config.eval_size), dest="eval_size",
                   help="judge brush-stroke scale at this size (or sum of several sizes) "
                        "while --size stays the output resolution")
    p.add_argument("--eval-size-weights", type=float, nargs="+", default=Config.eval_size_weights,
                   dest="eval_size_weights", help="per-scale weight, matched by position to --eval-size")
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--optimizer", choices=("lbfgs", "adam"), default=Config.optimizer)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--content-weight", type=float, default=Config.content_weight, dest="content_weight")
    p.add_argument("--style-weight", type=float, default=Config.style_weight, dest="style_weight")
    p.add_argument("--tv-weight", type=float, default=Config.tv_weight, dest="tv_weight")
    p.add_argument("--coherence-mask-strength", type=float, default=Config.coherence_mask_strength,
                   dest="coherence_mask_strength",
                   help="0-1: damps style_loss where the content has no dominant local edge direction "
                        "to align a brush stroke to (fixes a maze/fingerprint texture in flat regions)")
    p.add_argument("--init", choices=("content", "noise"), default=Config.init)
    p.add_argument("--pool", choices=("max", "avg"), default=Config.pool)
    p.add_argument("--device", default=Config.device)
    a = p.parse_args(argv)
    eval_size_weights = tuple(a.eval_size_weights) if a.eval_size_weights is not None else None
    cfg = Config(image_size=a.image_size, eval_size=tuple(a.eval_size), eval_size_weights=eval_size_weights,
                 steps=a.steps, optimizer=a.optimizer, lr=a.lr, content_weight=a.content_weight,
                 style_weight=a.style_weight, tv_weight=a.tv_weight,
                 coherence_mask_strength=a.coherence_mask_strength,
                 init=a.init, pool=a.pool, device=a.device)
    return a.content, a.style, a.out, cfg


def main(argv: list[str]) -> None:
    content_path, style_paths, out_path, cfg = _parse(argv)
    content = load_image(content_path, cfg.image_size)
    styles = [load_image(p, cfg.image_size) for p in style_paths]

    art = StyleTransfer(content, styles, cfg)
    eval_str = f"  eval_size={cfg.eval_size}" if cfg.eval_size else ""
    if cfg.eval_size_weights:
        eval_str += f" weights={cfg.eval_size_weights}"
    print(f"device {art.device}  content {tuple(content.shape)}  "
          f"style images {len(styles)} ({', '.join(style_paths)})  "
          f"{cfg.optimizer} lr={cfg.lr}  steps {cfg.steps}  "
          f"cw={cfg.content_weight} sw={cfg.style_weight:g} tv={cfg.tv_weight} "
          f"coherence_mask={cfg.coherence_mask_strength}{eval_str}")
    art.run()
    print("wrote", save_image(art.best_image, out_path), f"(best total {art._best_loss:.2f})")


def _selfcheck() -> None:
    torch.manual_seed(0)
    content, style = torch.rand(3, 64, 64), torch.rand(3, 64, 64)
    try:                                             # real weights -> a meaningful drop
        pretrained, drop = True, 0.5
        StyleTransfer(content, [style], Config(image_size=64, steps=1, device="cpu"))
    except FileNotFoundError:                        # LFS not pulled -> exercise plumbing only
        pretrained, drop = False, 1.0
        print("  (no pretrained weights; run `git lfs pull` in weights/vgg16.tv_in1k)")

    cfg = Config(image_size=64, steps=6, lbfgs_iter=5, device="cpu",
                 pretrained=pretrained, verbose=False)
    art = StyleTransfer(content, [style], cfg)
    first = art.step()["total"]
    for _ in range(6):
        last = art.step()["total"]
    assert 0 < last < first * drop, f"loss did not drop enough: {first:.2f} -> {last:.2f}"
    assert art.best_image.shape == (3, 64, 64), art.best_image.shape
    assert art._best_loss <= first
    grad_ok = art.img.grad is not None and art.img.grad.abs().sum() > 0
    assert grad_ok, "no gradient on the image"
    print(f"ok (self-check: total {first:.1f} -> {last:.1f}, best {art._best_loss:.1f}, grad flows)")

    # multiple style images: runs end-to-end and produces a valid image
    style2, style3 = torch.rand(3, 64, 64), torch.rand(3, 64, 64)
    multi = StyleTransfer(content, [style, style2, style3], cfg)
    m_first = multi.step()["total"]
    for _ in range(6):
        m_last = multi.step()["total"]
    assert 0 < m_last < m_first * drop, f"multi-style loss did not drop: {m_first:.2f} -> {m_last:.2f}"
    assert multi.best_image.shape == (3, 64, 64), multi.best_image.shape
    print(f"ok (multi-style: total {m_first:.1f} -> {m_last:.1f})")

    # duplicates of the same image must average to exactly that image's own Gram
    dup = StyleTransfer(content, [style, style, style], cfg)
    single = StyleTransfer(content, [style], cfg)
    for l in cfg.style_layers:
        assert torch.allclose(dup.style_grams[0][l], single.style_grams[0][l], atol=1e-6), \
            f"duplicate-style average drifted from the single-style target at {l}"
    print("ok (N copies of one style == that one style)")

    # the averaging arithmetic itself: manually average two distinct images' Grams
    # and check it matches what StyleTransfer actually built
    from projects.artist.losses import gram_matrices as _gram_matrices
    pair = StyleTransfer(content, [style, style2], cfg)
    with torch.no_grad():
        f1 = pair.vgg(to_vgg(style.unsqueeze(0)))
        f2 = pair.vgg(to_vgg(style2.unsqueeze(0)))
    g1, g2 = _gram_matrices(f1, cfg.style_layers), _gram_matrices(f2, cfg.style_layers)
    for l in cfg.style_layers:
        expected = (g1[l] + g2[l]) / 2
        assert torch.allclose(pair.style_grams[0][l], expected, atol=1e-6), f"average is wrong at {l}"
    print("ok (two-image average matches a manual (g1+g2)/2)")

    # eval_size: output stays at full image_size even though the loss judges a
    # smaller scale; content_targets come out at that smaller spatial size
    no_eval = StyleTransfer(content, [style], Config(image_size=64, device="cpu", pretrained=pretrained))
    e_cfg = Config(image_size=64, eval_size=(32,), steps=6, lbfgs_iter=5, device="cpu",
                   pretrained=pretrained, verbose=False)
    e_art = StyleTransfer(content, [style], e_cfg)
    layer = cfg.content_layers[0]
    assert len(e_art.content_targets) == 1, "single-scale should give one target dict"
    assert e_art.content_targets[0][layer].shape[-1] < no_eval.content_targets[0][layer].shape[-1], \
        "eval_size should make the content target spatially smaller"
    e_first = e_art.step()["total"]
    for _ in range(6):
        e_last = e_art.step()["total"]
    assert 0 < e_last < e_first, f"eval_size loss did not drop: {e_first:.2f} -> {e_last:.2f}"
    assert e_art.best_image.shape == (3, 64, 64), e_art.best_image.shape  # output stays at image_size
    print(f"ok (eval_size: output {tuple(e_art.best_image.shape)} at image_size=64, "
          f"loss judged at eval_size=32, total {e_first:.1f} -> {e_last:.1f})")

    # multi-scale eval_size: two scales, summed, each gets its own target dict;
    # the larger scale (== image_size) should skip resizing self.img entirely
    ms_cfg = Config(image_size=64, eval_size=(16, 64), steps=6, lbfgs_iter=5, device="cpu",
                    pretrained=pretrained, verbose=False)
    ms_art = StyleTransfer(content, [style], ms_cfg)
    assert len(ms_art.content_targets) == len(ms_art.style_grams) == 2, "two scales -> two target dicts"
    assert ms_art.content_targets[0][layer].shape[-1] < ms_art.content_targets[1][layer].shape[-1], \
        "the two scales' targets should differ in spatial size"
    assert ms_art._at_scale(ms_art.img, 64) is ms_art.img, "no-op resize should skip resize_longer_side"
    ms_first = ms_art.step()["total"]
    for _ in range(6):
        ms_last = ms_art.step()["total"]
    assert 0 < ms_last < ms_first, f"multi-scale loss did not drop: {ms_first:.2f} -> {ms_last:.2f}"
    assert ms_art.best_image.shape == (3, 64, 64), ms_art.best_image.shape
    print(f"ok (multi-scale eval_size=(16,64): total {ms_first:.1f} -> {ms_last:.1f})")

    # per-scale weights: doubling every weight should exactly double content+style
    # (relative to tv, which isn't touched by eval_size_weights)
    base = StyleTransfer(content, [style], Config(image_size=64, eval_size=(16, 64), device="cpu",
                                                   pretrained=pretrained, verbose=False))
    weighted = StyleTransfer(content, [style], Config(image_size=64, eval_size=(16, 64),
                                                       eval_size_weights=(2.0, 2.0), device="cpu",
                                                       pretrained=pretrained, verbose=False))
    with torch.no_grad():
        base.img.copy_(weighted.img)  # same starting canvas for a fair comparison
    base._closure()
    weighted._closure()
    assert torch.allclose(torch.tensor(weighted._last["content"]), 2 * torch.tensor(base._last["content"]), atol=1e-4)
    assert torch.allclose(torch.tensor(weighted._last["style"]), 2 * torch.tensor(base._last["style"]), atol=1e-4)
    print("ok (eval_size_weights scales content/style contributions as expected)")

    # coherence_mask_strength: wiring check -- style_masks get built (one dict per
    # scale, one tensor per style layer, resized to that layer's own feature-map
    # size), and the run stays numerically sane end-to-end. The mechanism itself
    # (coherence_map, style_loss_masked) is unit-tested in losses.py; this only
    # checks artist.py actually threads it through correctly.
    masked = StyleTransfer(content, [style], Config(image_size=64, coherence_mask_strength=0.5,
                                                      device="cpu", pretrained=pretrained, verbose=False))
    assert len(masked.style_masks) == 1, "one scale -> one mask dict"
    assert set(masked.style_masks[0]) == set(cfg.style_layers), "one mask tensor per style layer"
    assert (masked.style_masks[0]["conv1_1"].shape[-1] > masked.style_masks[0]["conv5_1"].shape[-1]), \
        "each layer's mask should be resized to that layer's own (pooled-down) feature-map size"
    m_masked = masked.step()
    assert all(v == v and abs(v) < float("inf") for v in m_masked.values()), f"bad losses: {m_masked}"
    print("ok (coherence_mask_strength: style_masks built per scale/layer, run stays finite)")

    # strength=0 must behave identically to no masking at all (same seed/canvas)
    plain0 = StyleTransfer(content, [style], Config(image_size=64, device="cpu",
                                                     pretrained=pretrained, verbose=False))
    off0 = StyleTransfer(content, [style], Config(image_size=64, coherence_mask_strength=0.0,
                                                   device="cpu", pretrained=pretrained, verbose=False))
    assert off0.style_masks == [], "strength=0 should skip building masks entirely"
    with torch.no_grad():
        plain0.img.copy_(off0.img)
    plain0._closure()
    off0._closure()
    assert torch.allclose(torch.tensor(plain0._last["style"]), torch.tensor(off0._last["style"]), atol=1e-6)
    print("ok (coherence_mask_strength=0.0 behaves exactly like the feature not existing)")

    # coherence_threshold: should binarise the (already-blurred) mask to exactly {0, 1}
    thresholded = StyleTransfer(content, [style], Config(image_size=64, coherence_mask_strength=1.0,
                                                          coherence_threshold=0.4, device="cpu",
                                                          pretrained=pretrained, verbose=False))
    # the per-layer mask is bilinear-resized from the thresholded field, which smooths
    # {0,1} into a gradient at region boundaries (desirable -- no razor-sharp cliff
    # feeding into the Gram computation) -- check it's still clearly bimodal, not that
    # every value is exactly 0 or 1
    m0 = thresholded.style_masks[0][cfg.style_layers[0]]
    assert m0.min() < 0.05 and m0.max() > 0.95, \
        f"expected the thresholded mask to reach near 0 and near 1, got [{m0.min():.3f}, {m0.max():.3f}]"
    m_thresh = thresholded.step()
    assert all(v == v and abs(v) < float("inf") for v in m_thresh.values()), f"bad losses: {m_thresh}"
    print("ok (coherence_threshold: binarises the mask to {0, 1}, run stays finite)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _selfcheck()
