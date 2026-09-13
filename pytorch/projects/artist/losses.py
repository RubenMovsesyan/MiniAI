"""Style-transfer losses (Gatys, Ecker & Bethge 2015).

    content = MSE(F_gen, F_content)                      at a deep layer — preserves layout
    style   = sum_l MSE(Gram(F_gen^l), Gram_target^l)    shallow..deep — matches texture
    tv      = anisotropic total variation of the image   — suppresses noise / speckle
    color   = MSE(chrominance(gen), chrominance(content)) — keeps the content's own colour

`gen` / `target` are the `{layer_name: activation}` dicts that `VGGFeatures.forward`
returns. Style targets are Gram matrices, precomputed once via `gram_matrices` since
the style image never changes. Detaching the targets is the caller's job.

`color_loss` (unlike the other three) is a soft, additive alternative to
`images.preserve_color`'s hard post-hoc swap: instead of stitching the stylised
luminance onto the content's chrominance after the fact (which can clip out-of-
gamut RGB for a few pixels), it pulls the optimiser toward finding one
self-consistent image that already satisfies both the style and the colour
target together.

Run from inside pytorch/:  python -m projects.artist.losses
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from projects.artist.images import to_ycbcr

FeatMap = dict[str, torch.Tensor]


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """[N,C,H,W] activation -> [N,C,C] Gram matrix (channel correlations),
    normalised by C*H*W so it doesn't scale with feature-map size."""
    n, c, h, w = feat.shape
    f = feat.reshape(n, c, h * w)
    return (f @ f.transpose(1, 2)) / (c * h * w)


def gram_matrices(feats: FeatMap, layers: tuple[str, ...]) -> FeatMap:
    """{layer: activation} -> {layer: Gram} for `layers`. Build the style targets once."""
    return {l: gram_matrix(feats[l]) for l in layers}


def content_loss(gen: FeatMap, target: FeatMap, layers: tuple[str, ...]) -> torch.Tensor:
    """Mean-squared error between generated and content activations, summed over `layers`."""
    return sum(F.mse_loss(gen[l], target[l]) for l in layers)


def style_loss(gen: FeatMap, target_grams: FeatMap, layers: tuple[str, ...],
               weights: dict[str, float] | None = None) -> torch.Tensor:
    """MSE between the generated image's Gram matrices and the precomputed style
    Grams, summed over `layers`. `weights` optionally scales each layer's term
    (default 1.0 each).

    `gen` may carry a real batch (many images being trained at once, e.g. in
    train_style.py) while `target_grams` is always a single style image (batch 1)
    -- the same one target compared against every image in the batch. That's
    done via an explicit `expand_as`, not implicit broadcasting inside
    `F.mse_loss`, which otherwise warns (and is fragile: it can't tell an
    intentional batch-broadcast from an accidental shape mismatch)."""
    total = gen[layers[0]].new_zeros(())
    for l in layers:
        g = gram_matrix(gen[l])
        w = 1.0 if weights is None else weights[l]
        total = total + w * F.mse_loss(g, target_grams[l].expand_as(g))
    return total


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a CHW/NCHW image: mean abs difference of
    vertically and horizontally neighbouring pixels."""
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
    return dh + dw


def color_loss(gen_img: torch.Tensor, content_img: torch.Tensor) -> torch.Tensor:
    """MSE between gen_img's and content_img's chrominance (Cb, Cr) -- a direct,
    pixel-space pull toward the content's actual colour. Leaves luminance (and
    therefore the VGG-driven texture/style) untouched. [0,1], CHW or NCHW, both
    the same shape."""
    return F.mse_loss(to_ycbcr(gen_img)[..., 1:3, :, :], to_ycbcr(content_img)[..., 1:3, :, :])


def color_tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of just img's chrominance (Cb, Cr) -- same
    idea as `tv_loss`, but restricted to colour so it never smooths the
    luminance-driven brush texture. A strong color_loss pull can correct each
    pixel's colour almost independently of its neighbours (no receptive field,
    no smoothness prior), breaking coherent strokes into a bubbly/cellular
    texture even when judged at multiple eval_size scales, since a flat
    (low-gradient) region of the content gives style_loss nothing to lock a
    stroke direction onto either. This penalises adjacent pixels having
    different colour directly, wherever that happens, regardless of scale.
    [0,1], CHW or NCHW."""
    cbcr = to_ycbcr(img)[..., 1:3, :, :]
    dh = (cbcr[..., 1:, :] - cbcr[..., :-1, :]).abs().mean()
    dw = (cbcr[..., :, 1:] - cbcr[..., :, :-1]).abs().mean()
    return dh + dw


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- gram_matrix ---
    g = gram_matrix(torch.randn(2, 4, 5, 6))
    assert g.shape == (2, 4, 4), g.shape
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6), "not symmetric"
    assert torch.allclose(gram_matrix(torch.ones(1, 3, 2, 2)), torch.full((1, 3, 3), 1 / 3)), "value"
    x = torch.randn(1, 8, 4, 4)
    assert torch.allclose(gram_matrix(x), gram_matrix(x.repeat(1, 1, 2, 1)), atol=1e-6), "scale-variant"
    print("ok (gram_matrix: shape, symmetry, value, scale-invariance)")

    # --- content_loss ---
    t = torch.randn(1, 16, 8, 8)
    assert content_loss({"a": t}, {"a": t.clone()}, ("a",)) == 0
    two = content_loss({"a": t, "b": t}, {"a": t + 1, "b": t + 1}, ("a", "b"))
    assert torch.allclose(two, torch.tensor(2.0)), two  # mse of a constant-1 diff, twice
    print("ok (content_loss: zero on match, sums layers)")

    # --- style_loss ---
    feats = {"a": torch.randn(1, 8, 6, 6), "b": torch.randn(1, 8, 5, 5)}
    tg = gram_matrices(feats, ("a", "b"))
    assert style_loss(feats, tg, ("a", "b")) == 0, "zero when gen matches target"
    base = style_loss({"a": feats["a"] + 0.5}, tg, ("a",))
    dbl = style_loss({"a": feats["a"] + 0.5}, tg, ("a",), weights={"a": 2.0})
    assert torch.allclose(dbl, 2 * base), "per-layer weight"
    print("ok (style_loss: zero on match, per-layer weight)")

    # batch>1 gen vs batch-1 target (train_style.py's real shape) -- must not warn, and
    # must equal comparing each batch element to the same target individually
    import warnings
    batched = {"a": feats["a"].repeat(3, 1, 1, 1) + torch.randn(3, 8, 6, 6) * 0.1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning (e.g. mismatched-size broadcast) fails the test
        loss_batched = style_loss(batched, tg, ("a",))
    expected = sum(style_loss({"a": batched["a"][i:i + 1]}, tg, ("a",)) for i in range(3)) / 3
    assert torch.allclose(loss_batched, expected, atol=1e-6), (loss_batched, expected)
    print("ok (style_loss: batch>1 gen vs batch-1 target, no warning, matches per-image average)")

    # --- tv_loss ---
    assert tv_loss(torch.ones(1, 3, 8, 8)) == 0
    assert tv_loss(torch.randn(1, 3, 8, 8)) > 0
    print("ok (tv_loss: zero on flat, positive on noise)")

    # --- color_loss ---
    from projects.artist.images import to_rgb

    rgb = torch.rand(1, 3, 8, 8)
    assert color_loss(rgb, rgb) == 0, "zero when images match exactly"
    ycbcr = to_ycbcr(rgb)
    shifted = to_rgb(torch.cat([ycbcr[:, 0:1], ycbcr[:, 1:3] + 0.05], dim=1)).clamp(0, 1)
    assert color_loss(shifted, rgb) > 0, "different chrominance (same luminance) should be nonzero"
    print("ok (color_loss: zero on match, positive when only chrominance differs)")

    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    color_loss(x, torch.rand(1, 3, 8, 8)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("ok (color_loss: grad reaches the image)")

    # --- color_tv_loss ---
    flat = torch.full((1, 3, 8, 8), 0.4)
    assert color_tv_loss(flat) == 0, "flat colour -> zero"
    # a checkerboard in Cb/Cr only (luminance untouched) should be nonzero...
    y0 = to_ycbcr(flat)[:, 0:1]
    noisy_cbcr = torch.rand(1, 2, 8, 8)
    noisy = to_rgb(torch.cat([y0, noisy_cbcr], dim=1)).clamp(0, 1)
    assert color_tv_loss(noisy) > 0, "varying chrominance -> positive"
    # ...but pure LUMINANCE noise (same flat colour) should still read as zero,
    # since color_tv_loss only ever looks at Cb/Cr
    y_noisy = to_ycbcr(flat)[:, 0:1] + (torch.rand(1, 1, 8, 8) - 0.5) * 0.5
    luminance_only_noise = to_rgb(torch.cat([y_noisy, to_ycbcr(flat)[:, 1:3]], dim=1)).clamp(0, 1)
    assert torch.allclose(color_tv_loss(luminance_only_noise), torch.tensor(0.0), atol=1e-5), \
        "luminance-only noise must not register -- color_tv_loss should ignore brightness/texture"
    print("ok (color_tv_loss: zero on flat/luminance-only noise, positive on chrominance noise)")

    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    color_tv_loss(x).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("ok (color_tv_loss: grad reaches the image)")

    # --- gradients flow to the image ---
    img = torch.randn(1, 8, 6, 6, requires_grad=True)
    (content_loss({"a": img}, {"a": t[:, :8, :6, :6].detach() * 0}, ("a",))
     + style_loss({"a": img}, tg, ("a",)) + tv_loss(img)).backward()
    assert img.grad is not None and img.grad.abs().sum() > 0
    print("ok (grad reaches image for all three)")
