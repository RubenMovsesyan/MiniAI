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

`coherence_map` + `gram_matrix_masked` / `style_loss_masked` are a fix for a
different problem: Gram matrices are position-blind, so a content region with
no dominant local edge direction (e.g. flat skin) gives style_loss nothing to
align a brush stroke to, and the optimiser settles into a maze/fingerprint
texture instead. `coherence_map` measures, per content pixel, how consistently
its local gradient points in one direction; `style_loss_masked` uses it to
weight down the generated image's contribution to the Gram statistic in
low-coherence regions specifically, rather than changing the loss everywhere.

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


def gram_matrix_masked(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Like `gram_matrix`, but each spatial position's contribution is weighted by
    `mask` ([N,1,H,W], broadcastable, typically in [0,1]) before the channel
    correlations are computed. Normalised by C * sum(mask^2) instead of C*H*W, so
    the result's scale doesn't collapse just because much of the image is masked
    down -- reduces exactly to `gram_matrix` when mask is all ones."""
    n, c, h, w = feat.shape
    f = (feat * mask).reshape(n, c, h * w)
    denom = (mask.reshape(mask.shape[0], 1, h * w).pow(2).sum(dim=-1) * c).clamp_min(1e-8)
    return (f @ f.transpose(1, 2)) / denom.unsqueeze(-1)


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


def style_loss_masked(gen: FeatMap, target_grams: FeatMap, layers: tuple[str, ...],
                       mask: FeatMap, weights: dict[str, float] | None = None) -> torch.Tensor:
    """Like `style_loss`, but the GENERATED image's contribution to each layer's
    Gram matrix is weighted by `mask[layer]` first (see `gram_matrix_masked`) --
    e.g. a content-coherence map (`coherence_map`), resized to match each layer's
    spatial size, so regions the content gives no direction to align a brush
    stroke to pull less hard on matching the style's texture statistics. The
    style images' own targets are unaffected -- masking is a property of THIS
    content image, not the style."""
    total = gen[layers[0]].new_zeros(())
    for l in layers:
        g = gram_matrix_masked(gen[l], mask[l])
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


def coherence_map(img: torch.Tensor) -> torch.Tensor:
    """[N,3,H,W] [0,1] RGB -> [N,1,H,W] structure-tensor orientation coherence,
    in [0,1]: 0 = locally isotropic (nearby gradients point every which way --
    e.g. flat skin with fine random shading, no dominant local edge direction),
    1 = a single dominant local edge direction (e.g. a fabric fold or hair
    strand). Used by `style_loss_masked` (via `gram_matrix_masked`) to damp
    style_loss's pull on the generated image specifically where the content
    gives it no direction to align a brush stroke to -- see artist.py's
    `coherence_mask_strength`."""
    r, g, b = img[..., 0:1, :, :], img[..., 1:2, :, :], img[..., 2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    sobel_x = img.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gp = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx, gy = F.conv2d(gp, sobel_x), F.conv2d(gp, sobel_y)

    sxx, sxy, syy = gx * gx, gx * gy, gy * gy
    blur = img.new_full((1, 1, 9, 9), 1 / 81)
    pad9 = lambda t: F.pad(t, (4, 4, 4, 4), mode="reflect")  # noqa: E731
    sxx, sxy, syy = F.conv2d(pad9(sxx), blur), F.conv2d(pad9(sxy), blur), F.conv2d(pad9(syy), blur)

    return ((sxx - syy) ** 2 + 4 * sxy ** 2).sqrt() / (sxx + syy + 1e-8)


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

    # --- gram_matrix_masked ---
    feat = torch.randn(1, 6, 4, 4)
    ones_mask = torch.ones(1, 1, 4, 4)
    assert torch.allclose(gram_matrix_masked(feat, ones_mask), gram_matrix(feat), atol=1e-6), \
        "all-ones mask should reduce exactly to gram_matrix"

    # a binary mask should match computing gram_matrix on just the kept positions
    binary_mask = torch.zeros(1, 1, 4, 4)
    binary_mask[:, :, :, :2] = 1.0  # keep the left half's columns
    kept = feat[:, :, :, :2].reshape(1, 6, 8, 1)  # same positions, any h,w split works
    assert torch.allclose(gram_matrix_masked(feat, binary_mask), gram_matrix(kept), atol=1e-5), \
        "binary mask should match gram_matrix computed on just the kept positions"
    print("ok (gram_matrix_masked: all-ones == gram_matrix, binary mask == subset gram_matrix)")

    x = torch.randn(1, 6, 4, 4, requires_grad=True)
    gram_matrix_masked(x, torch.rand(1, 1, 4, 4)).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("ok (gram_matrix_masked: grad reaches the features)")

    # --- style_loss_masked ---
    assert torch.allclose(style_loss_masked(feats, tg, ("a", "b"),
                                             {"a": torch.ones(1, 1, 6, 6), "b": torch.ones(1, 1, 5, 5)}),
                           style_loss(feats, tg, ("a", "b")), atol=1e-5), \
        "all-ones mask should reduce exactly to style_loss"

    # masking OUT a mismatched region should lower the loss relative to leaving it in
    mismatched = {"a": feats["a"].clone()}
    mismatched["a"][:, :4] += 5.0  # corrupt half the channels badly
    full_mask = {"a": torch.ones(1, 1, 6, 6)}
    zero_mask = {"a": torch.zeros(1, 1, 6, 6)}
    loss_full = style_loss_masked(mismatched, tg, ("a",), full_mask)
    loss_masked_out = style_loss_masked(mismatched, tg, ("a",), zero_mask)
    assert loss_masked_out < loss_full, \
        f"masking out the mismatched region should reduce the loss: {loss_masked_out} >= {loss_full}"
    print("ok (style_loss_masked: all-ones == style_loss, masking down a region reduces its pull)")

    # --- coherence_map ---
    flat = torch.full((1, 3, 16, 16), 0.4)
    assert torch.allclose(coherence_map(flat), torch.zeros(1, 1, 16, 16), atol=1e-5), \
        "a flat image has no defined local direction -- coherence should read as 0, not NaN"

    # vertical stripes: a clean, single dominant local direction almost everywhere
    stripes = torch.zeros(1, 3, 32, 32)
    stripes[:, :, :, ::4] = 1.0
    stripes[:, :, :, 1::4] = 1.0
    # isotropic noise: comparable raw contrast/edge energy, but no consistent local direction
    torch.manual_seed(0)
    noise = torch.rand(1, 3, 32, 32)

    stripe_coherence = coherence_map(stripes).mean()
    noise_coherence = coherence_map(noise).mean()
    assert stripe_coherence > noise_coherence + 0.1, \
        f"a coherent stripe pattern should score higher than isotropic noise: " \
        f"{stripe_coherence:.3f} vs {noise_coherence:.3f}"
    print(f"ok (coherence_map: flat -> 0, coherent stripes ({stripe_coherence:.3f}) > "
          f"isotropic noise ({noise_coherence:.3f}))")

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

    # --- gradients flow to the image ---
    img = torch.randn(1, 8, 6, 6, requires_grad=True)
    (content_loss({"a": img}, {"a": t[:, :8, :6, :6].detach() * 0}, ("a",))
     + style_loss({"a": img}, tg, ("a",)) + tv_loss(img)).backward()
    assert img.grad is not None and img.grad.abs().sum() > 0
    print("ok (grad reaches image for all three)")
