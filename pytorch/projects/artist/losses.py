"""Style-transfer losses (Gatys, Ecker & Bethge 2015).

    content = MSE(F_gen, F_content)                      at a deep layer — preserves layout
    style   = sum_l MSE(Gram(F_gen^l), Gram_target^l)    shallow..deep — matches texture
    tv      = anisotropic total variation of the image   — suppresses noise / speckle

`gen` / `target` are the `{layer_name: activation}` dicts that `VGGFeatures.forward`
returns. Style targets are Gram matrices, precomputed once via `gram_matrices` since
the style image never changes. Detaching the targets is the caller's job.

Run from inside pytorch/:  python -m projects.artist.losses
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

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
    (default 1.0 each)."""
    return sum(
        (1.0 if weights is None else weights[l]) * F.mse_loss(gram_matrix(gen[l]), target_grams[l])
        for l in layers
    )


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a CHW/NCHW image: mean abs difference of
    vertically and horizontally neighbouring pixels."""
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
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

    # --- tv_loss ---
    assert tv_loss(torch.ones(1, 3, 8, 8)) == 0
    assert tv_loss(torch.randn(1, 3, 8, 8)) > 0
    print("ok (tv_loss: zero on flat, positive on noise)")

    # --- gradients flow to the image ---
    img = torch.randn(1, 8, 6, 6, requires_grad=True)
    (content_loss({"a": img}, {"a": t[:, :8, :6, :6].detach() * 0}, ("a",))
     + style_loss({"a": img}, tg, ("a",)) + tv_loss(img)).backward()
    assert img.grad is not None and img.grad.abs().sum() > 0
    print("ok (grad reaches image for all three)")
