"""Run a trained TransformNet checkpoint on new images -- the feed-forward
INFERENCE side of train_style.py. One forward pass, no VGG, no optimisation:
this is the actual mobile-deployment path (minus exporting to a mobile runtime).

`--depthwise` must match however the checkpoint was trained (train_style.py's
`--depthwise`) -- the architecture has to be told, not inferred. Checkpoints are
saved either as a bare state_dict (older files, and stylize's own weight-only
export) or as a {"net", "opt", "step"} bundle (train_style.py's checkpoints,
which also carry optimizer state for --continue-from) -- load_checkpoint accepts
both and only ever needs the weights.

`--preserve-color` fixes a style-heavy global palette (e.g. Van Gogh's blue/
yellow) getting imposed on content it doesn't belong on: the network's output
keeps its stylised luminance (brush texture lives almost entirely there) but
its colour is replaced with the original photo's own colour (see
images.preserve_color). No retraining needed -- purely a post-processing step.

Run from inside pytorch/:
    python -m projects.artist.stylize <checkpoint.pt> <image.jpg> [-o out.png] [--depthwise] [--preserve-color]
"""

from __future__ import annotations

import argparse
import sys

import torch

from projects.artist.images import load_image, preserve_color, save_image
from projects.artist.transform import TransformNet, crop_to_multiple


def load_checkpoint(path: str, depthwise: bool = False) -> TransformNet:
    """Build a TransformNet matching how it was trained and load its weights."""
    net = TransformNet(depthwise=depthwise)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    net.load_state_dict(state)
    return net


def stylize(net: TransformNet, image: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Run one photo (CHW [0,1], any size) through a trained TransformNet.
    Returns CHW [0,1] -- possibly a few pixels smaller (see crop_to_multiple)."""
    device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
    net = net.to(device).eval()
    image = crop_to_multiple(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(image)
    return out[0].cpu()


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="stylize", description="apply a trained style network to an image")
    p.add_argument("checkpoint")
    p.add_argument("image")
    p.add_argument("-o", "--out", default="stylized.png")
    p.add_argument("--size", type=int, default=1024, help="longer side; the network runs at any size")
    p.add_argument("--depthwise", action="store_true", help="must match how the checkpoint was trained")
    p.add_argument("--preserve-color", action="store_true", dest="preserve_color",
                   help="keep the stylised brush texture but restore the original photo's own colour")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    net = load_checkpoint(a.checkpoint, depthwise=a.depthwise)
    image = load_image(a.image, a.size)
    out = stylize(net, image, device=a.device)

    if tuple(out.shape[-2:]) != tuple(image.shape[-2:]):
        print(f"note: cropped {tuple(image.shape[-2:])} -> {tuple(out.shape[-2:])} (multiple of 4)")
    if a.preserve_color:
        out = preserve_color(out, crop_to_multiple(image))  # same crop stylize() applied, for shape-matching
    path = save_image(out, a.out)
    print("wrote", path)


def _selfcheck() -> None:
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    torch.manual_seed(0)
    net = TransformNet()
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "net.pt"
        torch.save(net.state_dict(), ckpt)

        loaded = load_checkpoint(str(ckpt), depthwise=False)
        for p1, p2 in zip(net.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2), "checkpoint round-trip changed weights"
        print("ok (checkpoint save/load round-trips)")

        # train_style.py's checkpoints are a {"net", "opt", "step"} bundle, not a bare
        # state_dict -- load_checkpoint must accept that shape too
        bundle_ckpt = Path(d) / "bundle.pt"
        torch.save({"net": net.state_dict(), "opt": {}, "step": 123}, bundle_ckpt)
        loaded_bundle = load_checkpoint(str(bundle_ckpt), depthwise=False)
        for p1, p2 in zip(net.parameters(), loaded_bundle.parameters()):
            assert torch.equal(p1, p2), "bundled-checkpoint weights don't match"
        print("ok (accepts train_style.py's {net, opt, step} checkpoint bundle too)")

        src = Path(d) / "in.jpg"
        Image.fromarray(np.random.randint(0, 256, (67, 101, 3), dtype=np.uint8)).save(src)
        image = load_image(src, size=101)
        out = stylize(loaded, image, device="cpu")
        assert out.shape[0] == 3 and out.min() >= 0.0 and out.max() <= 1.0, "bad stylize output"
        assert out.shape[-2:] != image.shape[-2:] or all(d % 4 == 0 for d in image.shape[-2:]), \
            "should only differ in shape if input wasn't a multiple of 4"
        print(f"ok (stylize: {tuple(image.shape[-2:])} -> {tuple(out.shape[-2:])})")

        out_path = save_image(out, Path(d) / "out.png")
        assert out_path.exists()
        print("ok (save_image writes the result)")

        # --preserve-color, exercised through main()'s real CLI path: output colour
        # should track the ORIGINAL content image, not the (randomly-initialised,
        # arbitrary-colour-shifting) network's raw output
        content_cropped = crop_to_multiple(image)
        plain_out = stylize(loaded, image, device="cpu")
        recolored_path = Path(d) / "recolored.png"
        main([str(ckpt), str(src), "-o", str(recolored_path), "--size", "101", "--preserve-color"])
        # load the saved file's exact pixels, no further resizing (load_image would resize again)
        recolored = torch.from_numpy(
            np.asarray(Image.open(recolored_path), dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        from projects.artist.images import to_ycbcr
        # mean, not per-pixel, tolerance: a few pixels can legitimately need clamping
        # (an out-of-gamut Y/chrominance combination -- see images.py's self-check),
        # which is expected and not what this check is verifying
        chroma_diff = (to_ycbcr(recolored.unsqueeze(0))[:, 1:3]
                       - to_ycbcr(content_cropped.unsqueeze(0))[:, 1:3]).abs().mean()
        assert chroma_diff < 1e-3, f"--preserve-color should restore the content image's colour: {chroma_diff}"
        assert not torch.allclose(recolored, plain_out, atol=1e-3), \
            "--preserve-color should actually change something vs. the plain output"
        print("ok (--preserve-color: restores content colour via the real CLI path)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _selfcheck()
