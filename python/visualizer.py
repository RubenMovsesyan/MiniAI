"""Interactive MNIST prediction viewer — correct vs incorrect side by side,
with the conv feature maps the model sees for each image.

Plug your trained model straight in. Two ways:

    python visualizer.py                 # trains via main.main(), then opens the viewer
    # or, from your own code:
    from visualizer import show
    show(my_trained_model)               # skip training, view an existing model

Two panes. Left cycles images the model got RIGHT, right the ones it got WRONG.
Each pane shows: the image, the 10 output logits (0-9), and one grid of feature
maps per convolutional layer (post-ReLU activations for that image). Each pane
has its own "Next" button.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from model import Config
from data import load_split

BLUE, GREEN, RED = "#58a6ff", "#3fb950", "#f85149"
NCOLS = 8   # feature-map grid width


def show(model: torch.nn.Module | None = None,
         cfg: Config | None = None,
         shuffle: bool = True) -> None:
    cfg = cfg or Config.from_env()
    if model is None:
        from train import train
        model = train(cfg)                       # trains and returns the model
    model = model.to(cfg.device).eval()

    x, y = load_split(cfg.data_dir, "test")      # (N, 1, 28, 28) float32, (N,) int64

    # Discover conv layers in order; hook each to grab its activations per image.
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    acts: dict[int, torch.Tensor] = {}
    for k, c in enumerate(convs):
        c.register_forward_hook(lambda mod, inp, out, k=k: acts.__setitem__(k, out.detach()))

    @torch.no_grad()
    def all_logits() -> np.ndarray:
        outs = []
        for i in range(0, len(x), 1000):
            xb = torch.from_numpy(x[i:i + 1000]).to(cfg.device)
            outs.append(model(xb).cpu().numpy())
        return np.concatenate(outs, 0)

    logits = all_logits()
    pred = logits.argmax(1)
    correct_mask = pred == y
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]
    if shuffle:
        np.random.shuffle(correct_idx)
        np.random.shuffle(wrong_idx)

    @torch.no_grad()
    def infer(idx: int):
        xb = torch.from_numpy(x[idx:idx + 1]).to(cfg.device)   # (1, 1, 28, 28)
        lg = model(xb).cpu().numpy().ravel()                   # (10,)  fires the hooks
        maps = [F.relu(acts[k])[0].cpu().numpy() for k in range(len(convs))]  # each (C, 28, 28)
        return lg, maps

    def bar_colors(p: int, t: int) -> list[str]:
        out = []
        for k in range(10):
            if k == p:
                out.append(GREEN if k == t else RED)
            elif k == t:
                out.append(GREEN)
            else:
                out.append(BLUE)
        return out

    # ── Layout: two pane-subfigures; each stacks [image+bar] / conv grids / button.
    fig = plt.figure(figsize=(14, 9))
    conv_rows = [int(np.ceil(c.out_channels / NCOLS)) for c in convs]
    acc = correct_mask.mean()
    fig.suptitle(f"test accuracy: {len(correct_idx)}/{len(y)} = {acc:.2%}   "
                 f"({len(wrong_idx)} wrong)", fontsize=13)

    pane_sfigs = fig.subfigures(1, 2, wspace=0.04)
    panes = []
    for name, idxs, sf in [("Correct", correct_idx, pane_sfigs[0]),
                           ("Incorrect", wrong_idx, pane_sfigs[1])]:
        ratios = [2.6] + [max(r, 1) * 0.85 for r in conv_rows] + [0.45]
        rows = sf.subfigures(len(ratios), 1, height_ratios=ratios)

        ax_img, ax_bar = rows[0].subplots(1, 2)

        conv_axes = []
        for k, c in enumerate(convs):
            grid = rows[1 + k].subplots(conv_rows[k], NCOLS, squeeze=False)
            rows[1 + k].suptitle(f"conv{k + 1}: {c.out_channels} channels", fontsize=9)
            conv_axes.append([grid[a][b] for a in range(conv_rows[k]) for b in range(NCOLS)])

        btn = Button(rows[-1].add_axes([0.3, 0.1, 0.4, 0.75]), f"Next {name.lower()} ▶")
        panes.append({"name": name, "idx": idxs, "pos": 0,
                      "img": ax_img, "bar": ax_bar, "conv_axes": conv_axes, "btn": btn})

    def draw_pane(pane: dict) -> None:
        ax_img, ax_bar, lst = pane["img"], pane["bar"], pane["idx"]
        ax_img.clear(); ax_bar.clear()
        for grid in pane["conv_axes"]:
            for a in grid:
                a.clear(); a.axis("off")

        if len(lst) == 0:
            ax_img.set_title(f"{pane['name']}  0/0")
            ax_img.text(0.5, 0.5, "none", ha="center", va="center", color=BLUE)
            ax_img.axis("off"); ax_bar.axis("off")
            fig.canvas.draw_idle()
            return

        idx = int(lst[pane["pos"]])
        lg, maps = infer(idx)
        p, t = int(lg.argmax()), int(y[idx])

        ax_img.imshow(x[idx].squeeze(), cmap="gray")
        ax_img.set_title(f"{pane['name']}  {pane['pos'] + 1}/{len(lst)}   #{idx}",
                         color=GREEN if pane["name"] == "Correct" else RED)
        ax_img.axis("off")

        ax_bar.bar(range(10), lg, color=bar_colors(p, t))
        ax_bar.axhline(0, color="#888", linewidth=0.6)
        ax_bar.set_xticks(range(10))
        ax_bar.set_title(f"pred: {p}   true: {t}", fontsize=9)

        for k, grid in enumerate(pane["conv_axes"]):
            fmap = maps[k]                         # (C, 28, 28)
            for ch in range(fmap.shape[0]):
                grid[ch].imshow(fmap[ch], cmap="viridis")
                grid[ch].axis("off")

        fig.canvas.draw_idle()

    def make_next(pane: dict):
        def _next(_event=None) -> None:
            if len(pane["idx"]):
                pane["pos"] = (pane["pos"] + 1) % len(pane["idx"])
                draw_pane(pane)
        return _next

    for pane in panes:
        pane["btn"].on_clicked(make_next(pane))
    fig._viz_buttons = [p["btn"] for p in panes]   # keep widgets live

    for pane in panes:
        draw_pane(pane)
    plt.show()


if __name__ == "__main__":
    show()
