"""Generate a self-contained HTML/SVG diagram of the network architecture.

Reads the layers off a live Net (nn.Module) by tracing a dummy forward pass, so
model.py is never touched. Horizontal left->right pipeline; the SVG viewBox +
preserveAspectRatio letterboxes the whole diagram into the viewport, so it scales
to the page width and never needs scrolling.

    python netviz.py                       # render current architecture -> network.html
    from netviz import render
    render(trained_model)                  # from your own code
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import torch
import torch.nn as nn

from model import Config, Net

# ── palette (from python/docs/style.css) ───────────────────────────────────────
BG, PANEL, PANEL2, BORDER = "#0d1117", "#161b22", "#1c2230", "#30363d"
TEXT, DIM = "#c9d1d9", "#8b949e"
BLUE, GREEN, ORANGE, RED, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"

MAX_NODES = 9      # sampled nodes drawn per dense column
MAX_CH = 16        # channel squares drawn per conv grid
H = 620            # viewBox height
MIDY = 320         # vertical center of the layer band
GAP = 34           # horizontal space (arrow) between blocks
COLH = 210         # height of a dense node column
FONT = 'font-family="ui-sans-serif,system-ui,-apple-system,sans-serif"'


# ── 1. trace the model to an ordered layer list ────────────────────────────────
def trace(model: nn.Module, input_shape=(1, 1, 28, 28)) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    records: list[dict] = [{"kind": "Input", "out_shape": tuple(input_shape)}]

    def make_hook(mod):
        def hook(m, inp, out):
            rec = {"kind": type(m).__name__,
                   "in_shape": tuple(inp[0].shape), "out_shape": tuple(out.shape)}
            if isinstance(m, nn.Conv2d):
                rec.update(in_ch=m.in_channels, out_ch=m.out_channels, k=m.kernel_size[0])
            elif isinstance(m, nn.Linear):
                rec.update(in_f=m.in_features, out_f=m.out_features)
            elif isinstance(m, nn.Dropout):
                rec.update(p=m.p)
            records.append(rec)
        return hook

    handles = [m.register_forward_hook(make_hook(m))
               for m in model.modules() if not list(m.children())]
    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    for h in handles:
        h.remove()
    return records


# ── 2. small SVG primitives ─────────────────────────────────────────────────────
def _text(x, y, s, color=TEXT, size=13, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {FONT}>{html.escape(s)}</text>')


def _box(x, y, w, h, stroke, fill=PANEL, rx=10):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def _channel_grid(cx, cy, n, color=BLUE):
    """Grid of up to MAX_CH squares, centered at (cx, cy). Returns (svg, w, h)."""
    shown = min(n, MAX_CH)
    cols = max(1, math.ceil(math.sqrt(shown)))
    rows = math.ceil(shown / cols)
    sq, g = 9, 2
    gw, gh = cols * (sq + g) - g, rows * (sq + g) - g
    x0, y0 = cx - gw / 2, cy - gh / 2
    s = ""
    for i in range(shown):
        r, c = divmod(i, cols)
        s += (f'<rect x="{x0 + c * (sq + g):.1f}" y="{y0 + r * (sq + g):.1f}" '
              f'width="{sq}" height="{sq}" rx="1.5" fill="{color}" opacity="0.85"/>')
    return s, gw, gh


def _arrow(x1, x2, color=DIM, y=MIDY):
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 7:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"/>')


def _col_ys(n, cy=MIDY, h=COLH):
    """y positions for n nodes spread over a column of height h."""
    if n == 1:
        return [cy]
    step = h / (n - 1)
    return [cy - h / 2 + i * step for i in range(n)]


# ── 3. per-layer block drawers: each returns (svg, width) centered at MIDY ──────
def draw_input(x, rec):
    w, bh = 92, 92
    y = MIDY - bh / 2
    c, hh, ww = rec["out_shape"][1], rec["out_shape"][2], rec["out_shape"][3]
    s = _box(x, y, w, bh, DIM)
    # faint pixel grid to read as an image
    for i in range(1, 4):
        s += (f'<line x1="{x + i * w / 4:.1f}" y1="{y + 8}" x2="{x + i * w / 4:.1f}" y2="{y + bh - 8}" stroke="{BORDER}" stroke-width="1"/>'
              f'<line x1="{x + 8}" y1="{y + i * bh / 4:.1f}" x2="{x + w - 8}" y2="{y + i * bh / 4:.1f}" stroke="{BORDER}" stroke-width="1"/>')
    s += _text(x + w / 2, y - 12, "Input", DIM, 12, weight="bold")
    s += _text(x + w / 2, y + bh + 18, f"{hh}×{ww}×{c}", TEXT, 13)
    return s, w


def draw_conv(x, rec):
    w, bh = 172, 150
    y = MIDY - bh / 2
    s = _box(x, y, w, bh, BLUE)
    s += _text(x + w / 2, y - 12, "Conv2d", BLUE, 13, weight="bold")
    gin, _, _ = _channel_grid(x + 44, MIDY, rec["in_ch"])
    gout, _, _ = _channel_grid(x + w - 44, MIDY, rec["out_ch"])
    s += gin + _arrow(x + w / 2 - 16, x + w / 2 + 16, BLUE) + gout
    s += _text(x + 44, MIDY + 56, f'in {rec["in_ch"]}', DIM, 11)
    s += _text(x + w - 44, MIDY + 56, f'out {rec["out_ch"]}', DIM, 11)
    s += _text(x + w / 2, y + bh + 18, f'{rec["k"]}×{rec["k"]} kernel', DIM, 12)
    return s, w


def draw_pool(x, rec):
    w, bh = 78, 96
    y = MIDY - bh / 2
    a, b = rec["in_shape"][2], rec["out_shape"][2]
    s = _box(x, y, w, bh, PURPLE)
    s += _text(x + w / 2, y - 12, "MaxPool", PURPLE, 12, weight="bold")
    s += _text(x + w / 2, MIDY - 4, "↓2", PURPLE, 26, weight="bold")
    s += _text(x + w / 2, MIDY + 22, f"{a}→{b}", DIM, 11)
    return s, w


def draw_flatten(x, rec):
    w, bh = 148, 200
    y = MIDY - bh / 2
    s = _box(x, y, w, bh, ORANGE)
    s += _text(x + w / 2, y - 12, "Flatten", ORANGE, 13, weight="bold")
    # left: small 2D grid ; right: 1D column ; fan lines between
    gx, gy = x + 34, MIDY
    grid, gw, gh = _channel_grid(gx, gy, 9, ORANGE)
    s += grid
    col_x = x + w - 26
    ys = _col_ys(7, cy=MIDY, h=150)
    for yy in ys:
        s += f'<line x1="{gx + gw / 2:.1f}" y1="{MIDY:.1f}" x2="{col_x:.1f}" y2="{yy:.1f}" stroke="{ORANGE}" stroke-width="0.7" opacity="0.5"/>'
    for yy in ys:
        s += f'<circle cx="{col_x:.1f}" cy="{yy:.1f}" r="3.2" fill="{ORANGE}"/>'
    s += _text(x + w / 2, y + bh + 18, f'2D → 1D  ({rec["out_shape"][1]})', DIM, 12)
    return s, w


def draw_dense(x, rec, is_output=False):
    w, bh = 158, COLH + 8
    y = MIDY - bh / 2
    color = GREEN
    in_f, out_f = rec["in_f"], rec["out_f"]
    s = _box(x, y, w, bh, color)
    title = "Output" if is_output else "Dense"
    s += _text(x + w / 2, y - 12, title, color, 13, weight="bold")

    lx, rx = x + 30, x + w - 30
    lys = _col_ys(min(in_f, MAX_NODES))
    rys = _col_ys(min(out_f, MAX_NODES))
    # webbing: every shown left node -> every shown right node
    for ly in lys:
        for ry in rys:
            s += f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="{color}" stroke-width="0.5" opacity="0.22"/>'
    for ly in lys:
        s += f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.4" fill="{DIM}"/>'
    for ry in rys:
        s += f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="3.8" fill="{color}"/>'
    s += _text(x + w / 2, y + bh + 18, f"{in_f} → {out_f}", DIM, 12)
    return s, w


def draw_dropout(x, rec):
    w, bh = 104, COLH + 8
    y = MIDY - bh / 2
    p = rec["p"]
    s = _box(x, y, w, bh, RED)
    s += _text(x + w / 2, y - 12, "Dropout", RED, 13, weight="bold")
    cx = x + w / 2
    n = MAX_NODES
    ys = _col_ys(n)
    n_off = round(p * n)
    off = set(range(0, n, max(1, round(n / n_off)))[:n_off]) if n_off else set()
    for i, yy in enumerate(ys):
        if i in off:
            s += (f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="4" fill="none" stroke="{DIM}" stroke-width="1.2" opacity="0.5"/>'
                  f'<line x1="{cx - 4:.1f}" y1="{yy - 4:.1f}" x2="{cx + 4:.1f}" y2="{yy + 4:.1f}" stroke="{RED}" stroke-width="1.2"/>'
                  f'<line x1="{cx - 4:.1f}" y1="{yy + 4:.1f}" x2="{cx + 4:.1f}" y2="{yy - 4:.1f}" stroke="{RED}" stroke-width="1.2"/>')
        else:
            s += f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="4" fill="{RED}"/>'
    s += _text(x + w / 2, y + bh + 18, f"p={p:g}  ({p:.0%} off)", DIM, 12)
    return s, w


DRAWERS = {
    "Input": draw_input, "Conv2d": draw_conv, "MaxPool2d": draw_pool,
    "Flatten": draw_flatten, "Linear": draw_dense, "Dropout": draw_dropout,
}


# ── 4. assemble the full SVG + HTML ─────────────────────────────────────────────
def render_svg(records: list[dict]) -> str:
    # last Linear is the output head
    last_linear = max(i for i, r in enumerate(records) if r["kind"] == "Linear")
    body, x = "", 40
    prev_right = None
    for i, rec in enumerate(records):
        if rec["kind"] == "Linear":
            svg, w = draw_dense(x, rec, is_output=(i == last_linear))
        else:
            svg, w = DRAWERS[rec["kind"]](x, rec)
        if prev_right is not None:
            body = _arrow(prev_right, x) + body   # connector behind blocks
        body += svg
        prev_right = x + w
        x += w + GAP
    total_w = x - GAP + 40

    # title + legend inside the viewBox so everything scales together
    title = _text(total_w / 2, 46, f"Network Architecture — {len(records)} layers", TEXT, 22, weight="bold")
    legend_items = [("Conv2d", BLUE), ("MaxPool", PURPLE), ("Flatten", ORANGE),
                    ("Dense", GREEN), ("Dropout", RED)]
    lg, lx = "", total_w / 2 - len(legend_items) * 80 / 2
    for name, col in legend_items:
        lg += (f'<rect x="{lx:.1f}" y="{H - 42}" width="13" height="13" rx="3" fill="{col}"/>'
               + _text(lx + 20, H - 31, name, DIM, 12, anchor="start"))
        lx += 90

    defs = (f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" '
            f'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{DIM}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {H}" '
            f'width="100%" height="100%" preserveAspectRatio="xMidYMid meet">'
            f'{defs}{title}{body}{lg}</svg>')


def render(model: nn.Module, out_path: str = "network.html", input_shape=(1, 1, 28, 28)) -> str:
    svg = render_svg(trace(model, input_shape))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Network Architecture</title>'
           f'<style>html,body{{margin:0;height:100%;overflow:hidden;background:{BG}}}'
           f'.wrap{{height:100vh;display:flex;align-items:center;justify-content:center}}</style>'
           f'</head><body><div class="wrap">{svg}</div></body></html>')
    path = Path(out_path)
    path.write_text(doc, encoding="utf-8")
    return str(path.resolve())


if __name__ == "__main__":
    cfg = Config.from_env()
    # Structure only — weights don't matter for the diagram, so no training needed.
    out = render(Net(cfg))
    print(f"wrote {out}")
