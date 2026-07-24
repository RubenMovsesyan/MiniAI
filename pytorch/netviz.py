"""Architecture diagram. Traces a dummy forward on any nn.Module -> self-contained HTML/SVG.

    from netviz import render
    render(model)                 # -> network.html
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import torch
import torch.nn as nn

BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, DIM = "#c9d1d9", "#8b949e"
BLUE, GREEN, ORANGE, RED, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"

MAX_NODES, MAX_CH = 9, 16
H, MIDY, GAP, COLH = 620, 320, 34, 210
FONT = 'font-family="ui-sans-serif,system-ui,-apple-system,sans-serif"'


def trace(model: nn.Module, input_shape=(1, 1, 28, 28)) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    records = [{"kind": "Input", "out_shape": tuple(input_shape)}]

    def make_hook(m):
        def hook(mod, inp, out):
            rec = {"kind": type(mod).__name__, "in_shape": tuple(inp[0].shape), "out_shape": tuple(out.shape)}
            if isinstance(mod, nn.Conv2d):
                rec.update(in_ch=mod.in_channels, out_ch=mod.out_channels, k=mod.kernel_size[0])
            elif isinstance(mod, nn.Linear):
                rec.update(in_f=mod.in_features, out_f=mod.out_features)
            elif isinstance(mod, nn.Dropout):
                rec.update(p=mod.p)
            records.append(rec)
        return hook

    handles = [m.register_forward_hook(make_hook(m)) for m in model.modules() if not list(m.children())]
    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    for h in handles:
        h.remove()
    return records


def _text(x, y, s, color=TEXT, size=13, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {FONT}>{html.escape(s)}</text>')


def _box(x, y, w, h, stroke, fill=PANEL, rx=10):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def _channel_grid(cx, cy, n, color=BLUE):
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
    if n == 1:
        return [cy]
    step = h / (n - 1)
    return [cy - h / 2 + i * step for i in range(n)]


def draw_input(x, rec):
    w = bh = 92
    y = MIDY - bh / 2
    c, hh, ww = rec["out_shape"][1], rec["out_shape"][2], rec["out_shape"][3]
    s = _box(x, y, w, bh, DIM)
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
    gx = x + 34
    grid, gw, _ = _channel_grid(gx, MIDY, 9, ORANGE)
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
    in_f, out_f = rec["in_f"], rec["out_f"]
    s = _box(x, y, w, bh, GREEN)
    s += _text(x + w / 2, y - 12, "Output" if is_output else "Dense", GREEN, 13, weight="bold")
    lx, rx = x + 30, x + w - 30
    lys, rys = _col_ys(min(in_f, MAX_NODES)), _col_ys(min(out_f, MAX_NODES))
    for ly in lys:
        for ry in rys:
            s += f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="{GREEN}" stroke-width="0.5" opacity="0.22"/>'
    for ly in lys:
        s += f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.4" fill="{DIM}"/>'
    for ry in rys:
        s += f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="3.8" fill="{GREEN}"/>'
    s += _text(x + w / 2, y + bh + 18, f"{in_f} → {out_f}", DIM, 12)
    return s, w


def draw_dropout(x, rec):
    w, bh = 104, COLH + 8
    y = MIDY - bh / 2
    p = rec["p"]
    s = _box(x, y, w, bh, RED)
    s += _text(x + w / 2, y - 12, "Dropout", RED, 13, weight="bold")
    cx = x + w / 2
    ys = _col_ys(MAX_NODES)
    n_off = round(p * MAX_NODES)
    off = set(range(0, MAX_NODES, max(1, round(MAX_NODES / n_off)))[:n_off]) if n_off else set()
    for i, yy in enumerate(ys):
        if i in off:
            s += (f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="4" fill="none" stroke="{DIM}" stroke-width="1.2" opacity="0.5"/>'
                  f'<line x1="{cx - 4:.1f}" y1="{yy - 4:.1f}" x2="{cx + 4:.1f}" y2="{yy + 4:.1f}" stroke="{RED}" stroke-width="1.2"/>'
                  f'<line x1="{cx - 4:.1f}" y1="{yy + 4:.1f}" x2="{cx + 4:.1f}" y2="{yy - 4:.1f}" stroke="{RED}" stroke-width="1.2"/>')
        else:
            s += f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="4" fill="{RED}"/>'
    s += _text(x + w / 2, y + bh + 18, f"p={p:g}  ({p:.0%} off)", DIM, 12)
    return s, w


_DRAWERS = {"Input": draw_input, "Conv2d": draw_conv, "MaxPool2d": draw_pool,
            "Flatten": draw_flatten, "Dropout": draw_dropout}


def render_svg(records: list[dict]) -> str:
    linear_idxs = [i for i, r in enumerate(records) if r["kind"] == "Linear"]
    last_linear = max(linear_idxs) if linear_idxs else -1
    body, x, prev_right = "", 40, None
    for i, rec in enumerate(records):
        if rec["kind"] == "Linear":
            svg, w = draw_dense(x, rec, is_output=(i == last_linear))
        elif rec["kind"] in _DRAWERS:
            svg, w = _DRAWERS[rec["kind"]](x, rec)
        else:
            continue
        if prev_right is not None:
            body = _arrow(prev_right, x) + body
        body += svg
        prev_right = x + w
        x += w + GAP
    total_w = x - GAP + 40

    title = _text(total_w / 2, 46, f"Network Architecture — {len(records)} layers", TEXT, 22, weight="bold")
    legend = [("Conv2d", BLUE), ("MaxPool", PURPLE), ("Flatten", ORANGE), ("Dense", GREEN), ("Dropout", RED)]
    lg, lx = "", total_w / 2 - len(legend) * 90 / 2
    for name, col in legend:
        lg += f'<rect x="{lx:.1f}" y="{H - 42}" width="13" height="13" rx="3" fill="{col}"/>' + _text(lx + 20, H - 31, name, DIM, 12, anchor="start")
        lx += 90

    defs = (f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" '
            f'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{DIM}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {H}" '
            f'width="100%" height="100%" preserveAspectRatio="xMidYMid meet">{defs}{title}{body}{lg}</svg>')


def render(model: nn.Module, out_path: str = "network.html", input_shape=(1, 1, 28, 28)) -> str:
    svg = render_svg(trace(model, input_shape))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Network Architecture</title>'
           f'<style>html,body{{margin:0;height:100%;overflow:hidden;background:{BG}}}'
           f'.wrap{{height:100vh;display:flex;align-items:center;justify-content:center}}</style>'
           f'</head><body><div class="wrap">{svg}</div></body></html>')
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(Path(out_path).resolve())
