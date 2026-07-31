"""Architecture diagram. Traces a dummy forward on any nn.Module -> self-contained HTML/SVG.

    from utils import netviz
    netviz.render(model)                              # -> network.html
    netviz.render(model, input_shape=(1, 3, 640, 640))

Tracing happens at *block* granularity: a module whose type name has a drawer in
`_DRAWERS` is drawn as one node and not descended into, so YOLO's 346 leaf modules
render as ~50 blocks. `torch.cat` / `+` between blocks are captured too, so the
FPN/PAN skip connections show up as real edges instead of vanishing.

The result is a layered DAG (columns = depth from input), pan/zoom in the browser.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, DIM = "#c9d1d9", "#8b949e"
BLUE, GREEN, ORANGE, RED, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"
TEAL, PINK = "#39c5cf", "#f778ba"

MAX_NODES, MAX_CH = 9, 16
FONT = 'font-family="ui-sans-serif,system-ui,-apple-system,sans-serif"'

TITLE_H, SUB_H, SUB_LINE = 22, 22, 15   # label bands above/below every node box
GAP_X, GAP_Y, MARGIN = 46, 30, 48       # column gap, row gap, canvas margin
HEAD_H = 120                            # title + legend band at the top of the canvas


# --- trace --------------------------------------------------------------------


def _blocks_of(mod) -> list[nn.Module]:
    b = getattr(mod, "blocks", None)
    if b is None:
        return []
    return list(b.children()) if isinstance(b, nn.Sequential) else [b]


def _act_name(mod) -> str:
    act = getattr(mod, "act", None)
    return type(act).__name__ if act is not None else ""


# per-kind extra fields pulled off the live module (shapes are added by the tracer)
_EXTRACT = {
    "Conv2d": lambda m: dict(in_ch=m.in_channels, out_ch=m.out_channels, k=m.kernel_size[0],
                             stride=m.stride[0], groups=m.groups),
    "Linear": lambda m: dict(in_f=m.in_features, out_f=m.out_features),
    "Dropout": lambda m: dict(p=m.p),
    "Upsample": lambda m: dict(scale=m.scale_factor, mode=m.mode),
    "Conv": lambda m: dict(in_ch=m.conv.in_channels, out_ch=m.conv.out_channels,
                           k=m.conv.kernel_size[0], stride=m.conv.stride[0],
                           groups=m.conv.groups, act=_act_name(m)),
    "DWConv": lambda m: dict(in_ch=m.depthwise.conv.in_channels,
                             out_ch=m.pointwise.conv.out_channels,
                             k=m.depthwise.conv.kernel_size[0], act=_act_name(m.depthwise)),
    "C3k": lambda m: dict(n_layers=len(m.layers), add=m.add,
                          ks=[c.conv.kernel_size[0] for c in m.layers]),
    "PSA": lambda m: dict(hidden=m.hidden, dpu_aware=m.dpu_aware, act=_act_name(m)),
    "SPPF": lambda m: dict(hidden=m.cv1.conv.out_channels, k=m.pool.kernel_size),
    "C3k2": lambda m: dict(n_blocks=len(_blocks_of(m)), hidden=m.cv1.conv.out_channels,
                           inner=type(_blocks_of(m)[0]).__name__ if _blocks_of(m) else "?",
                           bypass="conv"),
    "C2PSA": lambda m: dict(n_blocks=len(_blocks_of(m)), hidden=m.cv1.conv.out_channels,
                            inner=type(_blocks_of(m)[0]).__name__ if _blocks_of(m) else "?",
                            bypass="raw"),
}

_OPS = {"cat": "Concat", "concat": "Concat", "add": "Add", "__add__": "Add"}


def trace(model: nn.Module, input_shape=(1, 1, 28, 28)) -> tuple[list[dict], dict]:
    """Run one dummy forward, return (nodes, meta).

    A node is {idx, kind, ins:[parent idx], out_shape, ...kind-specific}. Only modules
    with a drawer are emitted; anything else passes its producer through so edges stay
    connected (a bare ReLU between two drawn layers doesn't break the chain).
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(*input_shape, device=device)

    nodes = [{"idx": 0, "kind": "Input", "ins": [], "out_shape": tuple(input_shape)}]
    producer: dict[int, int] = {}
    keep: dict[int, torch.Tensor] = {}   # hold refs so ids can't be reused mid-trace
    depth = [0]

    def claim(t, idx):
        producer[id(t)] = idx
        keep[id(t)] = t

    def parents(tensors):
        seen, out = set(), []
        for t in tensors:
            p = producer.get(id(t))
            if p is not None and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def emit(kind, inputs, out, extra=None):
        rec = {"idx": len(nodes), "kind": kind, "ins": parents(inputs),
               "out_shape": tuple(out.shape)}
        if inputs and isinstance(inputs[0], torch.Tensor):
            rec["in_shape"] = tuple(inputs[0].shape)
        rec.update(extra or {})
        nodes.append(rec)
        claim(out, rec["idx"])
        return rec

    claim(x, 0)

    def pre_hook(mod, inp):
        if type(mod).__name__ in _DRAWERS:
            depth[0] += 1

    def post_hook(mod, inp, out):
        kind = type(mod).__name__
        drawn = kind in _DRAWERS
        if drawn:
            depth[0] -= 1
        if depth[0] != 0 or not isinstance(out, torch.Tensor):
            return
        if drawn:
            extract = _EXTRACT.get(kind)
            emit(kind, list(inp), out, extract(mod) if extract else None)
        elif id(out) not in producer:
            # not drawn (ReLU, a Sequential wrapper, ...): forward the producer along
            src = producer.get(id(inp[0])) if inp and isinstance(inp[0], torch.Tensor) else None
            if src is not None:
                claim(out, src)

    class _OpTracer(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            op = _OPS.get(getattr(func, "__name__", ""))
            if op and depth[0] == 0 and isinstance(out, torch.Tensor):
                srcs = list(args[0]) if op == "Concat" else [a for a in args if isinstance(a, torch.Tensor)]
                # two tensor operands and at least one traced source: skips `x + 1` noise
                # but still catches a self-concat, which dedupes down to a single edge
                if len(srcs) >= 2 and parents(srcs):
                    emit(op, srcs, out)
            return out

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))
    try:
        with torch.no_grad(), _OpTracer():
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    meta = {"name": type(model).__name__,
            "params": sum(p.numel() for p in model.parameters())}
    return nodes, meta


# --- svg primitives -----------------------------------------------------------


def _text(x, y, s, color=TEXT, size=13, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {FONT}>{html.escape(str(s))}</text>')


def _box(x, y, w, h, stroke, fill=PANEL, rx=10, dash=None, width=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')


def _pill(x, y, w, h, label, color, size=11):
    return (_box(x, y, w, h, color, fill="none", rx=h / 2, width=1.2)
            + _text(x + w / 2, y + h / 2 + size * 0.36, label, color, size))


def _line(x1, y1, x2, y2, color=DIM, width=1.4, dash=None, arrow=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = ' marker-end="url(#ah)"' if arrow else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}{a}/>')


def _channel_grid(cx, cy, n, color=BLUE, sq=9):
    shown = min(n, MAX_CH)
    cols = max(1, math.ceil(math.sqrt(shown)))
    rows = math.ceil(shown / cols)
    g = 2
    gw, gh = cols * (sq + g) - g, rows * (sq + g) - g
    x0, y0 = cx - gw / 2, cy - gh / 2
    s = ""
    for i in range(shown):
        r, c = divmod(i, cols)
        s += (f'<rect x="{x0 + c * (sq + g):.1f}" y="{y0 + r * (sq + g):.1f}" '
              f'width="{sq}" height="{sq}" rx="1.5" fill="{color}" opacity="0.85"/>')
    return s, gw, gh


def _col_ys(n, cy, h):
    if n <= 1:
        return [cy]
    step = h / (n - 1)
    return [cy - h / 2 + i * step for i in range(n)]


def _shape(rec) -> str:
    s = rec.get("out_shape")
    if not s or len(s) != 4:
        return " × ".join(str(v) for v in s[1:]) if s else ""
    return f"{s[1]}×{s[2]}×{s[3]}"


# --- drawers ------------------------------------------------------------------
# Each returns a spec dict: title, color, w, h (box only), inner (svg in box-local
# coords, origin = box top-left), sub (list of caption lines under the box).


def draw_input(rec):
    w = h = 92
    s = ""
    for i in range(1, 4):
        s += (_line(i * w / 4, 8, i * w / 4, h - 8, BORDER, 1, arrow=False)
              + _line(8, i * h / 4, w - 8, i * h / 4, BORDER, 1, arrow=False))
    return dict(title="Input", color=DIM, w=w, h=h, inner=s, sub=[_shape(rec)])


def draw_conv2d(rec):
    w, h = 168, 116
    gin, _, _ = _channel_grid(46, h / 2, rec["in_ch"])
    gout, _, _ = _channel_grid(w - 46, h / 2, rec["out_ch"])
    s = gin + gout + _line(w / 2 - 15, h / 2, w / 2 + 15, h / 2, BLUE)
    s += _text(46, h - 10, f'in {rec["in_ch"]}', DIM, 10)
    s += _text(w - 46, h - 10, f'out {rec["out_ch"]}', DIM, 10)
    st = f' /{rec["stride"]}' if rec.get("stride", 1) != 1 else ""
    return dict(title="Conv2d", color=BLUE, w=w, h=h, inner=s,
                sub=[f'{rec["k"]}×{rec["k"]} kernel{st}', _shape(rec)])


def draw_conv(rec):
    """Custom Conv block: Conv2d -> BatchNorm2d -> activation."""
    w, h = 176, 56
    pw, y = 46, h / 2 - 13
    act = rec.get("act", "SiLU") or "SiLU"
    dw = rec.get("groups", 1) > 1
    s = _pill(10, y, pw, 26, f'{rec["k"]}×{rec["k"]}', BLUE)
    s += _line(58, h / 2, 70, h / 2, DIM, 1.2)
    s += _pill(72, y, 38, 26, "BN", TEAL)
    s += _line(112, h / 2, 124, h / 2, DIM, 1.2)
    s += _pill(126, y, w - 136, 26, act, GREEN)
    st = f'  stride {rec["stride"]}' if rec.get("stride", 1) != 1 else ""
    sub = [f'{rec["in_ch"]} → {rec["out_ch"]}{st}', _shape(rec)]
    if dw:
        sub[0] += f'  g={rec["groups"]}'
    return dict(title="Conv", color=BLUE, w=w, h=h, inner=s, sub=sub)


def draw_dwconv(rec):
    w, h = 186, 56
    y = h / 2 - 13
    s = _pill(10, y, 78, 26, f'DW {rec["k"]}×{rec["k"]}', PURPLE)
    s += _line(90, h / 2, 102, h / 2, DIM, 1.2)
    s += _pill(104, y, 72, 26, "PW 1×1", BLUE)
    return dict(title="DWConv", color=PURPLE, w=w, h=h, inner=s,
                sub=[f'{rec["in_ch"]} → {rec["out_ch"]}  depthwise sep.', _shape(rec)])


def draw_c3k(rec):
    n, ks = rec["n_layers"], rec.get("ks", [])
    pw, gap = 44, 12
    w = 20 + n * pw + (n - 1) * gap
    h = 74
    cy = 46
    s = ""
    for i in range(n):
        x = 10 + i * (pw + gap)
        k = ks[i] if i < len(ks) else 3
        s += _pill(x, cy - 13, pw, 26, f"{k}×{k}", BLUE)
        if i:
            s += _line(x - gap, cy, x - 2, cy, DIM, 1.2)
    if rec.get("add"):
        s += (f'<path d="M8,{cy} C8,14 {w - 8},14 {w - 8},{cy}" fill="none" '
              f'stroke="{GREEN}" stroke-width="1.2" stroke-dasharray="3,3"/>')
        s += _text(w / 2, 18, "⊕ residual", GREEN, 10)
    return dict(title="C3k", color=BLUE, w=w, h=h, inner=s,
                sub=[f"{n} conv chain", _shape(rec)])


def _csp(rec, title, color, inner_label):
    """Shared CSP glyph: input splits into a block path + a bypass path, then merges."""
    w, h = 196, 92
    top, bot = 30, 70
    s = _line(8, h / 2, 22, top, color, 1.2, arrow=False)
    s += _line(8, h / 2, 22, bot, color, 1.2, arrow=False)
    s += f'<circle cx="8" cy="{h / 2}" r="3.5" fill="{color}"/>'
    s += _pill(24, top - 14, 118, 28, inner_label, color)
    bypass = "1×1 conv" if rec.get("bypass") == "conv" else "identity"
    s += _pill(24, bot - 13, 118, 26, bypass, DIM, size=10)
    mx = 156
    s += _line(144, top, mx - 4, h / 2 - 6, color, 1.2, arrow=False)
    s += _line(144, bot, mx - 4, h / 2 + 6, DIM, 1.2, arrow=False)
    s += _box(mx, h / 2 - 15, 30, 30, color, rx=6)
    s += _text(mx + 15, h / 2 + 5, "⧺", color, 15, weight="bold")
    return dict(title=title, color=color, w=w, h=h, inner=s,
                sub=[f'hidden {rec["hidden"]}  →  concat + 1×1', _shape(rec)])


def draw_c3k2(rec):
    n = rec["n_blocks"]
    return _csp(rec, "C3k2", ORANGE, f'{rec["inner"]} × {n}')


def draw_c2psa(rec):
    n = rec["n_blocks"]
    return _csp(rec, "C2PSA", PINK, f'{rec["inner"]} × {n}')


def draw_psa(rec):
    w, h = 196, 100
    s = ""
    for i, (lbl, col) in enumerate((("Q", BLUE), ("K", BLUE), ("V", TEAL))):
        s += _pill(14, 12 + i * 28, 34, 24, lbl, col)
    s += _line(50, 52, 74, 52, DIM, 1.2)
    mode = "elem" if rec.get("dpu_aware") else "matmul"
    s += _box(76, 34, 62, 36, PINK, rx=8)
    s += _text(107, 51, "attn", PINK, 11)
    s += _text(107, 63, mode, DIM, 9)
    s += _line(140, 52, 158, 52, DIM, 1.2)
    s += f'<circle cx="168" cy="52" r="11" fill="none" stroke="{GREEN}" stroke-width="1.4"/>'
    s += _text(168, 57, "⊕", GREEN, 13)
    s += (f'<path d="M6,52 C6,88 168,92 168,63" fill="none" stroke="{DIM}" '
          f'stroke-width="1.2" stroke-dasharray="3,3"/>')
    s += _text(88, 92, "history (identity)", DIM, 9)
    return dict(title="PSA", color=PINK, w=w, h=h, inner=s,
                sub=[f'hidden {rec["hidden"]}  {rec.get("act", "")}', _shape(rec)])


def draw_sppf(rec):
    w, h = 236, 96
    k = rec.get("k", 5)
    s = _pill(8, 20, 40, 26, "1×1", BLUE)
    xs = [60, 104, 148]
    for x in xs:
        s += _box(x, 20, 34, 26, PURPLE, rx=6)
        s += _text(x + 17, 37, f"{k}×{k}", PURPLE, 10)
        s += _line(x - 12, 33, x - 3, 33, DIM, 1.2)
    bar_y = 68
    s += _box(52, bar_y, 130, 16, TEAL, rx=5)
    s += _text(117, bar_y + 12, "concat ×4", TEAL, 10)
    for x in [28] + [x + 17 for x in xs]:
        s += _line(x, 48, x, bar_y - 2, DIM, 1, dash="2,2")
    s += _line(182, 33, 192, 33, DIM, 1.2)
    s += _pill(194, 20, 34, 26, "1×1", BLUE)
    return dict(title="SPPF", color=PURPLE, w=w, h=h, inner=s,
                sub=[f'hidden {rec["hidden"]}  ({k}×{k} pools, stride 1)', _shape(rec)])


def draw_pool(rec):
    w, h = 82, 72
    a = rec.get("in_shape", (0, 0, 0))[2] if rec.get("in_shape") else 0
    b = rec["out_shape"][2]
    s = _text(w / 2, h / 2 + 4, "↓2", PURPLE, 24, weight="bold")
    return dict(title="MaxPool", color=PURPLE, w=w, h=h, inner=s,
                sub=[f"{a}→{b}" if a else _shape(rec)])


def draw_upsample(rec):
    w, h = 88, 72
    scale = rec.get("scale") or 2
    s = _text(w / 2, h / 2 + 4, f"↑{scale:g}", TEAL, 24, weight="bold")
    return dict(title="Upsample", color=TEAL, w=w, h=h, inner=s,
                sub=[rec.get("mode", ""), _shape(rec)])


def draw_concat(rec):
    w, h = 72, 66
    s = _line(10, 16, w / 2 - 4, h / 2 - 4, TEAL, 1.3, arrow=False)
    s += _line(10, h - 16, w / 2 - 4, h / 2 + 4, TEAL, 1.3, arrow=False)
    s += _text(w / 2 + 6, h / 2 + 6, "⧺", TEAL, 20, weight="bold")
    return dict(title="Concat", color=TEAL, w=w, h=h, inner=s, sub=[_shape(rec)])


def draw_add(rec):
    w, h = 66, 66
    s = f'<circle cx="{w / 2}" cy="{h / 2}" r="18" fill="none" stroke="{GREEN}" stroke-width="1.6"/>'
    s += _text(w / 2, h / 2 + 7, "⊕", GREEN, 20)
    return dict(title="Add", color=GREEN, w=w, h=h, inner=s, sub=[_shape(rec)])


def draw_flatten(rec):
    w, h = 150, 170
    grid, gw, _ = _channel_grid(36, h / 2, 9, ORANGE)
    s = grid
    col_x = w - 26
    for yy in _col_ys(7, h / 2, 128):
        s += _line(36 + gw / 2, h / 2, col_x, yy, ORANGE, 0.7, arrow=False)
    for yy in _col_ys(7, h / 2, 128):
        s += f'<circle cx="{col_x}" cy="{yy:.1f}" r="3.2" fill="{ORANGE}"/>'
    return dict(title="Flatten", color=ORANGE, w=w, h=h, inner=s,
                sub=[f'2D → 1D  ({rec["out_shape"][1]})'])


def draw_dense(rec):
    w, h = 158, 190
    in_f, out_f = rec["in_f"], rec["out_f"]
    lx, rx = 30, w - 30
    lys = _col_ys(min(in_f, MAX_NODES), h / 2, h - 30)
    rys = _col_ys(min(out_f, MAX_NODES), h / 2, h - 30)
    s = ""
    for ly in lys:
        for ry in rys:
            s += _line(lx, ly, rx, ry, GREEN, 0.5, arrow=False).replace("/>", ' opacity="0.22"/>')
    for ly in lys:
        s += f'<circle cx="{lx}" cy="{ly:.1f}" r="3.4" fill="{DIM}"/>'
    for ry in rys:
        s += f'<circle cx="{rx}" cy="{ry:.1f}" r="3.8" fill="{GREEN}"/>'
    return dict(title="Dense", color=GREEN, w=w, h=h, inner=s, sub=[f"{in_f} → {out_f}"])


def draw_dropout(rec):
    w, h = 104, 190
    p = rec["p"]
    cx = w / 2
    ys = _col_ys(MAX_NODES, h / 2, h - 30)
    n_off = round(p * MAX_NODES)
    off = set(range(0, MAX_NODES, max(1, round(MAX_NODES / n_off)))[:n_off]) if n_off else set()
    s = ""
    for i, yy in enumerate(ys):
        if i in off:
            s += (f'<circle cx="{cx}" cy="{yy:.1f}" r="4" fill="none" stroke="{DIM}" stroke-width="1.2" opacity="0.5"/>'
                  + _line(cx - 4, yy - 4, cx + 4, yy + 4, RED, 1.2, arrow=False)
                  + _line(cx - 4, yy + 4, cx + 4, yy - 4, RED, 1.2, arrow=False))
        else:
            s += f'<circle cx="{cx}" cy="{yy:.1f}" r="4" fill="{RED}"/>'
    return dict(title="Dropout", color=RED, w=w, h=h, inner=s, sub=[f"p={p:g}  ({p:.0%} off)"])


# Registry doubles as the block-boundary rule: a module type listed here is one node
# and is never descended into. Add a drawer to collapse a new block type.
_DRAWERS = {
    "Input": draw_input, "Conv2d": draw_conv2d, "MaxPool2d": draw_pool,
    "Flatten": draw_flatten, "Linear": draw_dense, "Dropout": draw_dropout,
    "Upsample": draw_upsample,
    "Conv": draw_conv, "DWConv": draw_dwconv, "C3k": draw_c3k, "PSA": draw_psa,
    "SPPF": draw_sppf, "C3k2": draw_c3k2, "C2PSA": draw_c2psa,
    "Concat": draw_concat, "Add": draw_add,
}


# --- layout -------------------------------------------------------------------


def _node_svg(spec, vertical: bool) -> dict:
    """Wrap a drawer spec into a full node, drawn in node-local coords (origin = top-left).

    `cy` is the vertical centre of the box (edge anchor in horizontal flow); `top`/`bot`
    are the box edges (edge anchors in vertical flow). Both skip the label bands.

    In vertical flow the edges run down the box centreline, straight through where a
    centred caption would sit — so the labels move aside: left-aligned on a wide box,
    pushed out past the right edge on a narrow one where left-aligned still overlaps.
    """
    w, bh = spec["w"], spec["h"]
    subs = [s for s in spec.get("sub", []) if s]
    h = TITLE_H + bh + (SUB_H + SUB_LINE * (len(subs) - 1) if subs else 6)
    outside = vertical and w < 120
    lx, anchor = ((w + 8 if outside else 4), "start") if vertical else (w / 2, "middle")
    s = _text(lx, TITLE_H - 8, spec["title"], spec["color"], 12, anchor, weight="bold")
    s += _box(0, TITLE_H, w, bh, spec["color"])
    s += f'<g transform="translate(0,{TITLE_H})">{spec["inner"]}</g>'
    for i, line in enumerate(subs):
        s += _text(lx, TITLE_H + bh + 16 + i * SUB_LINE, line, DIM, 11, anchor)
    # labels parked beside a narrow box still need floor space, or the next column
    # gets laid out on top of them
    text_w = max([len(spec["title"]) * 7.4] + [len(t) * 6.0 for t in subs])
    return {"svg": s, "w": w, "h": h, "cy": TITLE_H + bh / 2,
            "top": TITLE_H, "bot": TITLE_H + bh,
            "ext_w": w + 8 + text_w if outside else w}


def _levels(nodes) -> list[int]:
    """Longest-path depth per node. Nodes are already in execution order, so one
    forward sweep is enough — every parent index is smaller than its child's."""
    lv = [0] * len(nodes)
    for n in nodes:
        if n["ins"]:
            lv[n["idx"]] = 1 + max(lv[i] for i in n["ins"])
    return lv


def _layout(nodes, vertical: bool):
    """Place nodes in bands by depth, ordered inside a band by parent barycenter.

    Written on a flow/cross axis pair so one implementation covers both directions:
    horizontal flow is left→right (flow = x), vertical flow is top→bottom (flow = y).
    """
    specs = {}
    for n in nodes:
        drawer = _DRAWERS.get(n["kind"])
        if drawer:
            specs[n["idx"]] = _node_svg(drawer(n), vertical)

    lv = _levels(nodes)
    bands: dict[int, list[int]] = {}
    for n in nodes:
        if n["idx"] in specs:
            bands.setdefault(lv[n["idx"]], []).append(n["idx"])

    # extent along each axis, and the cross-axis anchor used for barycenter ordering
    ext_flow = (lambda s: s["h"]) if vertical else (lambda s: s["w"])
    ext_cross = (lambda s: s["ext_w"]) if vertical else (lambda s: s["h"])
    anchor = (lambda i: pos[i][1] + specs[i]["w"] / 2) if vertical else (lambda i: pos[i][1] + specs[i]["cy"])
    gap_flow, gap_cross = (GAP_Y + 10, GAP_X) if vertical else (GAP_X, GAP_Y)

    pos = {}   # idx -> (flow_pos, cross_pos), top-left along each axis
    flow = 0.0
    for level in sorted(bands):
        idxs = bands[level]
        placed = lambda i: [p for p in nodes[i]["ins"] if p in pos]  # noqa: E731
        idxs.sort(key=lambda i: (sum(anchor(p) for p in placed(i)) / len(placed(i))
                                 if placed(i) else 0.0, i))
        band = max(ext_flow(specs[i]) for i in idxs)
        total = sum(ext_cross(specs[i]) for i in idxs) + gap_cross * (len(idxs) - 1)
        cross = -total / 2
        for i in idxs:
            sp = specs[i]
            pos[i] = (flow + (band - ext_flow(sp)) / 2, cross)
            cross += ext_cross(sp) + gap_cross
        flow += band + gap_flow
    return specs, pos, lv, flow - gap_flow


def _xy(i, pos, vertical) -> tuple[float, float]:
    """Node top-left in page coords (flow/cross unpacked back to x/y)."""
    f, c = pos[i]
    return (c, f) if vertical else (f, c)


def _rounded(points, r=13) -> str:
    """Path through an orthogonal polyline with the corners filleted."""
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for i in range(1, len(points) - 1):
        (ax, ay), (bx, by), (cx, cy) = points[i - 1], points[i], points[i + 1]
        d1 = math.hypot(bx - ax, by - ay) or 1.0
        d2 = math.hypot(cx - bx, cy - by) or 1.0
        k1, k2 = min(r, d1 / 2) / d1, min(r, d2 / 2) / d2
        d += (f" L{bx + (ax - bx) * k1:.1f},{by + (ay - by) * k1:.1f}"
              f" Q{bx:.1f},{by:.1f} {bx + (cx - bx) * k2:.1f},{by + (cy - by) * k2:.1f}")
    return d + f" L{points[-1][0]:.1f},{points[-1][1]:.1f}"


def render_svg(nodes: list[dict], meta: dict | None = None, direction: str = "auto") -> str:
    meta = meta or {}
    specs, pos, lv, flow_len = _layout(nodes, vertical=False)
    if not pos:
        raise ValueError("nothing to draw — no traced module had a drawer")
    if direction == "auto":
        # a long chain letterboxed into a 16:9 viewport is unreadable; flip it upright
        direction = "v" if max(lv) > 12 else "h"
    vertical = direction == "v"
    if vertical:
        specs, pos, lv, flow_len = _layout(nodes, vertical=True)

    xs = [_xy(i, pos, vertical)[0] for i in pos]
    ys = [_xy(i, pos, vertical)[1] for i in pos]
    x0 = min(xs) - MARGIN
    y0 = min(ys) - HEAD_H - MARGIN

    def place(i):
        x, y = _xy(i, pos, vertical)
        return x - x0, y - y0

    graph_w = max(x + specs[i]["ext_w"] for i, x in zip(pos, xs)) - x0
    graph_h = max(y + specs[i]["h"] for i, y in zip(pos, ys)) - y0

    # A skip edge drawn straight would run down the spine of every node it passes. Give
    # each one an orthogonal lane off to the side of the graph, widest span outermost so
    # nested fusions (FPN inside PAN) never cross.
    skips = sorted(((lv[n["idx"]] - lv[p], p, n["idx"]) for n in nodes if n["idx"] in pos
                    for p in n["ins"] if p in pos and lv[n["idx"]] - lv[p] > 1), reverse=True)
    lane_of = {(p, c): k for k, (_, p, c) in enumerate(skips)}
    LANE0, LANE_STEP, STUB = 34, 26, 24
    lane_span = LANE0 + LANE_STEP * len(skips) if skips else 0
    total_w = graph_w + MARGIN + (lane_span if vertical else 0)
    total_h = graph_h + MARGIN + (0 if vertical else lane_span)

    edges = ""
    for n in nodes:
        i = n["idx"]
        if i not in pos:
            continue
        cx, cy = place(i)
        sp = specs[i]
        # child-side anchor: left edge (horizontal flow) or top edge (vertical flow)
        ax, ay = (cx + sp["w"] / 2, cy + sp["top"]) if vertical else (cx, cy + sp["cy"])
        for p in n["ins"]:
            if p not in pos:
                continue
            px, py = place(p)
            psp = specs[p]
            bx, by = (px + psp["w"] / 2, py + psp["bot"]) if vertical else (px + psp["w"], py + psp["cy"])
            if (p, i) in lane_of:
                lane = LANE0 + LANE_STEP * lane_of[(p, i)]
                if vertical:
                    lx, sy = graph_w + lane, ay - STUB
                    pts = [(px + psp["w"], py + psp["cy"]), (lx, py + psp["cy"]),
                           (lx, sy), (ax, sy), (ax, ay - 8)]
                else:
                    ly, sx = graph_h + lane, ax - STUB
                    pts = [(px + psp["w"] / 2, py + psp["bot"]), (px + psp["w"] / 2, ly),
                           (sx, ly), (sx, ay), (ax - 8, ay)]
                d, col, wdt, op = _rounded(pts), ORANGE, 1.8, 0.95
            else:
                if vertical:
                    mid = (by + ay) / 2
                    d = (f'M{bx:.1f},{by:.1f} C{bx:.1f},{mid:.1f} {ax:.1f},{mid:.1f} '
                         f'{ax:.1f},{ay - 8:.1f}')
                else:
                    mid = (bx + ax) / 2
                    d = (f'M{bx:.1f},{by:.1f} C{mid:.1f},{by:.1f} {mid:.1f},{ay:.1f} '
                         f'{ax - 8:.1f},{ay:.1f}')
                col, wdt, op = DIM, 1.5, 0.7
            edges += (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wdt}" '
                      f'marker-end="url({"#ahs" if col is ORANGE else "#ah"})" opacity="{op}"/>')

    body = ""
    for i in pos:
        x, y = place(i)
        body += f'<g transform="translate({x:.1f},{y:.1f})">{specs[i]["svg"]}</g>'

    params = meta.get("params", 0)
    pstr = f"{params / 1e6:.2f}M params" if params >= 1e6 else f"{params:,} params"
    title = _text(total_w / 2, 42, f'{meta.get("name", "Network")} — {len(pos)} blocks, {pstr}',
                  TEXT, 24, weight="bold")
    sub = _text(total_w / 2, 64, "scroll to zoom · drag to pan · double-click to reset", DIM, 12)

    # legend lives in the header band, not the footer — a vertical graph is thousands of
    # pixels tall and a footer legend would be off-screen the whole time
    used = {nodes[i]["kind"] for i in pos}
    legend_items = [(k, c) for k, c in (
        ("Conv", BLUE), ("DWConv", PURPLE), ("C3k", BLUE), ("C3k2", ORANGE),
        ("C2PSA", PINK), ("SPPF", PURPLE), ("PSA", PINK), ("Upsample", TEAL),
        ("Concat", TEAL), ("Add", GREEN), ("Conv2d", BLUE), ("MaxPool2d", PURPLE),
        ("Flatten", ORANGE), ("Linear", GREEN), ("Dropout", RED)) if k in used]
    if skips:
        legend_items.append(("skip / fusion edge", ORANGE))
    lg, lx = "", total_w / 2 - sum(len(k) * 7.2 + 32 for k, _ in legend_items) / 2
    for name, col in legend_items:
        lg += (f'<rect x="{lx:.1f}" y="{88}" width="13" height="13" rx="3" fill="{col}"/>'
               + _text(lx + 20, 99, name, DIM, 12, anchor="start"))
        lx += len(name) * 7.2 + 32

    defs = (f'<defs>'
            f'<marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="{DIM}"/></marker>'
            f'<marker id="ahs" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="{ORANGE}"/></marker></defs>')
    return (f'<svg id="net" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
            f'width="100%" height="100%" preserveAspectRatio="xMidYMid meet">{defs}'
            f'<g id="cam">{title}{sub}{edges}{body}{lg}</g></svg>')


_PANZOOM = """
const svg=document.getElementById('net'),cam=document.getElementById('cam');
let s=1,tx=0,ty=0,drag=null;
const apply=()=>cam.setAttribute('transform',`translate(${tx},${ty}) scale(${s})`);
const at=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());};
// preserveAspectRatio letterboxes the whole graph, which for a graph 10x taller than
// the window means "readable at 4% zoom". When the aspect mismatch is that bad, open
// filling the short axis instead and let the user pan along the long one. Screen coords
// are  screen = inset + fit*(t + s*user),  so solve for the t that puts the content edge
// where we want it.
function home(){
  const vb=svg.viewBox.baseVal,r=svg.getBoundingClientRect();
  if(!r.width||!r.height){s=1;tx=ty=0;return apply();}
  const fw=r.width/vb.width,fh=r.height/vb.height,fit=Math.min(fw,fh);
  s=Math.max(fw,fh)/fit>2?Math.max(1,Math.min(Math.max(fw,fh),1.6)/fit):1;
  const cw=vb.width*fit*s,ch=vb.height*fit*s;
  tx=((cw<r.width?(r.width-cw)/2:0)-(r.width-vb.width*fit)/2)/fit;
  ty=((ch<r.height?(r.height-ch)/2:0)-(r.height-vb.height*fit)/2)/fit;
  apply();
}
svg.addEventListener('wheel',e=>{e.preventDefault();const p=at(e),k=Math.exp(-e.deltaY*0.0015),
  n=Math.min(40,Math.max(0.2,s*k));tx=p.x-(p.x-tx)*(n/s);ty=p.y-(p.y-ty)*(n/s);s=n;apply();},{passive:false});
svg.addEventListener('pointerdown',e=>{drag=at(e);svg.setPointerCapture(e.pointerId);svg.style.cursor='grabbing';});
svg.addEventListener('pointermove',e=>{if(!drag)return;const p=at(e);tx+=p.x-drag.x;ty+=p.y-drag.y;apply();});
svg.addEventListener('pointerup',()=>{drag=null;svg.style.cursor='grab';});
svg.addEventListener('dblclick',home);
window.addEventListener('resize',home);
svg.style.cursor='grab';
home();
"""


def render(model: nn.Module, out_path: str = "network.html", input_shape=(1, 1, 28, 28),
           direction: str = "auto") -> str:
    """Trace `model` and write a self-contained HTML diagram. `direction` is "h", "v",
    or "auto" (horizontal for short chains, vertical once the graph gets deep)."""
    nodes, meta = trace(model, input_shape)
    svg = render_svg(nodes, meta, direction)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>{html.escape(meta["name"])} — Architecture</title>'
           f'<style>html,body{{margin:0;height:100%;overflow:hidden;background:{BG}}}'
           f'.wrap{{height:100vh;display:flex;align-items:center;justify-content:center}}'
           f'svg{{touch-action:none;user-select:none}}</style>'
           f'</head><body><div class="wrap">{svg}</div>'
           f'<script>{_PANZOOM}</script></body></html>')
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(Path(out_path).resolve())


if __name__ == "__main__":
    from xml.etree import ElementTree

    def _check_svg(svg: str):
        assert svg.startswith("<svg")
        ElementTree.fromstring(svg)          # every drawer must emit well-formed XML
        return svg

    mnist = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10),
    )
    nodes, meta = trace(mnist, (1, 1, 28, 28))
    kinds = [n["kind"] for n in nodes]
    assert kinds == ["Input", "Conv2d", "MaxPool2d", "Conv2d", "MaxPool2d",
                     "Flatten", "Linear", "Dropout", "Linear"], kinds
    # a ReLU between two drawn layers must not break the edge chain
    assert all(n["ins"] == [n["idx"] - 1] for n in nodes[1:]), [n["ins"] for n in nodes]
    assert mnist.training is True, "trace must restore the original train/eval flag"
    _check_svg(render_svg(nodes, meta))
    print("ok (mnist chain)")

    from modules.conv import Conv
    from modules.csp import C3k2

    class Branchy(nn.Module):
        """Two paths that meet at a cat — the FPN/PAN shape in miniature."""

        def __init__(self):
            super().__init__()
            self.a = Conv(3, 8, stride=2)
            self.b = C3k2(8, 8)
            self.c = Conv(16, 8, kernel_size=1)

        def forward(self, x):
            a = self.a(x)
            return self.c(torch.cat([self.b(a), a], dim=1))

    nodes, meta = trace(Branchy(), (1, 3, 32, 32))
    kinds = [n["kind"] for n in nodes]
    assert kinds == ["Input", "Conv", "C3k2", "Concat", "Conv"], kinds
    assert nodes[3]["ins"] == [2, 1], nodes[3]["ins"]   # cat sees both branches
    lv = _levels(nodes)
    assert lv == [0, 1, 2, 3, 4], lv
    for d in ("h", "v", "auto"):
        _check_svg(render_svg(nodes, meta, direction=d))
    print("ok (branch + cat)")

    # every drawer renders, including the ones no test model above reaches
    from modules.bottlenecks.c3k import C3k
    from modules.bottlenecks.psa import PSA
    from modules.conv import DWConv
    from modules.csp import C2PSA
    from modules.sppf import SPPF

    class Zoo(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.dw = DWConv(8, 8)
            self.sppf = SPPF(8, 8)
            self.psa = PSA(8, 8)
            self.c3k = C3k(8, 8, num_layers=3, kernel_size=[1, 3, 1])
            self.c2psa = C2PSA(8, 8)
            self.c3k2 = C3k2(16, 8)

        def forward(self, x):
            x = self.c3k(self.dw(self.up(x)))
            x = self.c2psa(self.psa(self.sppf(x)) + x)      # bare `+` -> an Add node
            return self.c3k2(torch.cat([x, x], dim=1))      # torch.cat -> a Concat node

    nodes, meta = trace(Zoo(), (1, 8, 8, 8))
    kinds = [n["kind"] for n in nodes]
    assert kinds == ["Input", "Upsample", "DWConv", "C3k", "SPPF", "PSA", "Add",
                     "C2PSA", "Concat", "C3k2"], kinds
    assert nodes[6]["ins"] == [5, 3], nodes[6]["ins"]   # Add sees both branches
    assert nodes[8]["ins"] == [7], nodes[8]["ins"]      # self-concat dedupes to one edge
    _check_svg(render_svg(nodes, meta))
    untested = set(_DRAWERS) - set(kinds) - {"Conv2d", "MaxPool2d", "Flatten", "Linear",
                                             "Dropout", "Conv"}
    assert not untested, f"drawer never rendered: {untested}"
    print("ok (all drawers)")
