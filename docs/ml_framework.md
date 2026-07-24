# PyTorch ML Framework — Guide for Agents

This document describes the small, reusable PyTorch toolkit that lives in `pytorch/`.
It exists so you can **define a model, train it, save it, and visualize it** with a
few lines, without rebuilding the plumbing each time. Read this before touching
anything under `pytorch/`.

> Scope: this is the **PyTorch track**, separate from the C++/CUDA project in `src/`
> (which has its own `CLAUDE.md` docs and `./build` system). Nothing here touches or
> depends on the C++ build.

---

## TL;DR

```python
# from inside pytorch/  (flat modules, run scripts from this dir)
import torch.nn as nn
from data import mnist_loaders
from trainer import Trainer

train, test = mnist_loaders(batch=100)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
tr.fit(train, test, epochs=10, save_best="best.onnx")  # trains, keeps + exports best epoch
tr.visualize()             # -> network.html (architecture diagram)
tr.show_predictions(test)  # correct/incorrect + conv feature-map viewer
```

Define any `nn.Module`, point at a dataset, get training + best-iteration ONNX export
+ two visualizers for free.

---

## Directory layout

```
pytorch/                     the reusable toolkit (use this)
  data.py                    mnist_loaders() -> (train, test) DataLoaders
  trainer.py                 Trainer: fit / evaluate / export / visualize / show_predictions
  export.py                  Exporter interface + OnnxExporter (swap point for other formats)
  netviz.py                  render(model) -> self-contained network.html (SVG diagram)
  viewer.py                  show(model, loader, device) -> matplotlib prediction viewer
  example_mnist.py           end-to-end demo (the TL;DR above)
  setup.sh                   creates .venv, installs torch (cu128) + deps
  requirements.txt           torch, numpy, onnx, onnxruntime, matplotlib
  data_viz/                  standalone dataset browsers (not ML; see below)

pytorch_experimentation/     ARCHIVED first-pass MNIST rebuild + HTML math/ONNX docs.
                             Superseded by pytorch/. Read-only reference; don't build on it.
```

Modules are **flat** (no package `__init__`); run scripts from inside `pytorch/` so
`from data import ...` / `import netviz` resolve. Generated artifacts
(`*.onnx`, `network.html`) are gitignored.

---

## Setup

System Python is **3.14**, externally-managed (PEP 668) — a venv is mandatory.

```bash
cd pytorch
./setup.sh                     # venv + torch from the cu128 index + deps
source .venv/bin/activate
python example_mnist.py
```

- GPU: RTX 4080 SUPER, so `setup.sh` pulls the CUDA (`cu128`) torch wheel. Verified
  working torch is **2.11.0+cu128** on Python 3.14.
- Data: `mnist_loaders()` reads MNIST IDX files from `$MNIST_DIR`, defaulting to
  `~/Downloads/ml_training`. Images come out NCHW `(N,1,28,28)` in `[0,1]`, labels
  int64 class indices (for `nn.CrossEntropyLoss`).

---

## `Trainer` (trainer.py) — the core

```python
Trainer(model, optimizer="adam", lr=1e-3, device="cuda",
        exporter=OnnxExporter(), seed=None)
```

- `optimizer`: `"adam"` → `Adam(lr)`, `"sgd"` → `SGD(lr, momentum=0.9)`, or pass an
  optimizer instance.
- `seed`: sets `torch.manual_seed` — **use it**; without a seed, runs are
  nondeterministic and can randomly diverge (see Gotchas).
- `device`: falls back to CPU if CUDA is unavailable.

Methods:

| call | does |
|---|---|
| `fit(train, test, epochs, save_best=None, verbose=True) -> best_acc` | Trains; each epoch prints loss + test acc; **keeps the best-accuracy weights in memory** (`deepcopy`), restores them into the model at the end, and if `save_best` is set, exports that best model via the exporter. |
| `evaluate(loader) -> float` | Test accuracy in `[0,1]`. |
| `export(path, example=None) -> path` | Exports the current model (uses a cached example batch from `fit`, or pass `example=`). |
| `visualize(out="network.html") -> path` | Writes the architecture diagram (see netviz). |
| `show_predictions(loader)` | Opens the matplotlib prediction viewer (blocking window). |

**Why best-in-memory matters:** a model can hit high accuracy then diverge to chance
later in training. The Trainer preserves the best epoch's weights so a late collapse
doesn't cost you the good model.

---

## Saving / export (export.py)

Saving goes through an **`Exporter` interface** so formats are swappable. Only
**ONNX** is implemented now:

```python
class Exporter:
    ext = ""
    def save(self, model, path, example): ...   # override

class OnnxExporter(Exporter):
    ext = ".onnx"
    # torch.onnx.export(..., dynamo=False)
```

- ONNX is the deploy/portable artifact: self-contained (graph + weights), runs in
  `onnxruntime` (or the C++ engine in `src/`), no Python/model-class needed, safe to load.
- It is **inference-only** — no optimizer state, no training resume. For "save the best
  model to use later," ONNX is the right call; for resuming training you'd add a
  `state_dict` exporter (the interface leaves room — not built yet).
- **Gotcha:** export is pinned to `dynamo=False` (the legacy TorchScript exporter). The
  newer dynamo path needs the `onnxscript` package, which isn't installed; `dynamo=False`
  avoids that dependency.

To add a format later: subclass `Exporter`, implement `save`, pass it as
`Trainer(..., exporter=YourExporter())`.

---

## Visualizers

**Architecture diagram — `netviz.render(model, out="network.html")`**
- Traces a dummy forward with hooks (works on **any** `nn.Module`, no model edits) to
  get true layer order + shapes, then emits a self-contained HTML/SVG horizontal
  pipeline: input tile → Conv (in/out channel grids) → MaxPool (`↓2`, spatial) →
  Flatten (2D→1D) → Dense (webbing, `in→out`) → Dropout (`p`% crossed out) → Output.
- `viewBox` + `preserveAspectRatio` letterbox the whole thing into the viewport — scales
  to width, never scrolls. Open the HTML directly.

**Prediction viewer — `viewer.show(model, loader, device)`**
- Two matplotlib panes (correct vs incorrect), each showing the image, the 10 logits
  (bars, predicted/true colored), and one grid of **conv feature maps** per conv layer
  (post-ReLU activations). "Next" button per pane. Needs a display (TkAgg backend).

---

## `data_viz/` — dataset browsers (not ML)

Standalone, frontend-only tools for eyeballing datasets. First one: a **PASCAL VOC2012
browser**.

```bash
cd pytorch/data_viz
python3 build.py        # parses VOC XML -> data.js manifest (stdlib only, ~0.4s)
xdg-open index.html     # grid + class filter + paginate; click -> detail w/ boxes
```

- `build.py` reads object boxes (parts ignored) from the VOC `Annotations/` and writes
  `data.js` (metadata only — no image files created; the dataset dir is never modified).
  The VOC path is a constant at the top of `build.py`.
- `index.html` is a self-contained static page: paginated lazy grid, filter by class,
  click a thumbnail → detail modal with per-class colored bounding boxes (label at each
  box's top-right corner). No server; images loaded via `file://`. If a browser blocks
  cross-dir `file://` images, fall back to `python3 -m http.server` in `data_viz/`.
- Deep-link: `index.html#i<n>` opens image `n`'s detail directly.

Put future dataset viewers here as siblings.

---

## Gotchas / conventions

- **Always seed** (`Trainer(seed=...)`). Unseeded runs are nondeterministic; some seeds/
  LRs diverge to chance (loss pinned at `ln(10)=2.303`, ~11% acc on MNIST). Prefer `adam`
  over high-LR `sgd+momentum` for stability.
- **Best is kept in memory, exported once** at end of `fit`. The on-disk artifact is the
  best epoch, not necessarily the final one.
- **`.env` precedence:** the archived `pytorch_experimentation/` reads config from a
  `.env`; a value there overrides code defaults (a shell `VAR=... python ...` overrides
  both). The new `pytorch/` uses explicit args + `$MNIST_DIR` only — no `.env`.
- **Run from inside the dir** (flat modules). Generated files (`*.onnx`, `network.html`,
  `data_viz/data.js`) are gitignored.
- **Extending:** new datasets → a `*_loaders()` in `data.py` returning torch DataLoaders
  (TensorDataset so `viewer.py` can pull `.dataset.tensors`). New save formats →
  subclass `Exporter`. New layer types render in `netviz` automatically if they're
  standard `nn` modules; add a drawer in `netviz.py` for custom visuals.
```
