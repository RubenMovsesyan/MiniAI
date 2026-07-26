# PyTorch ML Framework — Guide for Agents

This document describes the small, reusable PyTorch toolkit that lives in `pytorch/`.
It exists so you can **define a model, train it, save it, and visualize it** with a
few lines, without rebuilding the plumbing each time. Read this before touching
anything under `pytorch/`.

> Scope: this is the **PyTorch track**, separate from the archived C++/CUDA project in
> `cuda_nn_from_scratch/` (own `CLAUDE.md` + `./build`). Nothing here touches or depends
> on the C++ build.

---

## TL;DR

```python
# from inside pytorch/, run as a module: python -m projects.mnist.mnist
import torch.nn as nn
from data.mnist import mnist_loaders
from training.trainer import Trainer

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
pytorch/
  modules/                   layer types, importable by any project
    conv.py                  ConvBNSiLU (Conv2d -> BatchNorm2d -> SiLU)
    c3k2.py                  C3k, C3k2 (CSP block)
    sppf.py                  SPPF (spatial pyramid pooling - fast)
  projects/                  one subdir per model
    YOLO/                    yolo.py (the model + entry point), voc.py (VOC2012 loader)
    mnist/                   mnist.py (end-to-end demo, the TL;DR above)
  training/
    trainer.py               Trainer: fit / evaluate / export / visualize / show_predictions
    export.py                Exporter interface + OnnxExporter (swap point for other formats)
  data/
    mnist.py                 mnist_loaders() -> (train, test) DataLoaders
  utils/
    netviz.py                render(model) -> self-contained network.html (SVG diagram)
    viewer.py                show(model, loader, device) -> matplotlib prediction viewer
  docs/ml_framework.md       this file
  setup.sh                   creates .venv, installs torch (cu128) + deps
  requirements.txt           torch, numpy, onnx, onnxruntime, matplotlib
  .venv/                     the virtualenv, ~7 GB (gitignored)
```

Adding a directory of a new kind (e.g. `losses/`, `metrics/`) is fine — it becomes an
importable package automatically, no `__init__.py` needed (PEP 420 namespace packages).

**Run everything from `pytorch/` as a module**, never by file path:

```bash
cd pytorch
.venv/bin/python -m projects.YOLO.yolo    # project entry point
.venv/bin/python -m modules.c3k2          # a module's __main__ self-check
```

`python projects/YOLO/yolo.py` fails — that puts `projects/YOLO/` on `sys.path` instead of
`pytorch/`, so `from modules... import` / `from training... import` don't resolve. (An
earlier `_lib.py` sys.path shim did this job; `-m` replaced it and the shim is gone.)

Generated artifacts (`*.onnx`, `network.html`, `__pycache__/`, `.venv/`) are gitignored.

---

## Setup

System Python is **3.14**, externally-managed (PEP 668) — a venv is mandatory.

```bash
cd pytorch
./setup.sh                     # venv + torch from the cu128 index + deps
source .venv/bin/activate
python -m projects.mnist.mnist
```

- GPU: RTX 4080 SUPER, so `setup.sh` pulls the CUDA (`cu128`) torch wheel. Verified
  working torch is **2.11.0+cu128** on Python 3.14.
- Data: `mnist_loaders()` reads MNIST IDX files from `$MNIST_DIR`, defaulting to
  `~/Downloads/ml_training`. Images come out NCHW `(N,1,28,28)` in `[0,1]`, labels
  int64 class indices (for `nn.CrossEntropyLoss`).

---

## `Trainer` (training/trainer.py) — the core

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

## Saving / export (training/export.py)

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
  `onnxruntime` (or the archived C++ engine in `cuda_nn_from_scratch/`), no Python/model-class needed, safe to load.
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

**Architecture diagram — `utils/netviz.py`, `netviz.render(model, out="network.html")`**
- Traces a dummy forward with hooks (works on **any** `nn.Module`, no model edits) to
  get true layer order + shapes, then emits a self-contained HTML/SVG horizontal
  pipeline: input tile → Conv (in/out channel grids) → MaxPool (`↓2`, spatial) →
  Flatten (2D→1D) → Dense (webbing, `in→out`) → Dropout (`p`% crossed out) → Output.
- `viewBox` + `preserveAspectRatio` letterbox the whole thing into the viewport — scales
  to width, never scrolls. Open the HTML directly.

**Prediction viewer — `utils/viewer.py`, `viewer.show(model, loader, device)`**
- Two matplotlib panes (correct vs incorrect), each showing the image, the 10 logits
  (bars, predicted/true colored), and one grid of **conv feature maps** per conv layer
  (post-ReLU activations). "Next" button per pane. Needs a display (TkAgg backend).

---

## `projects/` — one subdir per model

A project owns its model definition, its dataset loader, and its entry point. Everything
reusable belongs in `modules/`, `training/`, `data/`, or `utils/` instead.

- **`projects/mnist/mnist.py`** — the end-to-end demo above.
- **`projects/YOLO/`** — the active project.
  - `yolo.py` — the model, written **inline in one `nn.Sequential`** plus a `WIDTH`
    channel multiple. Entry point: `python -m projects.YOLO.yolo`.
  - `voc.py` — `voc_image_loader()` over PASCAL VOC2012 `JPEGImages`. Items are
    `(stem, CHW float tensor in [0,1])`; `pad_collate` zero-pads each batch to its max
    H,W (right/bottom, top-left origin kept so box coords stay valid). `$VOC_DIR`
    overrides the default dataset path. Annotation (XML) parsing is not built yet.

---

## Gotchas / conventions

- **Always seed** (`Trainer(seed=...)`). Unseeded runs are nondeterministic; some seeds/
  LRs diverge to chance (loss pinned at `ln(10)=2.303`, ~11% acc on MNIST). Prefer `adam`
  over high-LR `sgd+momentum` for stability.
- **Best is kept in memory, exported once** at end of `fit`. The on-disk artifact is the
  best epoch, not necessarily the final one.
- **`Trainer` is classification-only** — `nn.CrossEntropyLoss` plus `argmax`-based accuracy
  over `(x, y)` batches. Object detection needs a box/objectness/class loss and mAP, and
  `pad_collate` yields `(stems, images)`, so YOLO training requires a `Trainer` subclass or
  a pluggable-loss change to `training/trainer.py` first.
- **Run from `pytorch/` with `-m`** (see Directory layout). Running a file by path breaks
  the package imports. Generated files (`*.onnx`, `network.html`) are gitignored.
- **torchvision is not installed** — use PIL for image IO.
- **Extending:** new datasets → a `*_loaders()` in `data/` returning torch DataLoaders
  (TensorDataset so `utils/viewer.py` can pull `.dataset.tensors`). New layer types →
  `modules/`. New save formats → subclass `Exporter`. New layer types render in `netviz`
  automatically if they're standard `nn` modules; add a drawer in `utils/netviz.py` for
  custom visuals.
