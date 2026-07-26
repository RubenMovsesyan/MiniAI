# pytorch

Building blocks to create, train, save, and visualize models. Define any
`nn.Module`, point at a dataset, and get training + best-iteration ONNX export +
architecture diagram + prediction viewer.

## Setup

```bash
./setup.sh
source .venv/bin/activate
python -m projects.mnist.mnist
```

`MNIST_DIR` (env) points at the IDX files; defaults to `~/Downloads/ml_training`.
`VOC_DIR` points at PASCAL VOC2012 for the YOLO project.

## Running

Everything runs **from this directory, as a module** — package imports break if you
run a file by path.

```bash
python -m projects.YOLO.yolo      # a project's entry point
python -m modules.c3k2            # a module's self-check
```

## Layout

| Dir | What |
|---|---|
| `modules/` | layer types — `conv.py` (`ConvBNSiLU`), `c3k2.py` (`C3k`, `C3k2`), `sppf.py` (`SPPF`) |
| `projects/` | one subdir per model: `YOLO/` (`yolo.py` model, `voc.py` loader), `mnist/` |
| `training/` | `trainer.py` (`Trainer`: fit / evaluate / export / visualize / show_predictions), `export.py` (`Exporter` + `OnnxExporter`) |
| `data/` | dataset loaders — `mnist.py` (`mnist_loaders()`, IDX → DataLoaders) |
| `utils/` | `netviz.py` (`render(model)` → architecture HTML/SVG), `viewer.py` (`show()` → prediction + feature-map viewer) |
| `docs/` | `ml_framework.md` — full guide |

## Use

```python
import torch.nn as nn
from data.mnist import mnist_loaders
from training.trainer import Trainer

train, test = mnist_loaders(batch=100)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
tr.fit(train, test, epochs=10, save_best="best.onnx")   # keeps + exports the best epoch
tr.visualize()             # -> network.html
tr.show_predictions(test)  # correct/incorrect viewer
```

Saving is ONNX only for now, behind `Exporter` so other formats can drop in.
`best.onnx` / `network.html` are generated (gitignored).
