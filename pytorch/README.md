# pytorch

Building blocks to create, train, save, and visualize models. Define any
`nn.Module`, point at a dataset, and get training + best-iteration ONNX export +
architecture diagram + prediction viewer.

## Setup

```bash
./setup.sh
source .venv/bin/activate
python example_mnist.py
```

`MNIST_DIR` (env) points at the IDX files; defaults to `~/Downloads/ml_training`.

## Use

```python
import torch.nn as nn
from data import mnist_loaders
from trainer import Trainer

train, test = mnist_loaders(batch=100)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
tr.fit(train, test, epochs=10, save_best="best.onnx")   # keeps + exports the best epoch
tr.visualize()             # -> network.html
tr.show_predictions(test)  # correct/incorrect viewer
```

## Modules

| File | What |
|---|---|
| `data.py` | `mnist_loaders()` — IDX → torch DataLoaders |
| `trainer.py` | `Trainer` — fit / evaluate / export / visualize / show_predictions |
| `export.py` | `Exporter` interface + `OnnxExporter` (swap point for other formats) |
| `netviz.py` | `render(model)` — architecture diagram HTML/SVG |
| `viewer.py` | `show(model, loader, device)` — prediction + feature-map viewer |
| `example_mnist.py` | end-to-end demo |

Saving is ONNX only for now, behind `Exporter` so other formats can drop in.
`best.onnx` / `network.html` are generated (gitignored).
