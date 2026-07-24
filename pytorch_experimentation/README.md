# PyTorch track

The C++/CUDA build (`../src`) was a from-scratch exercise to understand the math
(documented in `~/Documents/ml_training`). This directory is the PyTorch
re-implementation — the goal is to learn PyTorch by rebuilding the same
784→128→10 MNIST classifier and reaching the same ~97.6% test accuracy.

The model (`model.py`) and training loop (`train.py`) are implemented; `main.py`
trains, prints stats, and opens the prediction viewer. ONNX export is still a TODO
— see the docs.

## Quickstart

```bash
./setup.sh                       # create .venv, install torch (cu128) + deps
source .venv/bin/activate
python data.py                   # self-check the IDX loader
python main.py                   # train -> stats -> visualize
```

`main.py` reads config from `.env` (copy from `.env.example`). `MNIST_DIR`
points at the IDX files (default `~/Downloads/ml_training`).

## Docs

Open `docs/index.html` in a browser:
- **`docs/pytorch_guide.html`** — the C++ → PyTorch API map.
- **`docs/onnx_and_engine.html`** — turning the trained model into an ONNX file
  and running it (onnxruntime now; your own CUDA engine later).

## Files

| File | What |
|---|---|
| `main.py` | Thin entry: train → stats → visualize. |
| `model.py` | `Config` + `Net`. |
| `train.py` | Training loop + `evaluate` (the meat). |
| `data.py` | IDX → NumPy/torch loaders. |
| `visualizer.py` | Correct/incorrect prediction viewer. |
| `setup.sh` / `requirements.txt` | Env setup. |
| `.env` / `.env.example` | Config. |
