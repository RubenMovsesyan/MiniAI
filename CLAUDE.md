# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

MiniAI is a repo for experimenting with AI, neural networks, and the math behind them.
It holds two independent tracks — they share no code and no build system:

```
pytorch/                 ACTIVE. PyTorch toolkit + projects (YOLO, MNIST).
cuda_nn_from_scratch/    ARCHIVED. The original hand-written CUDA neural network.
```

**Default to `pytorch/`.** Only touch `cuda_nn_from_scratch/` when explicitly asked.

## `pytorch/` — active work

Reusable building blocks so a project only has to define an `nn.Module`. Layout:

```
pytorch/
  modules/     layer types (Conv, SPPF), CSP wrappers (C3/C2), wrapped blocks (C3k2/C2PSA)
  projects/    one subdir per model: YOLO/, mnist/
  training/    trainer.py (Trainer), export.py (Exporter/OnnxExporter)
  data/        dataset loaders (mnist.py)
  utils/       netviz.py (architecture diagram), viewer.py (prediction viewer)
  docs/        ml_framework.md — the full guide, read it before changing anything here
  .venv/       the virtualenv (gitignored, ~7 GB)
```

**Run convention:** always from `pytorch/`, always as a module, always with the venv:

```bash
cd pytorch
.venv/bin/python -m projects.YOLO.yolo     # or: source .venv/bin/activate
```

Running a file by path (`python projects/YOLO/yolo.py`) breaks the package imports.
Every module has a `__main__` self-check that runs the same way
(`.venv/bin/python -m modules.c3k2`).

Setup: `./setup.sh` (venv + torch cu128 + deps). torch 2.11.0+cu128 on Python 3.14;
PIL available, **torchvision is not installed**.

## `cuda_nn_from_scratch/` — archived

Hand-written CUDA GPU primitives, forward and back propagation from scratch (~97% MNIST).
Has its own `CLAUDE.md` and custom C build system; run `./build` from inside that directory.
Feature-complete, no longer developed.
