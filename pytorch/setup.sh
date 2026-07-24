#!/usr/bin/env bash
# venv + torch (cu128, RTX 40-series) + deps.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
