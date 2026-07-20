#!/usr/bin/env bash
# Create the venv and install deps. System Python is 3.14 (PEP 668
# externally-managed) so a venv is mandatory. torch comes from the CUDA index
# (RTX 4080 SUPER, driver 610 -> cu128 wheels); everything else from PyPI.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
# torch first, from the CUDA wheel index.
pip install --index-url https://download.pytorch.org/whl/cu128 torch
# rest from PyPI.
pip install -r requirements.txt

echo "--- CUDA check ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Done. Activate with: source python/.venv/bin/activate"
