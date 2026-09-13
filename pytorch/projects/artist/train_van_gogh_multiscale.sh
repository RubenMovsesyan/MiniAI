#!/usr/bin/env bash
# Train the feed-forward Van Gogh style network with the settings tuned in the
# artist.py experiments: 2 style images (common style averaged), the shallow-
# layer reweight, multi-scale eval_size (256, 512) so the network learns bold
# AND genuinely-textured brushwork instead of a "filter" look, and color_weight
# (see below) so Van Gogh's blue/yellow palette doesn't get imposed regardless
# of a photo's own colour.
# Machine-specific (hardcoded COCO_DIR) -- gitignored, not shared.
#
# COST: dropped the 1024 scale (and crop_size along with it -- no reason to
# train on 1024px crops if nothing judges past 512) after measuring it at
# ~3 img/s, ~11h/epoch -- too slow to iterate on. At crop_size=512,
# eval_size=(256,512), batch_size=8 no longer OOMs (it did at crop=1024) and
# measures ~10.5 img/s, ~3.1h/epoch -- about 3.5x faster. See "ADJUSTABLE
# PARAMETERS" below for the knobs to change this further.
#
# PAUSE / SHUT DOWN AND CONTINUE LATER:
#   checkpoint-every (below) periodically saves weights + optimizer state + the
#   step counter to latest.pt -- not just weights -- specifically so a later
#   run can pick back up as if it had never stopped. To stop: Ctrl-C, or just
#   shut the machine down. To continue afterward:
#
#     ./train_van_gogh_multiscale.sh \
#       --continue-from checkpoints/van_gogh_multiscale/latest.pt
#
#   `--epochs` means "epochs THIS invocation will run", not "epochs left from
#   the original total" -- pass however many more epochs you want now.
set -euo pipefail

ARTIST_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTORCH_DIR="$(cd "$ARTIST_DIR/../.." && pwd)"

export COCO_DIR="$HOME/external1/ml_datasets/coco/images/train2017"
export PYTHONPATH="$PYTORCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- ADJUSTABLE PARAMETERS --------------------------------------------------
# Everything here can also be overridden per-run by passing flags when you
# call this script, e.g.: ./train_van_gogh_multiscale.sh --epochs 1
#
#   --style-weight        1e6      the main dial -- raise/lower if output looks
#                                  under/over-stylised (see the artist.py runs)
#   --content-weight      1.0
#   --tv-weight           0.0
#   --color-weight        20000    pulls output colour toward the source photo's own
#                                  (validated on Mona Lisa/starship in artist.py --
#                                  judged at each eval-size scale, same as content/style,
#                                  so it doesn't break strokes into a bubbly texture)
#   --style-layer-weights conv1_1=4.0 conv2_1=3.0 conv3_1=1.0 conv4_1=0.5 conv5_1=0.25
#                                  shallow-layer reweight from the tuning experiments
#   --eval-size           256 512   the multi-scale trick; every value must be <= --crop-size
#   --crop-size           512      training crop size; real detail ceiling for eval-size
#   --batch-size          8        fits at this crop/eval-size (16GB card); 1024 needed 2
#   --epochs              2        ~3.1h/epoch at these settings -- drop to 1 to halve it
#   --coco-dir             (env)   defaults to $COCO_DIR (train2017, 118,287 images);
#                                  point at a smaller folder (e.g. val2017, 5,000 images)
#                                  for a much faster full pass while iterating
#   --lr                  1e-3     Adam
#   --checkpoint-every    500      steps between checkpoint saves
#   --preview-every       200      steps between preview-image saves
# -----------------------------------------------------------------------------

exec "$PYTORCH_DIR/.venv/bin/python" -m projects.artist.train_style \
  "$ARTIST_DIR/images/van_gogh/starry_night.jpg" \
  "$ARTIST_DIR/images/van_gogh/wheatfield_with_crows.jpg" \
  --preview "$ARTIST_DIR/images/apple.jpg" \
  --style-size 512 \
  --crop-size 512 \
  --eval-size 256 512 \
  --batch-size 8 \
  --epochs 2 \
  --style-weight 1e6 \
  --content-weight 1.0 \
  --tv-weight 0.0 \
  --color-weight 20000 \
  --style-layer-weights conv1_1=4.0 conv2_1=3.0 conv3_1=1.0 conv4_1=0.5 conv5_1=0.25 \
  --checkpoint-dir "$ARTIST_DIR/checkpoints/van_gogh_multiscale" \
  --checkpoint-every 500 \
  --preview-every 200 \
  "$@"
