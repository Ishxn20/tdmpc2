#!/usr/bin/env bash
# Launch TD-MPC2 on Crafter (Craftax Classic).
#
#   bash run.sh                    # the 1M-step benchmark run
#   bash run.sh compile=false      # any extra args are passed to train.py
#   bash run.sh task=craftax       # full Craftax instead of Crafter
#
# Logs to stdout and to ~/train.log.

set -euo pipefail

cd "$(dirname "$0")/tdmpc2"

# Craftax runs on CPU so PyTorch keeps the whole GPU. Without this, a CUDA build
# of JAX would preallocate ~75% of GPU memory and starve training.
export JAX_PLATFORMS=cpu

python train.py \
	task=crafter \
	model_size=5 \
	steps=1000000 \
	wandb_project=tdmpc2-crafter \
	wandb_entity=LEQ \
	save_video=false \
	"$@" 2>&1 | tee ~/train.log
