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

# Activate the venv if it exists and is not already active, so the script works
# whether or not it was sourced first.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$HOME/tdmpc2-env/bin/activate" ]; then
	# shellcheck disable=SC1091
	source "$HOME/tdmpc2-env/bin/activate"
fi

# Ubuntu ships `python3`, not `python`; a venv provides both.
PYTHON="$(command -v python || command -v python3 || true)"
if [ -z "$PYTHON" ]; then
	echo "run.sh: no python found. Create the venv first -- see RUN.md." >&2
	exit 1
fi

# Craftax runs on CPU so PyTorch keeps the whole GPU. Without this, a CUDA build
# of JAX would preallocate ~75% of GPU memory and starve training.
export JAX_PLATFORMS=cpu

"$PYTHON" train.py \
	task=crafter \
	model_size=5 \
	steps=1000000 \
	wandb_project=tdmpc2-crafter \
	wandb_entity=LEQ \
	save_video=false \
	"$@" 2>&1 | tee ~/train.log
