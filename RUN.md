# Running Crafter on the GPU box

## One-time setup

```
python3 -m venv ~/tdmpc2-env
source ~/tdmpc2-env/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 gymnasium==0.29.1 hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 omegaconf==2.3.0 numpy==1.24.4 \
  tensordict==0.8.3 torchrl==0.8.1 termcolor==2.4.0 tqdm==4.66.4 \
  pandas==2.0.3 imageio==2.34.1 craftax==1.6.1 wandb
wandb login
```

`craftax` pulls in `jax` and `flax` itself. `dm_control`, `mujoco`, `metaworld`,
`maniskill` and `myosuite` are **not** needed — they sit behind try/except imports
and are only used by the continuous-control tasks.

Check it worked:

```
python -c "import torch;print(torch.cuda.is_available())"
python -m unittest discover -s tests
```

`train.py` asserts on CUDA, so the first must print `True`. The second should
report `Ran 11 tests ... OK`.

## Every run

```
tmux new -s train
source ~/tdmpc2-env/bin/activate
cd ~/tdmpc2
bash run.sh
```

Detach with `Ctrl-b` then `d`. Reattach with `tmux attach -t train`.
Scroll with `Ctrl-b` then `[`, and `q` to exit scrolling.

Extra arguments are passed straight through to `train.py`:

```
bash run.sh compile=false            # skip torch.compile
bash run.sh task=craftax             # full Craftax (43 actions, 67 achievements)
bash run.sh horizon=3                # cheaper planning
bash run.sh steps=20000 eval_freq=5000 craftax_max_steps=1000   # quick shakedown
```

## What to watch

At `wandb.ai/LEQ/tdmpc2-crafter`:

| metric | meaning |
| --- | --- |
| `train/steps_per_second` | check first; multiply out to 1M to see what you have committed to |
| `train/episode_achievements` | must leave zero — collect-wood comes easily |
| `eval/crafter_score` | the benchmark number, every 50k steps |

Startup order, so a quiet terminal does not look like a hang:

1. matplotlib font-cache warning (harmless, once)
2. `Episode length` / `Discount factor`
3. run info table, then the wandb run URL
4. `Compiling update function with torch.compile...` — a minute or two of silence
5. an evaluation with an untrained model, before training starts
6. the seed phase, then a 10,000-update burst

## Known gaps

- **No intermediate checkpoints.** The model is saved only when the run finishes.
  A crash at step 900k loses everything.
- **`save_video=false` is deliberate.** Video records every step of an eval
  episode; at 10,000 steps that is a ~123 MB, 11-minute clip per eval. To watch
  the agent, evaluate separately afterwards with short episodes:
  `python evaluate.py task=crafter checkpoint=<path> craftax_max_steps=500 eval_episodes=3 save_video=true`
- **`horizon: 10`** is set in `config.yaml`, above the TD-MPC2 default of 3. It is
  3.3x the planning cost, and `rho: 0.5` means training barely weights depths
  6-10 while the planner fully uses them. Revisit both together once
  steps-per-second is known.
