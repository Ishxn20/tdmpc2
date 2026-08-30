<h1>TD-MPC2</span></h1>

Official implementation of

[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://www.tdmpc2.com) by

[Nicklas Hansen](https://nicklashansen.github.io), [Hao Su](https://cseweb.ucsd.edu/~haosu)\*, [Xiaolong Wang](https://xiaolonw.github.io)\* (UC San Diego)</br>

<img src="assets/0.gif" width="12.5%"><img src="assets/1.gif" width="12.5%"><img src="assets/2.gif" width="12.5%"><img src="assets/3.gif" width="12.5%"><img src="assets/4.gif" width="12.5%"><img src="assets/5.gif" width="12.5%"><img src="assets/6.gif" width="12.5%"><img src="assets/7.gif" width="12.5%"></br>

[[Website]](https://www.tdmpc2.com) [[Paper]](https://arxiv.org/abs/2310.16828) [[Models]](https://www.tdmpc2.com/models)  [[Dataset]](https://www.tdmpc2.com/dataset)

----

**Announcement (Apr 2025): support for episodic tasks!**

We have added support for episodic RL (tasks with terminations) in the latest release. This functionality can be enabled with `episodic=true` but remains disabled by default to ensure reproducibility of results across releases.

----


## Overview

TD-MPC**2** is a scalable, robust model-based reinforcement learning algorithm. It compares favorably to existing model-free and model-based methods across **104** continuous control tasks spanning multiple domains, with a *single* set of hyperparameters (*right*). We further demonstrate the scalability of TD-MPC**2** by training a single 317M parameter agent to perform **80** tasks across multiple domains, embodiments, and action spaces (*left*). 

<img src="assets/8.png" width="100%" style="max-width: 640px"><br/>

This repository contains code for training and evaluating both single-task online RL and multi-task offline RL TD-MPC**2** agents. We additionally open-source **300+** [model checkpoints](https://www.tdmpc2.com/models) (including 12 multi-task models) across 4 task domains: [DMControl](https://arxiv.org/abs/1801.00690), [Meta-World](https://meta-world.github.io/), [ManiSkill2](https://maniskill2.github.io/), and [MyoSuite](https://sites.google.com/view/myosuite), as well as our [30-task and 80-task datasets](https://www.tdmpc2.com/dataset) used to train the multi-task models. Our codebase supports both state and pixel observations. We hope that this repository will serve as a useful community resource for future research on model-based RL.

----

## Getting started

You will need a machine with a GPU and at least 12 GB of RAM for single-task online RL with TD-MPC**2**, and 128 GB of RAM for multi-task offline RL on our provided 80-task dataset. A GPU with at least 8 GB of memory is recommended for single-task online RL and for evaluation of the provided multi-task models (up to 317M parameters). Training of the 317M parameter model requires a GPU with at least 24 GB of memory.

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd docker && docker build . -t <user>/tdmpc2:1.0.1
```

This docker image contains all dependencies needed for running DMControl. We also provide a pre-built docker image [here](https://hub.docker.com/repository/docker/nicklashansen/tdmpc2/tags/1.0.1/sha256-b07d4e04d4b28ffd9a63ac18ec1541950e874bb51d276c7d09b36135f170dd93).

If you prefer to use `conda` rather than docker, start by running the following command:

```
conda env create -f docker/environment.yaml
```

The `docker/environment.yaml` file installs dependencies required for training on DMControl tasks. Other domains can be installed by following the instructions in `docker/environment.yaml`.

If you want to run ManiSkill2, you will additionally need to download and link the necessary assets by running

```
python -m mani_skill2.utils.download_asset all
```

which downloads assets to `./data`. You may move these assets to any location. Then, add the following line to your `~/.bashrc`:

```
export MS2_ASSET_DIR=<path>/<to>/<data>
```

and restart your terminal. Note that Meta-World requires MuJoCo 2.1.0 and `gym==0.21.0` which is becoming increasingly difficult to install. We host the unrestricted MuJoCo 2.1.0 license (courtesy of Google DeepMind) at [https://www.tdmpc2.com/files/mjkey.txt](https://www.tdmpc2.com/files/mjkey.txt). You can download the license by running

```
wget https://www.tdmpc2.com/files/mjkey.txt -O ~/.mujoco/mjkey.txt
```

Depending on your existing system packages, you may need to install other dependencies. See `docker/Dockerfile` for a list of recommended system packages.

----

## Supported tasks

This codebase provides support for all **104** continuous control tasks from **DMControl**, **Meta-World**, **ManiSkill2**, and **MyoSuite** used in our paper. It additionally supports discrete-action Crafter and Craftax through the accelerator-friendly Craftax implementation. Specifically, it supports 39 tasks from DMControl (including 11 custom tasks), 50 tasks from Meta-World, 5 tasks from ManiSkill2, and 10 tasks from MyoSuite. See below table for expected name formatting for each task domain:

| domain | task
| --- | --- |
| dmcontrol | dog-run
| dmcontrol | cheetah-run-backwards
| metaworld | mw-assembly
| metaworld | mw-pick-place-wall
| maniskill | pick-cube
| maniskill | pick-ycb
| myosuite  | myo-key-turn
| myosuite  | myo-key-turn-hard
| crafter   | crafter
| craftax   | craftax-classic
| craftax   | craftax

which can be run by specifying the `task` argument for `evaluation.py`. Multi-task training and evaluation is specified by setting `task=mt80` or `task=mt30` for the 80-task and 30-task sets, respectively. While you generally do not need to access the underlying task IDs or embeddings during training or evaluation of our multi-task models, the mapping from task name to task embedding used in our work can be found [here](https://github.com/nicklashansen/tdmpc2/blob/7ec6bc83a82a5188ca3faddc59aea83f430ab570/tdmpc2/common/__init__.py#L26). As of April 2025, our codebase also provides basic support for other MuJoCo/Box2d Gymnasium tasks; refer to the `envs` directory for a list of tasks. It should be relatively straightforward to add support for custom tasks by following the examples in `envs`.

**Note:** we also provide support for image observations in the DMControl tasks. Use argument `obs=rgb` if you wish to train visual policies.

### Crafter and Craftax

The `crafter` task uses Craftax Classic, which implements Crafter's 17-action game and 22 achievements in JAX. `craftax-classic` is an equivalent explicit alias, while `craftax` selects the extended 43-action, multi-level game. Both symbolic (`obs=state`) and 64x64 frame-stacked pixel (`obs=rgb`) observations are supported.

TD-MPC2 uses a categorical policy prior and categorical CEM planner for these tasks. Actions are stored as one-hot vectors in replay and converted to integer environment actions by the environment wrapper.

Install the environment with `pip install craftax==1.6.1`, or recreate the provided Conda environment. Example runs:

```
$ python train.py task=crafter obs=state model_size=5 enable_wandb=false
$ python train.py task=crafter obs=rgb model_size=5 enable_wandb=false
$ python train.py task=craftax obs=state model_size=5 enable_wandb=false
```

Observations are `1345`-dim for `crafter` and `8268`-dim for `craftax` in `obs=state`, or `(3*frame_stack, 64, 64)` uint8 frames in `obs=rgb`. Both tasks are episodic: the agent terminates on death (and, for `craftax`, on beating the boss), while reaching `craftax_max_steps` is treated as a truncation so that TD-MPC2 bootstraps through it. `episodic=true` is therefore enabled automatically.

Two progress metrics are reported. `episode_achievements` (console column `A`) is the number of distinct achievements unlocked in an episode -- out of 22 for `crafter`, 67 for `craftax`. `crafter_score` (console column `C`) is the actual benchmark score from Hafner et al.: each achievement's success *rate* across the evaluation episodes, combined with a geometric mean. Because it needs rates measured over many episodes, it is computed during evaluation only, not per training episode. Note this is not the mean of Craftax's own per-episode `score` field -- rates have to be aggregated across episodes before the geometric mean, not after. `episode_success` is only meaningful for `craftax`, where it marks a boss kill.

Relevant arguments:

| argument | default | description |
| --- | --- | --- |
| `craftax_max_steps` | `10_000` | environment steps before truncation |
| `craftax_device` | `cpu` | JAX device; defaults to CPU so PyTorch owns the training GPU |
| `frame_stack` | `3` | stacked frames for `obs=rgb` |
| `entropy_coef_discrete` | `1e-2` | entropy bonus for the categorical policy |
| `categorical_tau` | `1.0` | Gumbel-Softmax temperature of the policy prior |
| `categorical_min_prob` | `0.01` | probability floor kept by the planner per action |
| `max_seed_steps` | `10_000` | cap on the seed phase, which scales with episode length |

`entropy_coef_discrete` is separate from `entropy_coef` on purpose: the categorical `scaled_entropy` is bounded in `[0, 1]`, whereas the Gaussian one is unbounded and roughly `action_dim` times larger, so the continuous default would leave the categorical policy with effectively no exploration pressure.


## Example usage

We provide examples on how to evaluate our provided TD-MPC**2** checkpoints, as well as how to train your own TD-MPC**2** agents, below.

### Evaluation

See below examples on how to evaluate downloaded single-task and multi-task checkpoints.

```
$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
```

All single-task checkpoints expect `model_size=5`. Multi-task checkpoints are available in multiple model sizes. Available arguments are `model_size={1, 5, 19, 48, 317}`. Note that single-task evaluation of multi-task checkpoints is currently not supported. See `config.yaml` for a full list of arguments.

### Training

See below examples on how to train TD-MPC**2** on a single task (online RL) and on multi-task datasets (offline RL). We recommend configuring [Weights and Biases](https://wandb.ai) (`wandb`) in `config.yaml` to track training progress.

```
$ python train.py task=mt80 model_size=48 batch_size=1024
$ python train.py task=mt30 model_size=317 batch_size=1024
$ python train.py task=dog-run steps=7000000
$ python train.py task=walker-walk obs=rgb
```

We recommend using default hyperparameters for single-task online RL, including the default model size of 5M parameters (`model_size=5`). Multi-task offline RL benefits from a larger model size, but larger models are also increasingly costly to train and evaluate. Available arguments are `model_size={1, 5, 19, 48, 317}`. See `config.yaml` for a full list of arguments.

----

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
as well as the original TD-MPC paper:
```
@inproceedings{hansen2022tdmpc,
  title={Temporal Difference Learning for Model Predictive Control},
  author={Nicklas Hansen and Xiaolong Wang and Hao Su},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

----

## Contributing

You are very welcome to contribute to this project. Feel free to open an issue or pull request if you have any suggestions or bug reports, but please review our [guidelines](CONTRIBUTING.md) first. Our goal is to build a codebase that can easily be extended to new environments and tasks, and we would love to hear about your experience!

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
