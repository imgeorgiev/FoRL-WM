# PWM
Policy learning through World Model

## Installation

Tested only on Ubuntu 22.04. Requires Python, conda, Nvidia GPU with >12GB VRAM and 112GM RAM.

1. `git clone --recursive <repo>`
2. `cd PWM`
3. `conda env create -f environment.yaml`
4. `ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64` (hack to get CUDA to work inside conda)
5. `pip install -e .`
6. `cd external/tdmpc2 && pip install -e .`
7. Install dm_control following [these instructions](https://github.com/nicklashansen/tdmpc2).

## Examples

### Dflex single-task experiments

```
cd scripts
conda activate pwm
python train_dflex.py env=dflex_ant alg=pwm
```

The script is fully configured and usable with [hydra](https://hydra.cc/docs/intro/).

### Multi-task experiments

1. Pretrain world model on data with `python wmlab/train_wm.py task=mt80 datadir=<path/top/data>`
2. Train PWM on a task from the mt80 set `python scripts/train_multitask.py -cn config_mt80 task=hopper-hop general.checkpoint=<path/to/pretrained/world/model>`

Check configs and provide correct paths to data and checkpoints.

