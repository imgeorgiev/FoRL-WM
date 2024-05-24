import gym
import numpy as np
import torch

import os, sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm

import hydra
from time import time

from envs import make_env
from pwm.utils.common import seeding
from common import TASK_SET
from copy import deepcopy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
import wandb
from pathlib import Path
from glob import glob
import pandas as pd
from torch.nn.utils.clip_grad import clip_grad_norm_
from envs.dmcontrol import make_env as make_dm_control_env


from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

torch.backends.cudnn.benchmark = True


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    task = job_config["task"]  # job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{task}_sweep_{job_config['general']['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{task}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


class MultitaskWrapper(gym.Wrapper):
    """
    Wrapper for multi-task environments.
    """

    def __init__(self, cfg, envs):
        super().__init__(envs[0])
        self.cfg = cfg
        self.envs = envs
        self._task = cfg.tasks[0]
        self._task_idx = 0
        self._obs_dims = [env.observation_space.shape[0] for env in self.envs]
        self._action_dims = [env.action_space.shape[0] for env in self.envs]
        self._episode_lengths = [env.max_episode_steps for env in self.envs]
        self._obs_shape = (max(self._obs_dims),)
        self._action_dim = max(self._action_dims)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )

    @property
    def task(self):
        return self._task

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def _env(self):
        return self.envs[self.task_idx]

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _pad_obs(self, obs):
        if obs.shape != self._obs_shape:
            obs = torch.cat(
                (
                    obs,
                    torch.zeros(
                        self._obs_shape[0] - obs.shape[0],
                        dtype=obs.dtype,
                        device=obs.device,
                    ),
                )
            )
        return obs

    def reset(self, task_idx=-1):
        self._task_idx = task_idx
        self._task = self.cfg.tasks[task_idx]
        self.env = self._env
        return self._pad_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(
            action[: self.env.action_space.shape[0]]
        )
        return self._pad_obs(obs), reward, done, info


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def eval(agent, env, eval_episodes):
    """Evaluate a TD-MPC2 agent."""
    results = dict()
    # for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
    ep_rewards, ep_successes = [], []
    for _ in range(eval_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:
            now = time()
            action = agent.act(obs, t == 0, True, None)
            action = action.cpu().numpy()
            print(f"took {time() - now:.6f} to act")
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
        ep_rewards.append(ep_reward)
        ep_successes.append(info["success"])
    results.update(
        {
            f"episode_reward": np.nanmean(ep_rewards),
            f"episode_success": np.nanmean(ep_successes),
        }
    )
    return results


@hydra.main(config_path="cfg", config_name="config_multitask.yaml", version_base="1.2")
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
                    `task`: task name (or mt30/mt80 for multi-task training)
                    `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
                    `steps`: number of training/environment steps (default: 10M)
                    `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
                    $ python train.py task=mt80 model_size=48
                    $ python train.py task=mt30 model_size=317
                    $ python train.py task=dog-run steps=7000000
    ```
    """
    assert torch.cuda.is_available()

    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    seeding(cfg.general.seed)

    task = cfg.task
    task_set = TASK_SET["mt80"] if "mt80" in cfg.data_dir else TASK_SET["mt30"]
    task_id = task_set.index(task)
    env = make_dm_control_env(cfg)

    os.makedirs(logdir, exist_ok=True)

    # Make algorithm
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = instantiate(cfg.alg, obs_dim=obs_dim, act_dim=act_dim)

    # load model
    if cfg.general.checkpoint:
        agent.load(cfg.general.checkpoint)
        agent.wm_bootstrapped = True

    cfg.buffer.buffer_size = 550_450_000 // 100

    # load dataset; buffer must happen here!
    buffer = instantiate(cfg.buffer)

    cfg.episode_length = 101 if "mt80" in cfg.data_dir else 501
    fp = Path(os.path.join(cfg.data_dir, "*.pt"))
    fps = sorted(glob(str(fp)))
    assert len(fps) > 0, f"No data found at {fp}"
    print(f"Found {len(fps)} files in {fp}")
    for fp in tqdm(fps, desc="Loading data"):
        print("Loading", fp)
        td = torch.load(fp)
        assert td.shape[1] == cfg.episode_length, (
            f"Expected episode length {td.shape[1]} to match config episode length {cfg.episode_length}, "
            f"please double-check your config."
        )
        idx = torch.all(td["task"] == task_id, dim=1)
        td = td[idx]
        print(f"Found {len(td)} episodes in file.")
        if td.shape[0] != 0:
            buffer.add_batch(td)

    if buffer.num_eps == 0:
        raise ValueError("No data found for task", task)

    # pre-train
    if cfg.general.pretrain_steps:
        print("Pretraining world model")
        for i in tqdm(range(0, cfg.general.pretrain_steps)):
            obs, act, rew = buffer.sample()
            obs = obs[..., :obs_dim]
            act = act[..., :act_dim]
            agent.wm_optimizer.zero_grad()
            wm_loss, dyn_loss, rew_loss = agent.compute_wm_loss(
                obs, act, rew, task=None
            )
            wm_loss.backward()
            wm_grad_norm = clip_grad_norm_(agent.wm.parameters(), agent.wm_grad_norm)
            agent.wm_optimizer.step()

            if i % 10_000 == 0:
                metrics = {
                    "pretrain/wm_loss": wm_loss.item(),
                    "pretrain/dyn_loss": dyn_loss,
                    "pretrain/rew_loss": rew_loss,
                    "pretrain/wm_grad_norm": wm_grad_norm.item(),
                }
                if cfg.general.run_wandb:
                    print(
                        f"[{i}/{cfg.general.pretrain_steps}]  WML:{wm_loss.item():.3f}  DL:{dyn_loss:.3f}  RL:{rew_loss:.3f}"
                    )
                    wandb.log(metrics)

        agent.save(f"pretrain_{cfg.general.pretrain_steps}", logdir)

    # train from dataset
    start_time = time()
    metrics_log = []
    for i in range(cfg.epochs):
        obs, act, rew = buffer.sample()
        obs = obs[..., :obs_dim]
        act = act[..., :act_dim]
        train_metrics = agent.update(obs, act, rew, None, cfg.finetune_wm)

        metrics = {
            "iteration": i,
            "total_time": time() - start_time,
        }
        metrics.update(train_metrics)

        # Evaluate agent periodically
        if i % cfg.eval_freq == 0:
            metrics.update(eval(agent, env, cfg.general.eval_runs))
            reward = metrics[f"episode_reward"]
            print(f"R: {reward:.2f}")
            if i > 0:
                agent.save(f"model_{i}", logdir)

        if i % 100 == 0:
            if "wm_loss" not in metrics:
                metrics["wm_loss"] = np.nan
            print(
                "[{:}/{:}]  AL:{:.3f}  VL:{:.3f}  WML:{:.3f}".format(
                    i,
                    cfg.epochs,
                    metrics["actor_loss"],
                    metrics["value_loss"],
                    metrics["wm_loss"],
                )
            )

            metrics_log.append(metrics)

            if cfg.general.run_wandb:
                wandb.log(metrics)

    agent.save(f"model_final", logdir)
    print("Final evaluation")

    metrics.update(eval(agent, env, cfg.general.eval_runs))
    reward = metrics[f"episode_reward"]
    print(f"Final reward: {reward:.2f}")

    # # Now do planning
    # agent.planning = True
    # planning_metrics = eval(agent, env, task_set, task_id, cfg.general.eval_runs)
    # metrics["episode_reward_planning"] = planning_metrics[f"episode_reward+{task}"]
    # metrics["episode_success_planning"] = planning_metrics[f"episode_success+{task}"]
    # print(f"Final reward with planning: {metrics['episode_reward_planning']:.2f}")

    if cfg.general.run_wandb:
        wandb.log(metrics)
        wandb.finish()
    print("\nTraining completed successfully")

    metrics_log.append(metrics)
    df = pd.DataFrame(metrics_log)
    df.to_csv(f"{logdir}/{task}_results.csv")


if __name__ == "__main__":
    train()
