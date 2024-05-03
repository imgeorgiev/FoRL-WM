import os, time, copy
from tensorboardX import SummaryWriter
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Optional, List, Tuple
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from gym import Env
import tensordict
from tensordict import TensorDict
from itertools import chain
import torch.nn.functional as F
from collections import OrderedDict


from forl.utils.common import *
import forl.utils.torch_utils as tu
from forl.utils.running_mean_std import RunningMeanStd
from forl.utils.dataset import CriticDataset
from forl.utils.time_report import TimeReport
from forl.utils.average_meter import AverageMeter
from forl.models.model_utils import Ensemble
from forl.utils.buffer import Buffer
from forl.models.world_model import WorldModel

tensordict.set_lazy_legacy(False).set()


class FOWM:
    """
    Short Horizon Actor Critic (SHAC) algorithm based on the paper
    Xu et al. Accelerated Policy Learning with Parallel Differentiable Simulation
    https://arxiv.org/abs/2204.07137
    """

    def __init__(
        self,
        env: Env,
        actor_config: DictConfig,
        critic_config: DictConfig,
        model_config: DictConfig,
        reward_config: DictConfig,
        terminate_config: DictConfig,
        horizon: int,  # horizon for short rollouts
        max_epochs: int,  # number of short rollouts to do (i.e. epochs)
        logdir: str,
        latent_dim: int,
        actor_grad_norm: Optional[float] = None,  # clip grad norms during training
        critic_grad_norm: Optional[float] = None,  # clip grad norms during training
        num_critics: int = 3,  # for critic ensembling
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        model_lr: float = 2e-3,
        betas: Tuple[float, float] = (0.7, 0.95),
        lr_schedule: str = "linear",
        gamma: float = 0.99,  # discount factor
        lam: float = 0.95,  # for TD(lambda)
        obs_rms: bool = False,  # running normalization of observations
        rew_rms: bool = False,
        ret_rms: bool = False,  # running normalization of returns
        critic_iterations: int = 16,
        critic_batches: int = 4,
        critic_method: str = "td-lambda",
        wm_batch_size: int = 256,
        wm_iterations: int = 2,
        wm_grad_norm: float = 20.0,
        save_interval: int = 500,  # how often to save policy
        device: str = "cuda",
        save_data: bool = False,
    ):
        # sanity check parameters
        assert horizon > 0
        assert max_epochs >= 0
        assert actor_lr >= 0
        assert critic_lr >= 0
        assert lr_schedule in ["linear", "constant"]
        assert 0 < gamma <= 1
        assert 0 < lam <= 1
        assert critic_iterations > 0
        assert critic_batches > 0
        assert critic_method in ["one-step", "td-lambda"]
        assert save_interval > 0
        # assert (horizon - 1) % wm_batch_size == 0

        self.env = env
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.save_data = save_data
        self.episode_data = [None] * self.num_envs

        self.horizon = horizon
        self.max_epochs = max_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model_lr = model_lr
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        self.lam = lam

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        self.critic_batch_size = self.num_envs * self.horizon // critic_batches
        self.wm_iterations = wm_iterations
        self.wm_batch_size = wm_batch_size
        self.wm_grad_norm = wm_grad_norm
        self.wm_bootstrapped = False

        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        self.rew_rms = None
        if rew_rms:
            self.rew_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.env_name = self.env.__class__.__name__
        self.name = self.__class__.__name__ + "_" + self.env_name

        # Buffer contains un-normalized data
        self.buffer = Buffer(
            buffer_size=10_000_000,
            batch_size=self.wm_batch_size,
            horizon=self.horizon,
            device=device,
        )

        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm
        self.save_interval = save_interval

        self.log_dir = logdir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))

        # Create actor and critic
        self.actor = instantiate(
            actor_config,
            obs_dim=latent_dim,
            action_dim=self.num_actions,
        ).to(self.device)

        critics = [
            instantiate(
                critic_config,
                obs_dim=latent_dim,
            ).to(self.device)
            for _ in range(num_critics)
        ]
        self.critic = Ensemble(critics)

        self.wm = WorldModel(
            self.num_obs, self.num_actions, self.latent_dim, mlp_dim=512, simnorm_dim=8
        ).to(self.device)

        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            self.actor_lr,
            betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            self.critic_lr,
            betas,
        )

        self.wm_optimizer = torch.optim.Adam(
            [
                {"params": self.wm._encoder.parameters()},
                {"params": self.wm._dynamics.parameters()},
                {"params": self.wm._reward.parameters()},
                {"params": self.wm._terminate.parameters()},
                {"params": (self.wm._task_emb.parameters() if False else [])},
            ],
            lr=self.model_lr,
        )

        # replay buffer
        self.obs_buf = torch.zeros(
            (self.horizon, self.num_envs, latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.term_buf = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.ret = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_primal = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.best_policy_loss = torch.inf
        self.actor_loss = torch.inf
        self.value_loss = torch.inf
        self.actor_grad_norm_before = torch.inf
        self.actor_grad_norm_after = torch.inf
        self.early_termination = 0
        self.episode_end = 0
        self.last_log_steps = 0

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_primal_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.horizon_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

    @property
    def mean_horizon(self):
        return self.horizon_length_meter.get_mean()

    # def model_step(self, obs, act, autoregressive=False):
    #     if autoregressive:
    #         H = act.shape[0]
    #         next_obs = [obs]
    #         # obs = obs[0]
    #         for h in range(H):
    #             obs = self.model(torch.cat((obs, act[h]), dim=-1))
    #             next_obs.append(obs)
    #         obs = torch.stack(tuple(next_obs[:-1]))
    #         next_obs = torch.stack(tuple(next_obs[1:]))
    #         inputs = torch.cat((obs, act), dim=-1)
    #     else:
    #         inputs = torch.cat((obs, act), dim=-1)
    #         next_obs = self.model(inputs)
    #     rew = self.reward(inputs)
    #     term = self.terminate(inputs)
    #     return next_obs, rew.squeeze(), term.squeeze()

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.horizon + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.horizon + 1, self.num_envs), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        primal = None

        # copy running mean and std so that we don't change during training
        # if self.obs_rms is not None:
        # obs_rms = copy.deepcopy(self.obs_rms)

        # initialize trajectory to cut off gradients between epochs
        try:
            # Note this doesn't reset the env, just re-inits the gradients
            obs = self.env.reset(grads=True)
        except:
            print_error(
                "Your environment should have a reset method that accepts grads=True"
            )
            raise AttributeError

        # update and normalize obs
        # if self.obs_rms:
        #     # self.obs_rms.update(obs)
        #     obs = self.obs_rms.normalize(obs)
        #     gt_obs = obs.clone()

        z = self.wm.encode(obs, task=None)

        # keeps track of the current length of the rollout
        rollout_len = torch.zeros((self.num_envs,), device=self.device)

        # Start short horizon rollout
        for i in range(self.horizon):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = z.clone()

            # act in environment
            actions = self.actor(z, deterministic=deterministic)
            actions = torch.tanh(actions)
            # TODO really move tanh inside actor
            z, rew, term = self.wm.step(z, actions, task=None)
            obs, gt_rew, gt_done, info = self.env.step(actions)
            term = info["termination"]
            gt_term = info["termination"]
            gt_trunc = info["truncation"]
            real_obs = info["obs_before_reset"]
            primal = info["primal"]

            # log data to buffer
            with torch.no_grad():

                for j in range(self.num_envs):
                    td = TensorDict(
                        dict(
                            obs=real_obs[j].unsqueeze(0),
                            action=actions[j].unsqueeze(0),
                            reward=gt_rew[j][None],
                            term=gt_term[j][None],
                        ),
                        (1,),
                    )
                    self.episode_data[j].append(td)

                gt_done_env_ids = gt_done.nonzero(as_tuple=False).squeeze(-1)
                for j in gt_done_env_ids:
                    td = torch.cat(self.episode_data[j])
                    self.buffer.add(td)

                    # reinint data tracker with with nan action and rewards
                    a = torch.full_like(torch.zeros(1, self.num_actions), torch.nan).to(
                        self.device
                    )
                    r = torch.full_like(
                        torch.zeros(
                            1,
                        ),
                        torch.nan,
                    ).to(self.device)
                    tt = torch.full_like(
                        torch.zeros(
                            1,
                        ),
                        torch.nan,
                        dtype=torch.bool,
                    ).to(self.device)
                    td = TensorDict(
                        dict(obs=obs[j].unsqueeze(0), action=a, reward=r, term=tt),
                        (1,),
                    )
                    self.episode_data[j] = [td]

            with torch.no_grad():
                raw_rew = gt_rew.clone()

            # # update and normalize obs
            # if self.obs_rms:
            #     # self.obs_rms.update(gt_obs)
            #     gt_obs = self.obs_rms.normalize(gt_obs)
            #     # real_obs = self.obs_rms.normalize(real_obs)

            self.episode_length += 1
            rollout_len += 1

            # sanity check; remove?
            if (~torch.isfinite(obs)).sum() > 0:
                print_warning("Got inf obs")

            next_values[i + 1] = self.critic(z).min(dim=0).values.squeeze()

            # handle terminated environments which stopped for some bad reason
            # since the reason is bad we set their value to 0
            term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
            for id in term_env_ids:
                next_values[i + 1, id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print_error("next value error")
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            done = gt_term | gt_trunc  # NOTE TEMPORARY
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            gt_done = gt_term | gt_trunc
            gt_done_env_ids = gt_done.nonzero(as_tuple=False).squeeze(-1)

            # for all done envs we reset observations and cut off gradients
            # Note this is important to do after critic next value compuataion!
            gt_z = self.wm.encode(obs, task=None)
            obs = torch.where(done[..., None], gt_z, z)

            self.early_termination += torch.sum(term).item()
            self.episode_end += torch.sum(gt_trunc).item()

            if i < self.horizon - 1:
                # first terminate all rollouts which are 'done'
                returns = (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values[i + 1, done_env_ids]
                )
                actor_loss[done_env_ids] += returns
            else:
                # terminate all envs because we reached the end of our rollout
                returns = (
                    -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                )
                actor_loss += returns

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon - 1:
                    self.done_mask[i] = gt_done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.term_buf[i] = gt_term.clone().to(torch.float32)
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                # collect episode stats
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_primal -= primal
                self.episode_gamma *= self.gamma

                # dump data from done episodes
                self.episode_loss_meter.update(self.episode_loss[gt_done_env_ids])
                self.episode_discounted_loss_meter.update(
                    self.episode_discounted_loss[gt_done_env_ids]
                )
                self.episode_primal_meter.update(self.episode_primal[gt_done_env_ids])
                self.episode_length_meter.update(self.episode_length[gt_done_env_ids])
                self.horizon_length_meter.update(rollout_len[gt_done_env_ids])

                # reset trackers
                rollout_len[gt_done_env_ids] = 0
                self.episode_loss[gt_done_env_ids] = 0.0
                self.episode_discounted_loss[gt_done_env_ids] = 0.0
                self.episode_primal[gt_done_env_ids] = 0.0
                self.episode_length[gt_done_env_ids] = 0
                self.episode_gamma[gt_done_env_ids] = 1.0
        self.horizon_length_meter.update(rollout_len)

        self.actor_loss_before = actor_loss.mean().item()

        if self.ret_rms is not None:
            self.ret_rms.update(actor_loss)
            actor_loss /= torch.sqrt(self.ret_rms.var + 1e-5)
        else:
            actor_loss /= self.horizon

        actor_loss = actor_loss.mean()

        self.actor_loss = actor_loss.detach().item()

        self.step_count += self.horizon * self.num_envs

        return actor_loss

    @torch.no_grad()
    def eval(self, num_games, deterministic=True):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_length = torch.zeros(self.num_envs, dtype=int)
        episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        obs = self.env.reset()
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        z = self.wm.encode(obs, task=None)

        games_cnt = 0
        while games_cnt < num_games:
            # if self.obs_rms is not None:
            #     obs = self.obs_rms.normalize(obs)

            actions = self.actor(z, deterministic=deterministic)
            actions = torch.tanh(actions)
            z, rew, trunc = self.wm.step(z, actions, task=None)

            _, _, done, _ = self.env.step(actions)

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print(
                        "loss = {:.2f}, len = {}".format(
                            episode_loss[done_env_id].item(),
                            episode_length[done_env_id],
                        )
                    )
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(
                        episode_discounted_loss[done_env_id].item()
                    )
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.0
                    episode_discounted_loss[done_env_id] = 0.0
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.0
                    games_cnt += 1

        mean_episode_length = torch.mean(torch.Tensor(episode_length_his))
        mean_policy_loss = torch.mean(torch.Tensor(episode_loss_his))
        mean_policy_discounted_loss = torch.mean(
            torch.Tensor(episode_discounted_loss_his)
        )

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = (
                    self.gamma
                    * (
                        self.next_values[i] * self.done_mask[i]
                        + Bi * (1.0 - self.done_mask[i])
                    )
                    + self.rew_buf[i]
                )
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample["obs"]).squeeze(-2)
        target_values = batch_sample["target_values"]
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    def train(self):

        self.save("init_policy")

        # path = "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-hopper/2/pretrain_tdmpc_simple/models/150000.pt"
        # print("Loading world model from", path)
        # self.load_wm(path)

        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        self.time_report.add_timer("world model training")
        self.time_report.start_timer("algorithm")

        # initializations
        obs = self.env.reset()
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_primal = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            # save data with nan action and rewards

            for id in range(self.num_envs):
                act = torch.full_like(torch.zeros(1, self.num_actions), torch.nan).to(
                    self.device
                )
                rew = torch.full_like(
                    torch.zeros(
                        1,
                    ),
                    torch.nan,
                ).to(self.device)
                term = torch.full_like(
                    torch.zeros(
                        1,
                    ),
                    torch.nan,
                    dtype=torch.bool,
                ).to(self.device)
                td = TensorDict(
                    dict(obs=obs[id].unsqueeze(0), action=act, reward=rew, term=term),
                    (1,),
                )
                self.episode_data[id] = [td]

        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            self.actor_grad_norm_before = tu.grad_norm(self.actor.parameters())
            self.actor_grad_norm_after = clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_norm
            )

            # sanity check
            if torch.isnan(self.actor_grad_norm_before):
                print_error("NaN gradient")
                raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                # actor learning rate
                actor_lr = (1e-5 - self.actor_lr) * float(
                    epoch / self.max_epochs
                ) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr

                # critic learning rate
                critic_lr = (1e-5 - self.critic_lr) * float(
                    epoch / self.max_epochs
                ) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr

                # world model learning rate
                model_lr = (1e-5 - self.model_lr) * float(
                    epoch / self.max_epochs
                ) + self.model_lr
                for param_group in self.wm_optimizer.param_groups:
                    param_group["lr"] = model_lr
            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(
                    self.critic_batch_size,
                    self.obs_buf,
                    self.target_values,
                )
            self.time_report.end_timer("prepare critic dataset")

            # critic training!
            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    # print("critic obs mean", batch_sample["obs"].mean())
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    critic_grad_norm = clip_grad_norm_(
                        self.critic.parameters(), self.critic_grad_norm
                    )
                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                self.value_loss = total_critic_loss / batch_cnt
                print(
                    f"value iter {j + 1}/{self.critic_iterations}, loss = {self.value_loss:.2f}",
                    end="\r",
                )
            self.time_report.end_timer("critic training")

            tot_wm_loss = tot_dynamics_loss = tot_reward_loss = tot_term_loss = 0.0
            sample_rew_mean = sample_rew_var = 0.0
            sample_obs_mean = sample_obs_var = 0.0
            wm_grad_norm = torch.inf

            self.time_report.start_timer("world model training")

            # world model training!
            if self.buffer.num_eps > 1:
                if self.wm_bootstrapped:
                    iters = self.wm_iterations
                else:
                    iters = self.env.episode_length
                    print(f"training wm for {iters} iterations")
                    self.wm_bootstrapped = True

                for i in range(0, iters):
                    obs, act, rew, term = self.buffer.sample()
                    # rew = rew.squeeze(-1)
                    # term = term.squeeze(-1)
                    if self.rew_rms:
                        for d in range(rew.shape[0]):
                            self.rew_rms.update(rew[d])
                        for d in range(rew.shape[0]):
                            rew[d] = self.rew_rms.normalize(rew[d])
                    sample_rew_mean += rew.mean().item()
                    sample_rew_var += rew.var().item()
                    sample_obs_mean += obs.mean(dim=0).mean().item()
                    sample_obs_var += obs.var(dim=0).mean().item()

                    self.wm_optimizer.zero_grad()
                    loss, dyn_loss, rew_loss, term_loss = self.compute_wm_loss(
                        obs, act, rew, term
                    )
                    loss.backward()
                    wm_grad_norm = clip_grad_norm_(
                        self.wm.parameters(), self.wm_grad_norm
                    )
                    self.wm_optimizer.step()
                    tot_wm_loss += loss.item()
                    tot_dynamics_loss += dyn_loss
                    tot_reward_loss += rew_loss
                    tot_term_loss += term_loss
                    print(
                        f"wm iter {j + 1}/{self.wm_iterations}, loss = {loss:.2f}",
                        end="\r",
                    )

                # normalize for logging; TODO simplify
                tot_wm_loss /= iters
                tot_dynamics_loss /= iters
                tot_reward_loss /= iters
                tot_term_loss /= iters
                sample_rew_mean /= iters
                sample_rew_var /= iters
                sample_obs_mean /= iters
                sample_obs_var /= iters

            self.time_report.end_timer("world model training")

            self.iter_count += 1

            time_end_epoch = time.time()

            fps = self.horizon * self.num_envs / (time_end_epoch - time_start_epoch)

            # logging
            self.log_scalar("actor_lr", lr)
            self.log_scalar("actor_loss", self.actor_loss)
            self.log_scalar("value_loss", self.value_loss)
            self.log_scalar("wm_loss", tot_wm_loss)
            self.log_scalar("dynamics_loss", tot_dynamics_loss)
            self.log_scalar("reward_loss", tot_reward_loss)
            self.log_scalar("term_loss", tot_term_loss)
            self.log_scalar("rollout_len", self.mean_horizon)
            self.log_scalar("fps", fps)

            mean_episode_length = self.episode_length_meter.get_mean()
            mean_policy_loss = self.episode_loss_meter.get_mean()
            mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()
            mean_episode_primal = self.episode_primal_meter.get_mean()

            if mean_policy_loss < self.best_policy_loss:
                print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                self.save(f"best_policy")
                self.best_policy_loss = mean_policy_loss

            self.log_scalar("policy_loss", mean_policy_loss)
            self.log_scalar("rewards", -mean_policy_loss)
            self.log_scalar("primal", -mean_episode_primal)
            self.log_scalar("policy_discounted_loss", mean_policy_discounted_loss)
            self.log_scalar("best_policy_loss", self.best_policy_loss)
            self.log_scalar("episode_lengths", mean_episode_length)
            ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()
            self.log_scalar("ac_std", ac_stddev)
            # TODO these shouldn't be class variables
            self.log_scalar("actor_grad_norm", self.actor_grad_norm_before)
            self.log_scalar("critic_grad_norm", critic_grad_norm)
            self.log_scalar("wm_grad_norm", wm_grad_norm)
            self.log_scalar("episode_end", self.episode_end)
            self.log_scalar("early_termination", self.early_termination)

            self.log_scalar("sample_rew_mean", sample_rew_mean)
            self.log_scalar("sample_rew_var", sample_rew_var)
            self.log_scalar("sample_obs_mean", sample_obs_mean)
            self.log_scalar("sample_obs_var", sample_obs_var)

            print(
                "[{:}/{:}]  R:{:.2f}  T:{:.1f}  H:{:.1f}  S:{:}  FPS:{:0.0f}  pi_loss:{:.2f}/{:.2f}  pi_grad:{:.2f}/{:.2f}  v_loss:{:.2f}  wm_loss:{:.2f}  rew_loss:{:.2f}  dyn_loss:{:.2f}".format(
                    self.iter_count,
                    self.max_epochs,
                    -mean_policy_loss,
                    mean_episode_length,
                    self.mean_horizon,
                    self.step_count,
                    fps,
                    self.actor_loss_before,
                    self.actor_loss,
                    self.actor_grad_norm_before,
                    self.actor_grad_norm_after,
                    self.value_loss,
                    tot_wm_loss,
                    tot_reward_loss,
                    tot_dynamics_loss,
                )
            )

            self.writer.flush()

            if self.iter_count % self.save_interval == 0:
                name = self.name + f"_iter{self.iter_count}_rew{-mean_policy_loss:0.0f}"
                self.save(name)

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        self.save("final_policy")

        self.writer.close()

    def save(self, filename):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms,
                "ret_rms": self.ret_rms,
            },
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.to(self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic.to(self.device)
        self.obs_rms = (
            checkpoint["obs_rms"].to(self.device)
            if checkpoint["obs_rms"] is not None
            else None
        )
        self.ret_rms = (
            checkpoint["ret_rms"].to(self.device)
            if checkpoint["ret_rms"] is not None
            else None
        )

    def load_wm(self, path):
        print("Loading world model from", path)
        checkpoint = torch.load(path)
        checkpoint = checkpoint["model"]
        # filter policy away
        # keys = list(checkpoint.keys())
        # for key in keys:
        #     if "_pi" in key or "_Qs" in key:
        #         del checkpoint[key]
        new_odict = OrderedDict()
        for key, value in checkpoint.items():
            print(key)
            if "_pi" in key:
                pass
            elif "_Qs" in key:
                pass
            else:
                if "_encoder" in key:
                    key = key.replace("state.", "")
                new_odict[key] = value
            # new_odict[key.replace("wm.", "")] = value
        missing = [
            "_terminate.0.weight",
            "_terminate.0.bias",
            "_terminate.0.ln.weight",
            "_terminate.0.ln.bias",
            "_terminate.1.weight",
            "_terminate.1.bias",
            "_terminate.1.ln.weight",
            "_terminate.1.ln.bias",
            "_terminate.2.weight",
            "_terminate.2.bias",
            # "_terminate.2.ln.weight",
            # "_terminate.2.ln.bias",
        ]
        for k in missing:
            replace_key = k.replace("_terminate", "_reward")
            new_odict[k] = torch.zeros_like(new_odict[replace_key])
        new_odict["_terminate.2.ln.weight"] = torch.zeros((1,))
        new_odict["_terminate.2.ln.bias"] = torch.zeros((1,))
        self.wm.load_state_dict(new_odict)

    def pretrain_wm(self, paths, num_iters):
        if type(paths) != List:
            paths = [paths]
        for path in paths:
            print("loading", path)
            td = torch.load(path)
            self.buffer.add_batch(td)

        print(f"Pretraining world model for {num_iters} iters")
        for i in range(0, num_iters):
            obs, act, rew, term = self.buffer.sample()
            if self.rew_rms:
                for d in range(rew.shape[0]):
                    self.rew_rms.update(rew[d])
                for d in range(rew.shape[0]):
                    rew[d] = self.rew_rms.normalize(rew[d])

            self.wm_optimizer.zero_grad()
            loss, dyn_loss, rew_loss, term_loss = self.compute_wm_loss(
                obs, act, rew, term
            )
            loss.backward()
            wm_grad_norm = clip_grad_norm_(self.wm.parameters(), self.wm_grad_norm)
            self.wm_optimizer.step()
            print(
                f"[{i}/{num_iters}]  L:{loss.item():.3f}  GN:{wm_grad_norm:.3f}  DL:{dyn_loss:.3f}  RL:{rew_loss:.3f}  TL:{term_loss:.3f}",
                end="\r",
            )

    def log_scalar(self, scalar, value):
        """Helper method for consistent logging"""
        self.writer.add_scalar(f"{scalar}", value, self.step_count)

    def compute_wm_loss(self, obs, act, rew, term):
        if term.dtype == torch.bool:
            term = term.float()

        horizon, batch_size, _ = obs.shape
        assert horizon == self.horizon + 1
        discount = (
            (self.gamma ** torch.arange(self.horizon))
            .view((self.horizon, 1, 1))
            .to(self.device)
        )

        # Compute targets
        with torch.no_grad():
            next_z = self.wm.encode(obs[1:], task=None)

        # Latent rollout
        zs = torch.empty(
            self.horizon + 1,
            batch_size,
            self.latent_dim,
            device=self.device,
        )

        z = self.wm.encode(obs[0], task=None)
        zs[0] = z

        dynamics_loss = 0.0
        for t in range(self.horizon):
            z = self.wm.next(z, act[t], task=None)
            dynamics_loss += F.mse_loss(z, next_z[t]) * self.gamma**t
            zs[t + 1] = z

        _zs = zs[:-1]
        rew_hat = self.wm.reward(_zs, act, task=None)
        reward_loss = (rew_hat - rew) ** 2 * discount
        reward_loss = reward_loss.mean()
        term_hat = self.wm.terminate(_zs, act, task=None)
        term_loss = F.binary_cross_entropy(term_hat, term, reduction="none")
        term_loss = term_loss.mean()

        total_loss = dynamics_loss + reward_loss + term_loss
        total_loss /= self.horizon
        return (
            total_loss,
            dynamics_loss / self.horizon,
            reward_loss.item() / self.horizon,
            term_loss.item() / self.horizon,
        )
