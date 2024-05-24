import os, time
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import tensordict
import torch.nn.functional as F
from collections import OrderedDict


from pwm.utils.common import *
import pwm.utils.torch_utils as tu
from pwm.utils.running_mean_std import RunningMeanStd
from pwm.utils.dataset import CriticDataset
from pwm.models.model_utils import Ensemble

tensordict.set_lazy_legacy(False).set()


class PWM:
    """
    Policy learning through World Models
    """

    def __init__(
        self,
        actor_config: DictConfig,
        critic_config: DictConfig,
        world_model_config: DictConfig,
        horizon: int,  # horizon for short rollouts
        latent_dim: int,
        obs_dim: int,
        act_dim: int,
        actor_grad_norm: float = 1.0,  # clip grad norms during training
        critic_grad_norm: float = 100.0,  # clip grad norms during training
        num_critics: int = 3,  # for critic ensembling
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        model_lr: float = 2e-3,
        lr_schedule: str = "linear",
        gamma: float = 0.99,  # discount factor
        lam: float = 0.95,  # for TD(lambda)
        obs_rms: bool = False,  # running normalization of observations
        rew_rms: bool = False,
        ret_rms: bool = False,  # running normalization of returns
        critic_iterations: int = 16,
        critic_batches: int = 4,
        critic_method: str = "td-lambda",
        wm_grad_norm: float = 20.0,
        device: str = "cuda",
        save_data: bool = False,
        detach: bool = False,
        planning: bool = False,
        num_pi_trajs: int = 24,
        num_samples: int = 512,
        min_std: float = 0.05,
        max_std: float = 2.0,
        iterations: int = 6,
        num_elites: int = 64,
        temperature: float = 0.5,
    ):
        # sanity check parameters
        assert horizon > 0
        assert actor_lr >= 0
        assert critic_lr >= 0
        assert lr_schedule in ["linear", "constant"]
        assert 0 < gamma <= 1
        assert 0 < lam <= 1
        assert critic_iterations > 0
        assert critic_batches > 0
        assert critic_method in ["one-step", "td-lambda"]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.save_data = save_data

        # planning
        self.planning = planning
        self.num_pi_trajs = num_pi_trajs
        self.num_samples = num_samples
        self.min_std = min_std
        self.max_std = max_std
        self.iterations = iterations
        self.num_elites = num_elites
        self.temperature = temperature

        self.horizon = horizon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model_lr = model_lr
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        self.lam = lam
        self.detach = detach
        self.critic_batches = critic_batches

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        self.wm_grad_norm = wm_grad_norm
        self.wm_bootstrapped = False

        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,), device=self.device)

        self.rew_rms = None
        if rew_rms:
            self.rew_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm

        # Create actor and critic
        self.actor = instantiate(
            actor_config,
            obs_dim=latent_dim,
            action_dim=self.act_dim,
        ).to(self.device)

        critics = [
            instantiate(
                critic_config,
                obs_dim=latent_dim,
            ).to(self.device)
            for _ in range(num_critics)
        ]
        self.critic = Ensemble(critics)

        self.wm = instantiate(
            world_model_config,
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            self.critic_lr,
        )

        self.wm_optimizer = torch.optim.Adam(
            [
                {"params": self.wm._encoder.parameters()},
                {"params": self.wm._dynamics.parameters()},
                {"params": self.wm._reward.parameters()},
                {"params": (self.wm._task_emb.parameters() if False else [])},
            ],
            lr=self.model_lr,
        )

        print(self.actor)
        print(self.critic)
        print(self.wm)

    def compute_actor_loss(self, obs, task):
        bsz, obs_dim = obs.shape

        # zero-out on-policy buffers
        self.obs_buf = torch.zeros(
            (self.horizon, bsz, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )

        rew_acc = torch.zeros((bsz, 1), dtype=torch.float32, device=self.device)

        # update and normalize obs
        if self.obs_rms:
            obs = self.obs_rms.normalize(obs)

        z = self.wm.encode(obs, task)

        # Start short horizon rollout
        for i in range(self.horizon):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = z.clone()

            # act in environment
            if self.detach:
                actions = self.actor(z.detach())
            else:
                actions = self.actor(z)

            actions = torch.tanh(actions)
            # TODO really move tanh inside actor
            z, rew = self.wm.step(z, actions, task)

            rew_acc += self.gamma**i * self.wm.almost_two_hot_inv(rew)

            next_values = self.critic(z).min(dim=0).values.squeeze()

            # collect data for critic training
            with torch.no_grad():
                if self.wm.num_bins:
                    rew = self.wm.almost_two_hot_inv(rew).flatten()
                self.rew_buf[i] = rew.clone()
                if i < self.horizon - 1:
                    self.done_mask[i, :] = 0.0
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values.clone()

        actor_loss = rew_acc + self.gamma**self.horizon * next_values[..., None]
        assert actor_loss.shape == (bsz, 1)

        if self.ret_rms is not None:
            self.ret_rms.update(actor_loss)
            actor_loss /= torch.sqrt(self.ret_rms.var + 1e-5)
        else:
            actor_loss /= self.horizon

        actor_loss = -actor_loss.mean()

        return actor_loss

    def critic_val(self, z):
        return self.critic(z).min(dim=0).values.squeeze()

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            bsz = self.rew_buf.shape[1]
            Ai = torch.zeros(bsz, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(bsz, dtype=torch.float32, device=self.device)
            lam = torch.ones(bsz, dtype=torch.float32, device=self.device)
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

    def update(self, obs, act, rew, task, finetune_wm=False):

        L, bsz, obs_dim = obs.shape

        # train world model
        if finetune_wm:
            self.wm_optimizer.zero_grad()
            wm_loss, dyn_loss, rew_loss = self.compute_wm_loss(obs, act, rew, task)
            wm_loss.backward()
            wm_grad_norm = clip_grad_norm_(self.wm.parameters(), self.wm_grad_norm)
            self.wm_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()

        # NOTE not sure about dimensionality below
        actor_loss = self.compute_actor_loss(obs[0], task)
        actor_loss.backward()

        self.actor_grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
        self.actor_grad_norm_after_clip = clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_norm
        )

        # sanity check
        if torch.isnan(self.actor_grad_norm_before_clip):
            print_error("NaN gradient")
            raise ValueError

        self.actor_optimizer.step()

        # prepare dataset
        critic_batch_size = bsz * self.horizon // self.critic_batches
        with torch.no_grad():
            self.compute_target_values()
            dataset = CriticDataset(
                critic_batch_size,
                self.obs_buf,
                self.target_values,
            )

        # critic training!
        value_loss = 0.0
        for j in range(self.critic_iterations):
            total_critic_loss = 0.0
            batch_cnt = 0
            for i in range(len(dataset)):
                batch_sample = dataset[i]
                self.critic_optimizer.zero_grad()
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

            value_loss += total_critic_loss / batch_cnt

        value_loss /= self.critic_iterations

        ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()

        metrics = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "actor_grad_norm": self.actor_grad_norm_before_clip.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }
        if finetune_wm:
            metrics["wm_loss"] = wm_loss
            metrics["dynamics_loss"] = dyn_loss
            metrics["reward_loss"] = rew_loss
            metrics["wm_grad_norm"] = wm_grad_norm
        metrics = filter_dict(metrics)
        return metrics

    def save(self, filename, log_dir):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "world_model": self.wm.state_dict(),
                "obs_rms": self.obs_rms,
                "rew_rms": self.rew_rms,
                "ret_rms": self.ret_rms,
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "world_model_opt": self.wm_optimizer.state_dict(),
            },
            os.path.join(log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.to(self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic.to(self.device)
        self.wm.load_state_dict(checkpoint["world_model"])
        self.wm.to(self.device)
        self.obs_rms = (
            checkpoint["obs_rms"].to(self.device)
            if checkpoint["obs_rms"] is not None
            else None
        )
        self.rew_rms = (
            checkpoint["rew_rms"].to(self.device)
            if checkpoint["rew_rms"] is not None
            else None
        )
        self.ret_rms = (
            checkpoint["ret_rms"].to(self.device)
            if checkpoint["ret_rms"] is not None
            else None
        )

        # need to also load last learning rates as they will be used to continue training
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.actor_lr = checkpoint["actor_opt"]["param_groups"][0]["lr"]
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.critic_lr = checkpoint["critic_opt"]["param_groups"][0]["lr"]
        self.wm_optimizer.load_state_dict(checkpoint["world_model_opt"])
        self.model_lr = checkpoint["world_model_opt"]["param_groups"][0]["lr"]

    def load_wm(self, path):
        print("Loading world model from", path)
        checkpoint = torch.load(path)
        checkpoint = checkpoint["model"]
        new_odict = OrderedDict()
        for key, value in checkpoint.items():
            if "_pi" in key:
                pass
            elif "_Qs" in key:
                pass
            else:
                if "_encoder" in key:
                    key = key.replace("state.", "")
                new_odict[key] = value

        self.wm.load_state_dict(new_odict)

    def compute_wm_loss(self, obs, act, rew, task):
        horizon, batch_size, obs_dim = obs.shape
        assert horizon == self.horizon + 1
        discount = (
            (self.gamma ** torch.arange(self.horizon))
            .view((self.horizon, 1, 1))
            .to(self.device)
        )

        # Compute targets
        with torch.no_grad():
            next_z = self.wm.encode(obs[1:], task)

        # Latent rollout
        zs = torch.empty(
            self.horizon + 1,
            batch_size,
            self.latent_dim,
            device=self.device,
        )

        z = self.wm.encode(obs[0], task)
        zs[0] = z

        dynamics_loss = 0.0
        for t in range(self.horizon):
            z = self.wm.next(z, act[t], task)
            dynamics_loss += F.mse_loss(z, next_z[t]) * self.gamma**t
            zs[t + 1] = z

        _zs = zs[:-1]
        rew_hat = self.wm.reward(_zs, act, task)
        reward_loss = (rew_hat - rew) ** 2 * discount
        reward_loss = reward_loss.mean()

        total_loss = dynamics_loss + reward_loss
        total_loss /= self.horizon
        return (
            total_loss,
            dynamics_loss / self.horizon,
            reward_loss.item() / self.horizon,
        )

    def act(self, obs, t0=False, deterministic=False, task=None):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)[None]
        z = self.wm.encode(obs, task)
        if self.planning:
            a = self.plan(z, t0, deterministic, task)
        else:
            a = self.actor(z, deterministic)
        return torch.tanh(a).cpu().detach().flatten()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.horizon):
            reward = self.wm.two_hot_inv(self.wm.reward(z, actions[t], task))
            z = self.wm.next(z, actions[t], task)
            G += discount * reward
            discount *= self.gamma
        return G + discount * self.critic_val(z)[..., None]

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                z (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        if self.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.horizon,
                self.num_pi_trajs,
                self.act_dim,
                device=self.device,
            )
            _z = z.repeat(self.num_pi_trajs, 1)
            for t in range(self.horizon - 1):
                pi_actions[t] = self.actor(_z)
                _z = self.wm.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.actor(_z)

        # Initialize state and parameters
        z = z.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.act_dim, device=self.device)
        std = self.max_std * torch.ones(self.horizon, self.act_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.horizon,
            self.num_samples,
            self.act_dim,
            device=self.device,
        )
        if self.num_pi_trajs > 0:
            actions[:, : self.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.iterations):

            # Sample actions
            actions[:, self.num_pi_trajs :] = (
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    self.horizon,
                    self.num_samples - self.num_pi_trajs,
                    self.act_dim,
                    device=std.device,
                )
            ).clamp(-1, 1)
            if self.wm.multitask:
                actions = actions * self.wm._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.temperature * (elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1
                )
                / (score.sum(0) + 1e-9)
            ).clamp_(self.min_std, self.max_std)
            if self.wm.multitask:
                mean = mean * self.wm._action_masks[task]
                std = std * self.wm._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        return a.clamp_(-1, 1)
