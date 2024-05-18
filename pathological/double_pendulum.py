import hydra
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from forl.utils import torch_utils as tu

from dflex.envs.double_pendulum import DoublePendulumEnv
from forl.utils.common import seeding
from utils import dflex_jacobian
from torch.autograd.functional import jacobian

# TDMPC libraries
from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.math import two_hot_inv
from common.init import weight_init

sns.set()

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Neutral", call_pdb=True
)


@hydra.main(config_name="config", config_path=".")
def main(cfg: dict):
    env = DoublePendulumEnv(
        render=False,
        num_envs=1,
        episode_length=240,
        no_grad=False,
        stochastic_init=False,
        device=cfg.device,
    )

    H = 50

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)

    # pre-trained TDMPC models; H: model_path
    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/0/pretrain_baseline/models/final.pt",
        # 8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/1/pretrain_baseline/models/final.pt",
        # 16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/2/pretrain_baseline/models/final.pt",
        # 32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/3/pretrain_baseline/models/final.pt",
    }
    tdmpc.load(models[4])
    seeding(0)
    tdmpc.model._pi.apply(weight_init)  # reset actor weight

    # save actions from first rollout to account for the fact that the model is stochastic
    actions = torch.empty((H, cfg.action_dim)).to(cfg.device)

    f, ax = plt.subplots(2, 2, figsize=(8, 6))
    # prediction error, per-step dynamics jacobian norm, per-step dynamics jacobian max eigenvalue, whole policy gradient norm
    ax = ax.flatten()

    # now also compute jacobians
    print("Computing jacobians for dflex")
    jac_norms = torch.empty((H,)).to(cfg.device)
    jac_eig = torch.empty((H, 4)).to(cfg.device)
    losses = torch.empty((H,)).to(cfg.device)

    seeding(0)
    obs = env.reset(grads=True)
    for h in tqdm(range(H)):
        z = tdmpc.model.encode(obs, None)
        a = tdmpc.model.pi(z.detach(), None)[1]
        # obs = obs_rms.normalize(obs)
        # act = actor(obs, deterministic=True)
        actions[h] = a.detach().clone()
        jac = dflex_jacobian(env, obs, a)
        jac = jac[..., :4, :4]  # need square matrix # TODO better code
        jac = torch.nan_to_num(jac)
        jac_norms[h] = jac.norm()
        jac_eig[h] = torch.real(torch.linalg.eigvals(jac))
        obs, rew, done, info = env.step(a)
        losses[h] = rew + losses[h - 1] if h > 0 else rew

    print("Computing policy grad norms for dflex")
    policy_norms = torch.zeros((H,)).to(cfg.device)
    for h in tqdm(range(1, H + 1)):
        obs = env.reset(grads=True)
        tdmpc.pi_optim.zero_grad()
        seeding(0)  # guarantee same policy actions
        rew = torch.zeros(env.num_envs).to(cfg.device)
        for t in range(h):
            # obs = obs_rms.normalize(obs)
            # a = actor(obs.detach())
            z = tdmpc.model.encode(obs, None)
            a = tdmpc.model.pi(z.detach(), None)[1]
            obs, reward, done, info = env.step(a)
            # do reward transformation to be comparable with tdmpc
            reward /= 21554.2305
            reward += 1.0
            rew += reward

        rew.mean().backward()
        policy_norms[h - 1] = tu.grad_norm(tdmpc.model._pi.parameters())

    ax[1].plot(jac_norms.detach().cpu(), label="GT")
    ax[2].plot(jac_eig.max(dim=1).values.detach().cpu(), label="GT")
    ax[3].plot(policy_norms.detach().cpu(), label="GT")

    print("Computing jacobians for tdmpc")
    jj = 0
    for hh, filepath in models.items():
        print(f"Loading TDMPC model with H={hh}")
        tdmpc.load(filepath)
        seeding(0)
        tdmpc.model._pi.apply(weight_init)  # reset actor weight

        # now compute just jacobians since those don't depend on deltas
        print("Computing jacobians for TDMPC")
        jac_norms = torch.empty((H,)).to(cfg.device)
        jac_eig = torch.empty((H, cfg.latent_dim)).to(cfg.device)
        td_losses = torch.empty((H,)).to(cfg.device)
        obs = env.reset(grads=True)
        obs = env.initialize_trajectory()
        z = tdmpc.model.encode(obs, None)
        for h in tqdm(range(H)):
            act = actions[h].unsqueeze(0)
            rew = two_hot_inv(tdmpc.model.reward(z, act, None), cfg)[0]
            z = tdmpc.model.next(z, act, None)
            jac = jacobian(tdmpc.model.next, (z, act, torch.zeros((1,))))
            jac = jac[0].squeeze()
            jac_norms[h] = jac.norm()
            jac_eig[h] = torch.real(torch.linalg.eigvals(jac))
            td_losses[h] = rew + td_losses[h - 1] if h > 0 else rew

        # error = torch.norm(td_losses - losses, dim=1)
        error = (td_losses - losses) ** 2
        ax[0].plot(error.detach().cpu(), label=f"TD H={hh}")
        ax[1].plot(jac_norms.detach().cpu(), label=f"TD H={hh}")
        ax[2].plot(
            jac_eig.max(dim=1).values.detach().cpu(),
            label=f"max H={hh}",
        )
        jj += 1

        print("Computing actor norms for TDMPC")
        policy_norms = torch.zeros((H,)).to(cfg.device)
        for h in tqdm(range(1, H + 1)):
            obs = env.reset(grads=True)
            z = tdmpc.model.encode(obs, None)
            tdmpc.pi_optim.zero_grad()
            seeding(0)  # guarantee same policy actions
            rew = torch.zeros(env.num_envs).to(cfg.device)
            for t in range(h):
                a = tdmpc.model.pi(z, None)[1]
                r = two_hot_inv(tdmpc.model.reward(z, a, None), cfg)[0]
                rew += r
                z = tdmpc.model.next(z, a, None)

            rew.mean().backward()
            policy_norms[h - 1] = tu.grad_norm(tdmpc.model._pi.parameters())

        ax[3].plot(policy_norms.detach().cpu(), label=f"TD H={hh}")

    print("Saving figure")
    ax[0].set_xlabel(r"$H$")
    ax[0].set_ylabel(r"Model prediction error")
    ax[0].legend()
    ax[1].set_xlabel(r"$H$")
    ax[1].set_ylabel(r"$\| \nabla f \|$")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[2].set_xlabel(r"$H$")
    ax[2].set_ylabel(r"Max Eigenval of $\nabla f$")
    ax[2].set_yscale("log")
    ax[3].set_xlabel(r"$H$")
    ax[3].set_ylabel(r"$\| J(\theta) \|$")
    ax[3].set_yscale("log")
    plt.tight_layout()
    plt.savefig("sensitivity.pdf")


if __name__ == "__main__":
    main()
