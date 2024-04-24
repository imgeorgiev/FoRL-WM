import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from forl.models.mlp import mlp, SimNorm
from torch.optim import Adam
import torch.nn as nn
from forl.models.actor import ActorDeterministicMLP, ActorStochasticMLP
from forl.utils import torch_utils as tu

from dflex.envs.double_pendulum import DoublePendulumEnv
from dflex.envs.cartpole_swing_up import CartPoleSwingUpEnv
from dflex.envs.hopper import HopperEnv
from forl.utils.common import seeding
import hydra
from utils import dflex_jacobian
from torch.autograd.functional import jacobian


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from IPython.core import ultratb
import sys

from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.math import two_hot_inv

# For debugging
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Neutral", call_pdb=True
)


@hydra.main(config_name="config", config_path=".")
def main(cfg: dict):
    # We need:
    # 1) Actor gradients from simulation
    # 2) Actor gradients from model
    #   2.1) We need data from the simulation
    #   2.2) We need to train the model
    # 3) Compute gradients, compare their
    #   - gradient norms
    #   - eigenvalues
    #   - variance
    #   - Signal to noise ratio
    device = "cuda"
    seeding(0, True)
    env = DoublePendulumEnv(
        render=False,
        num_envs=1,
        episode_length=240,
        no_grad=False,
        stochastic_init=False,
    )

    actor = ActorStochasticMLP(
        env.num_obs, env.num_acts, [400, 200, 100], torch.nn.Mish
    ).to(device)
    # cartpole
    # chkpt = torch.load(
    #     "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/scripts/outputs/2024-04-21/19-37-34/logs/best_policy.pt",
    #     map_location=device,
    # )
    # doublependulum
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/scripts/outputs/2024-04-23/16-14-35/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])
    og_state_dict = actor.state_dict()
    obs_rms = chkpt["obs_rms"].to(device)
    opt = Adam(actor.parameters(), lr=1e-3)
    H = 100

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)

    # pre-trained TDMPC models; H: model_path
    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/000/pretrain_baseline/models/final.pt",
        # 8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/1/pretrain_baseline/models/final.pt",
        # 16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/2/pretrain_baseline/models/final.pt",
        # 32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/3/pretrain_baseline/models/final.pt",
    }

    # save actions from first rollout to account for the fact that the model is stochastic
    actions = torch.empty((H, cfg.action_dim)).to(device)

    f, ax = plt.subplots(2, 2, figsize=(8, 6))
    # prediction error, per-step dynamics jacobian norm, per-step dynamics jacobian max eigenvalue, whole policy gradient norm
    ax = ax.flatten()

    # now also compute jacobians
    print("Computing jacobians for dflex")
    jac_norms = torch.empty((H,)).to(device)
    jac_eig = torch.empty((H, 4)).to(device)
    losses = torch.empty((H,)).to(device)

    torch.manual_seed(0)
    obs = env.reset()
    obs = env.initialize_trajectory()
    # rewards = torch.tensor([0.0]).to(device)
    for h in tqdm(range(H)):
        # opt.zero_grad()
        obs = obs_rms.normalize(obs)
        act = actor(obs, deterministic=True)
        actions[h] = act.detach().clone()
        jac = dflex_jacobian(env, obs, act)
        jac = jac[..., :4, :4]  # need square matrix # TODO better code
        jac = torch.nan_to_num(jac)
        jac_norms[h] = jac.norm()
        # breakpoint()
        jac_eig[h] = torch.real(torch.linalg.eigvals(jac))
        obs, rew, done, info = env.step(act)
        losses[h] = rew + losses[h - 1] if h > 0 else rew
        # rewards += rew
        # rewards.sum().backward(retain_graph=True)
        # policy_norms[h] = tu.grad_norm(actor.parameters())

    print("Computing policy grad norms for dflex")
    policy_norms = torch.zeros((H,)).to(device)
    for h in tqdm(range(1, H + 1)):
        obs = env.reset(grads=True)
        opt.zero_grad()
        torch.manual_seed(0)  # guarantee same policy actions
        rew = torch.zeros(env.num_envs).to(device)
        for t in range(h):
            obs = obs_rms.normalize(obs)
            a = actor(obs.detach())
            obs, reward, done, info = env.step(a)
            rew += reward

        rew.mean().backward()
        policy_norms[h - 1] = tu.grad_norm(actor.parameters())

    ax[1].plot(jac_norms.detach().cpu(), label="GT")
    ax[2].plot(jac_eig.max(dim=1).values.detach().cpu(), label="GT")
    ax[3].plot(policy_norms.detach().cpu(), label="GT")

    print("Computing jacobians for tdmpc")
    jj = 0
    for hh, filepath in models.items():
        print(f"Loading TDMPC model with H={hh}")
        tdmpc.load(filepath)

        # now compute just jacobians since those don't depend on deltas
        print("Computing jacobians for TDMPC")
        jac_norms = torch.empty((H,)).to(device)
        jac_eig = torch.empty((H, cfg.latent_dim)).to(device)
        td_losses = torch.empty((H,)).to(device)
        obs = env.reset(grads=True)
        obs = env.initialize_trajectory()
        z = tdmpc.model.encode(obs, None)
        for h in tqdm(range(H)):
            # breakpoint()
            act = actions[h].unsqueeze(0)
            rew = two_hot_inv(tdmpc.model.reward(z, act, None), cfg)[0]
            # print(rew)
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
        policy_norms = torch.zeros((H,)).to(device)
        for h in tqdm(range(1, H + 1)):
            obs = env.reset(grads=True)
            z = tdmpc.model.encode(obs, None)
            tdmpc.pi_optim.zero_grad()
            # opt.zero_grad()
            # actor.load_state_dict(og_state_dict)
            torch.manual_seed(0)  # guarantee same policy actions
            rew = torch.zeros(env.num_envs).to(device)
            for t in range(h):
                # obs = obs_rms.normalize(obs)
                # a = actor(obs.detach())
                a = tdmpc.model.pi(z, None)[1]
                r = two_hot_inv(tdmpc.model.reward(z, a, None), cfg)[0]
                rew += r
                # obs, reward, done, info = env.step(a)
                z = tdmpc.model.next(z, a, None)

            rew.mean().backward()
            policy_norms[h - 1] = tu.grad_norm(tdmpc.model._pi.parameters())

        # breakpoint()
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
