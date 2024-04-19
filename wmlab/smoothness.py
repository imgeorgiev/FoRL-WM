from dflex.envs import AntEnv
from forl.models.actor import ActorDeterministicMLP, ActorStochasticMLP
from forl.utils.common import seeding
import torch
import sys
from IPython.core import ultratb
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import hydra

from utils import dflex_jacobian
from torch.autograd.functional import jacobian

"""
This script loads multiple TDMPC world model trained with different H, an SHAC actor and throws them in simulation.
It produces a figure showing
- (1) actor loss landscapes for different horizons
- (2) prediction error of TDMPC models trained for differet horizons
- (3) the norm of the jacobian of ground-truth and TDMPC models
- (4) the max eigenvalues of the jacobian of ground-truth and TDMPC models

Note: output figure is saved in the hydra location
"""


sns.set()
colors = sns.color_palette()

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


@hydra.main(config_name="config", config_path=".")
def main(cfg: dict):

    device = "cuda"

    seeding(42, True)
    env = AntEnv(num_envs=1, no_grad=False, early_termination=False, device=device)
    actor = ActorStochasticMLP(
        env.num_obs, env.num_acts, [400, 200, 100], torch.nn.Mish
    ).to(device)
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoRL/scripts/outputs/2024-03-14/15-28-46/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])
    obs_rms = chkpt["obs_rms"].to(device)

    from tdmpc2 import TDMPC2
    from common.parser import parse_cfg
    from common.math import two_hot_inv

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)

    # H: model_path
    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/1/default/models/20000.pt",
        8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/5/default/models/20000.pt",
        16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/6/default/models/20000.pt",
        32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/7/default/models/20000.pt",
    }

    og_state_dict = actor.state_dict()

    deltas = torch.linspace(-1, 1, 50)
    H = 32
    grad_idx = 1

    losses = torch.empty((H, len(deltas))).to(device)
    grads = torch.empty((H, len(deltas))).to(device)
    actions = torch.empty((H, len(deltas), cfg.action_dim)).to(device)

    f, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax = ax.flatten()

    for i, d in enumerate(deltas):
        print(f"Step {i}/{len(deltas)}", end="\r")
        state_dict = deepcopy(og_state_dict)
        state_dict["mu_net.0.weight"][0][grad_idx] += d
        actor.load_state_dict(state_dict)

        obs = env.reset()
        obs = env.initialize_trajectory()
        obs = obs_rms.normalize(obs)

        for h in range(H):
            obs = obs_rms.normalize(obs)
            act = actor(obs.clone().detach(), deterministic=True)
            actions[h, i] = act
            obs, rew, done, info = env.step(act)
            losses[h, i] = rew + losses[h - 1, i] if h > 0 else rew

    # now also compute jacobians
    print("Computing jacobians")
    jac_norms = torch.empty((H,)).to(device)
    jac_eig = torch.empty((H, actor.obs_dim)).to(device)

    obs = env.reset()
    obs = env.initialize_trajectory()
    for h in range(H):
        print(f"Step {h}/{H}", end="\r")
        act = actions[h, 50 // 2]
        jac = dflex_jacobian(env, obs, act.unsqueeze(0))
        jac = jac[..., : actor.obs_dim]
        jac = torch.nan_to_num(jac)
        jac_norms[h] = jac.norm()
        jac_eig[h] = torch.real(torch.linalg.eigvals(jac))
        obs, rew, done, info = env.step(act)

    ax[0].plot(deltas.detach().cpu(), losses[-1].detach().cpu(), label="GT")
    ax[2].plot(jac_norms.detach().cpu(), label="GT")
    ax[3].plot(jac_eig.max(dim=1).values.detach().cpu(), c=colors[0], label="GT")

    jj = 0
    for hh, filepath in models.items():
        print(f"Loading TDMPC model with H={hh}")
        tdmpc.load(filepath)
        td_losses = torch.empty((H, len(deltas))).to(device)
        td_grads = torch.empty((H, len(deltas))).to(device)
        for i, d in enumerate(deltas):
            print(f"Step {i}/{len(deltas)}", end="\r")
            obs = env.reset()
            obs = env.initialize_trajectory()
            z = tdmpc.model.encode(obs, None)
            for h in range(H):
                act = actions[h, i].unsqueeze(0)
                rew = two_hot_inv(tdmpc.model.reward(z, act, None), cfg)[0]
                z = tdmpc.model.next(z, act, None)
                td_losses[h, i] = rew + td_losses[h - 1, i] if h > 0 else rew

        # now compute just jacobians since those don't depend on deltas
        print("Computing jacobians")
        jac_norms = torch.empty((H,)).to(device)
        jac_eig = torch.empty((H, cfg.latent_dim)).to(device)
        obs = env.reset()
        obs = env.initialize_trajectory()
        z = tdmpc.model.encode(obs, None)
        for h in range(H):
            print(f"Step {h}/{H}", end="\r")
            act = actions[h, 50 // 2].unsqueeze(0)
            z = tdmpc.model.next(z, act, None)
            jac = jacobian(tdmpc.model.next, (z, act, torch.zeros((1,))))
            jac = jac[0].squeeze()
            jac_norms[h] = jac.norm()
            jac_eig[h] = torch.real(torch.linalg.eigvals(jac))

        ax[0].plot(
            deltas.detach().cpu(), td_losses[-1].detach().cpu(), label=f"TD H={hh}"
        )
        error = torch.norm(td_losses - losses, dim=1)
        ax[1].plot(error.detach().cpu(), label=f"TD H={hh}")
        ax[2].plot(jac_norms.detach().cpu(), label=f"TD H={hh}")
        ax[3].plot(
            jac_eig.max(dim=1).values.detach().cpu(),
            c=colors[jj + 1],
            label=f"max H={hh}",
        )
        jj += 1

    ax[0].set_title("Opt. landscape for H=32")
    ax[0].set_xlabel(r"$\Delta \theta$")
    ax[0].set_ylabel(r"$J(\theta)$")
    ax[0].legend()
    ax[1].set_xlabel(r"$H$")
    ax[1].set_ylabel(r"Model prediction error")
    ax[1].legend()
    ax[2].set_xlabel(r"$H$")
    ax[2].set_ylabel(r"$\| \nabla f \|$")
    ax[2].set_yscale("log")
    ax[3].set_xlabel(r"$H$")
    ax[3].set_ylabel(r"Max Eigenval of $\| \nabla f \|$")
    ax[3].set_yscale("log")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("sensitivity.pdf")


if __name__ == "__main__":
    main()
