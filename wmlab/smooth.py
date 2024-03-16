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
    actor2 = ActorStochasticMLP(
        env.num_obs, env.num_acts, [400, 200, 100], torch.nn.Mish
    ).to(device)
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoRL/scripts/outputs/2024-03-14/15-28-46/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])
    actor2.load_state_dict(chkpt["actor"])

    from tdmpc2 import TDMPC2
    from common.parser import parse_cfg
    from common.math import two_hot_inv

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)
    tdmpc.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/1/default/models/20000.pt"
    )
    # Alternatively load a different actor

    og_state_dict = actor.state_dict()

    deltas = torch.linspace(-1, 1, 50)
    # H = 32

    grad_idx = 1

    Hs = [4, 8, 16, 32]
    f, axs = plt.subplots(len(Hs), 2, figsize=(8, 3 * len(Hs)))

    for k, H in enumerate(Hs):
        print(f"H={H}")
        losses = torch.empty_like(deltas).to(device)
        grads = torch.empty_like(deltas).to(device)
        td_losses = torch.empty_like(deltas).to(device)
        td_grads = torch.empty_like(deltas).to(device)

        for i, d in enumerate(deltas):
            print(f"H={H} Step {i}/{len(deltas)}", end="\r")

            state_dict = deepcopy(og_state_dict)
            state_dict["mu_net.0.weight"][0][grad_idx] += d
            actor.load_state_dict(state_dict)
            actor2.load_state_dict(state_dict)

            obs = env.reset()
            obs = env.initialize_trajectory()
            z = tdmpc.model.encode(obs, None)
            reward = torch.tensor([0.0]).to(device)
            td_reward = torch.tensor([0.0]).to(device)

            for _ in range(H):
                # vanilla NOTE: detaching gradients is not correct here but necessary for fair comparison
                act = actor(obs.clone().detach(), deterministic=True)
                act2 = actor2(obs.clone().detach(), deterministic=True)
                assert torch.all(act == act2)

                obs, rew, done, info = env.step(act)
                reward += rew

                rew = two_hot_inv(tdmpc.model.reward(z, act2, None), cfg)[0]
                z = tdmpc.model.next(z, act2, None)
                td_reward += rew

            losses[i] = reward.clone().detach()
            td_losses[i] = td_reward.clone().detach()
            reward.backward()
            td_reward.backward()
            grads[i] = actor.mu_net[0].weight.grad[0][grad_idx]
            td_grads[i] = actor2.mu_net[0].weight.grad[0][grad_idx]
            actor.zero_grad()
            actor2.zero_grad()
            tdmpc.model.zero_grad()

        losses = losses.detach().cpu()
        td_losses = td_losses.detach().cpu()
        grads = grads.detach().cpu()
        td_grads = td_grads.detach().cpu()
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        axs[k][0].plot(deltas, losses, label=f"H={H}")
        axs[k][0].plot(deltas, td_losses, label=f"TD H={H}")
        axs[k][0].set_xlabel(r"$\Delta \theta$")
        axs[k][0].set_ylabel(r"$J(\theta)$")
        axs[k][0].legend()
        axs[k][1].plot(deltas, grads, label=f"H={H}")
        axs[k][1].plot(deltas, td_grads, label=f"TD H={H}")
        axs[k][1].set_xlabel(r"$\Delta \theta$")
        axs[k][1].set_ylabel(r"$\nabla J(\theta)$")
    plt.tight_layout()
    plt.savefig(f"sensitivity.pdf")


if __name__ == "__main__":
    main()
