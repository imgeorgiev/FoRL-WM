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
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoRL/scripts/outputs/2024-03-14/15-28-46/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])

    from tdmpc2 import TDMPC2
    from common.parser import parse_cfg
    from common.math import two_hot_inv

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)
    # tdmpc4.load(
    #     "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/1/default/models/100000.pt"
    # )

    # tdmpc8 = TDMPC2(cfg)
    # tdmpc8.load(
    #     "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/2/default/models/100000.pt"
    # )

    # tdmpc16 = TDMPC2(cfg)
    # tdmpc16.load(
    #     "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/3/default/models/40000.pt"
    # )

    # tdmpc32 = TDMPC2(cfg)
    # tdmpc32.load(
    #     "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/4/default/models/20000.pt"
    # )

    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/1/default/models/20000.pt",
        8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/2/default/models/20000.pt",
        16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/3/default/models/20000.pt",
        32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/4/default/models/20000.pt",
    }

    # Alternatively load a different actor

    og_state_dict = actor.state_dict()

    deltas = torch.linspace(-1, 1, 50)
    H = 32

    grad_idx = 1

    losses = torch.empty((H, len(deltas))).to(device)
    grads = torch.empty((H, len(deltas))).to(device)
    actions = torch.empty((H, len(deltas), cfg.action_dim)).to(device)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    for i, d in enumerate(deltas):
        print(f"Step {i}/{len(deltas)}", end="\r")
        state_dict = deepcopy(og_state_dict)
        state_dict["mu_net.0.weight"][0][grad_idx] += d
        actor.load_state_dict(state_dict)

        obs = env.reset()
        obs = env.initialize_trajectory()

        for h in range(H):
            act = actor(obs.clone().detach(), deterministic=True)
            actions[h, i] = act
            obs, rew, done, info = env.step(act)
            losses[h, i] = rew + losses[h - 1, i] if h > 0 else rew

    ax1.plot(deltas.detach().cpu(), losses[-1].detach().cpu(), label="GT")

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

        ax1.plot(
            deltas.detach().cpu(), td_losses[-1].detach().cpu(), label=f"TD H={hh}"
        )
        error = torch.norm(td_losses - losses, dim=1)
        ax2.plot(error.detach().cpu(), label=f"TD H={hh}")

    ax1.set_title("H=32")
    ax1.set_xlabel(r"$\Delta \theta$")
    ax1.set_ylabel(r"$J(\theta)$")
    ax1.legend()
    ax2.set_xlabel(r"$H$")
    ax2.set_ylabel(r"Error")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("sensitivity_h.pdf")


if __name__ == "__main__":
    main()
