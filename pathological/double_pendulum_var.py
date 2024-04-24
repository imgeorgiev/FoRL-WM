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
import numpy as np


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


@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg: dict):
    device = "cpu"
    seeding(0, True)
    env = DoublePendulumEnv(
        render=False,
        num_envs=1,
        episode_length=240,
        no_grad=False,
        stochastic_init=False,
        start_state=[np.pi / 2, np.pi / 2, 0.0, 0.0],
        device=device,
    )

    actor = ActorStochasticMLP(
        env.num_obs, env.num_acts, [400, 200, 100], torch.nn.Mish
    ).to(device)
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/scripts/outputs/2024-04-23/16-14-35/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])
    og_state_dict = actor.state_dict()
    obs_rms = chkpt["obs_rms"].to(device)
    opt = Adam(actor.parameters(), lr=1e-3)
    H = 10

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)

    # pre-trained TDMPC models; H: model_path
    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/100/pretrain_baseline/models/final.pt",
        # 8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/1/pretrain_baseline/models/final.pt",
        # 16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/2/pretrain_baseline/models/final.pt",
        # 32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-cartpole/3/pretrain_baseline/models/final.pt",
    }

    tdmpc.load(models[4])

    # save actions from first rollout to account for the fact that the model is stochastic
    actions = torch.empty((H, cfg.action_dim)).to(device)

    f, ax = plt.subplots(1, 2, figsize=(8, 3))
    # prediction error, per-step dynamics jacobian norm, per-step dynamics jacobian max eigenvalue, whole policy gradient norm
    ax = ax.flatten()

    save_data = {}

    print("Computing policy variance for dflex")
    policy_variance = []
    policy_std = []
    policy_snr = []
    policy_snr_std = []
    samples = 10
    torch.manual_seed(0)
    for h in tqdm(range(1, H + 1)):
        grads = []
        for _ in range(samples):
            obs = env.reset(grads=True)
            opt.zero_grad()
            rew = torch.zeros(env.num_envs).to(device)
            for t in range(h):
                obs = obs_rms.normalize(obs)
                a = actor(obs.detach())
                obs, reward, done, info = env.step(a)
                # do reward transformation to be comparable with tdmpc
                reward /= 21554.2305
                reward += 1.0
                rew += reward

            (-rew).mean().backward()
            grads.append(get_grads(actor))
        grads = torch.vstack(grads)
        policy_variance.append(grads.var(dim=1).mean())
        policy_std.append(grads.var(dim=1).std())
        snr = grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)
        policy_snr.append(torch.mean(snr))
        policy_snr_std.append(torch.std(snr))

    policy_variance = torch.tensor(policy_variance).detach().numpy()
    policy_std = torch.tensor(policy_std).detach().numpy()
    save_data.update(
        {"dflex_variance": policy_variance, "dflex_variance_std": policy_std}
    )
    lower_bound = policy_variance - policy_std
    upper_bound = policy_variance + policy_std
    ax[0].plot(range(H), policy_variance, label="GT")
    # ax[0].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)

    policy_snr = torch.tensor(policy_snr).detach().numpy()
    policy_snr_std = torch.tensor(policy_snr_std).detach().numpy()
    save_data.update({"dflex_snr": policy_snr, "dflex_snr_std": policy_snr_std})
    lower_bound = policy_snr - policy_snr_std
    upper_bound = policy_snr + policy_snr_std
    ax[1].plot(range(H), policy_snr, label="GT")
    # ax[1].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)

    print("Computing policy variance for tdmpc")
    policy_variance = []
    policy_std = []
    policy_snr = []
    policy_snr_std = []

    torch.manual_seed(0)
    for h in tqdm(range(1, H + 1)):
        grads = []
        for _ in range(samples):
            obs = env.reset(grads=True)
            z = tdmpc.model.encode(obs, None)
            opt.zero_grad()
            rew = torch.zeros(env.num_envs).to(device)
            for _ in range(h):
                a = tdmpc.model.pi(z, None)[1]
                r = two_hot_inv(tdmpc.model.reward(z, a, None), cfg)[0]
                rew += r
                z = tdmpc.model.next(z, a, None)

            (-rew).mean().backward()
            grads.append(get_grads(tdmpc.model._pi))
        grads = torch.vstack(grads)
        policy_variance.append(grads.var(dim=1).mean())
        policy_std.append(grads.var(dim=1).std())
        snr = grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)
        policy_snr.append(torch.mean(snr))
        policy_snr_std.append(torch.std(snr))

    policy_variance = torch.tensor(policy_variance).detach().numpy()
    policy_std = torch.tensor(policy_std).detach().numpy()
    save_data.update(
        {"tdmpc_variance": policy_variance, "tdmpc_variance_std": policy_std}
    )
    lower_bound = policy_variance - policy_std
    upper_bound = policy_variance + policy_std
    ax[0].plot(range(H), policy_variance, label="TDMPC")
    # ax[0].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)

    policy_snr = torch.tensor(policy_snr).detach().numpy()
    policy_snr_std = torch.tensor(policy_snr_std).detach().numpy()
    save_data.update({"tdmpc_snr": policy_snr, "tdmpc_snr_std": policy_snr_std})
    lower_bound = policy_snr - policy_snr_std
    upper_bound = policy_snr + policy_snr_std
    ax[1].plot(range(H), policy_snr, label="TDMPC")
    # ax[1].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)

    print("Saving figure")
    ax[0].set_xlabel(r"$H$")
    ax[0].set_ylabel(r"Policy variance")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].set_xlabel(r"$H$")
    ax[1].set_ylabel(r"Policy ESNR")
    ax[1].set_yscale("log")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("sensitivity.pdf")

    # save data
    np.save("sensitivity.npy", save_data)


def get_grads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


if __name__ == "__main__":
    main()
