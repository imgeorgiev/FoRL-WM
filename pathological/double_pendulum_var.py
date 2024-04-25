import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dflex.envs.double_pendulum import DoublePendulumEnv
from forl.utils.common import seeding
import hydra


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from IPython.core import ultratb
import sys

from tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.math import two_hot_inv
from common.init import weight_init

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
        # start_state=[np.pi / 2, np.pi / 2, 0.0, 0.0],
        device=device,
    )
    H = 10
    samples = 10

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)

    # pre-trained TDMPC models; H: model_path
    models = {
        4: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/100/pretrain_baseline/models/final.pt",
        8: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/101/pretrain_baseline/models/final.pt",
        16: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/102/pretrain_baseline/models/final.pt",
        32: "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-doublependulum/103/pretrain_baseline/models/final.pt",
    }

    tdmpc.load(models[4])
    tdmpc.model._pi.apply(weight_init)  # reset actor weight
    tdmpc.model.eval()

    f, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.flatten()

    save_data = {}

    print("Computing policy variance for dflex")
    all_grads = []
    seeding(0)
    for h in tqdm(range(1, H + 1)):
        grads = []
        for _ in range(samples):
            obs = env.reset(grads=True)
            tdmpc.pi_optim.zero_grad()
            rew = torch.zeros(env.num_envs).to(device)
            for t in range(h):
                z = tdmpc.model.encode(obs, None)
                a = tdmpc.model.pi(z.detach(), None)[1]
                obs, reward, done, info = env.step(a)
                # do reward transformation to be comparable with tdmpc
                reward /= 21554.2305
                reward += 1.0
                rew += reward

            (-rew).mean().backward()
            grads.append(get_grads(tdmpc.model._pi))
        grads = torch.vstack(grads)
        all_grads.append(grads)

    # Plotting!
    grads = torch.stack(all_grads)  # H x samples x n_params
    save_data["dflex_grads"] = grads
    variance = grads.var(dim=1).mean(dim=1)
    variance_std = grads.var(dim=1).std(dim=1)
    policy_snr = (grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)).mean(dim=1)
    ax[0].plot(range(H), variance, label="GT")
    # ax[0].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)
    ax[1].plot(range(H), policy_snr, label="GT")

    print("Computing policy variance for tdmpc")
    seeding(0)
    all_grads = []
    for h in tqdm(range(1, H + 1)):
        grads = []
        for _ in range(samples):
            obs = env.reset(grads=True)
            z = tdmpc.model.encode(obs, None)
            tdmpc.pi_optim.zero_grad()
            rew = torch.zeros(env.num_envs).to(device)
            for _ in range(h):
                a = tdmpc.model.pi(z.detach(), None)[1]
                r = two_hot_inv(tdmpc.model.reward(z, a, None), cfg)[0]
                rew += r
                z = tdmpc.model.next(z, a, None)

            (-rew).mean().backward()
            grads.append(get_grads(tdmpc.model._pi))
        grads = torch.vstack(grads)
        all_grads.append(grads)

    # Plotting!
    grads = torch.stack(all_grads)  # H x samples x n_params
    save_data["tdmpc_grads"] = grads
    variance = grads.var(dim=1).mean(dim=1)
    variance_std = grads.var(dim=1).std(dim=1)
    policy_snr = (grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)).mean(dim=1)
    ax[0].plot(range(H), variance, label="TDMPC")
    # ax[0].fill_between(range(H), lower_bound, upper_bound, alpha=0.5)
    ax[1].plot(range(H), policy_snr, label="TDMPC")

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
    torch.save(save_data, "gradients.pt")


def get_grads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


if __name__ == "__main__":
    main()
