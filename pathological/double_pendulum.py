import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from forl.models.mlp import mlp, SimNorm
from torch.optim import Adam
import torch.nn as nn
from forl.models.actor import ActorDeterministicMLP
from forl.utils import torch_utils as tu

from dflex.envs.double_pendulum import DoublePendulumEnv
from dflex.envs.cartpole_swing_up import CartPoleSwingUpEnv
from dflex.envs.hopper import HopperEnv

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Neutral", call_pdb=True
)


if __name__ == "__main__":
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
    torch.manual_seed(0)
    env = CartPoleSwingUpEnv(
        render=False,
        num_envs=1,
        episode_length=240,
        no_grad=False,
        stochastic_init=False,
    )

    actor = ActorDeterministicMLP(
        env.num_obs, env.num_actions, [32, 32], activation_class=nn.ELU
    ).to(device)
    opt = Adam(actor.parameters(), lr=1e-3)
    H = 50
    grad_norms = []
    for h in tqdm(range(1, H + 1)):
        obs = env.reset(grads=True)
        opt.zero_grad()
        rew = torch.zeros(env.num_envs).to(device)
        for t in range(h):
            a = actor(obs)
            obs, reward, done, info = env.step(a)
            rew += reward

        rew.mean().backward()
        norm = tu.grad_norm(actor.parameters())
        grad_norms.append(norm)
        # opt.step()

    grad_norms = torch.Tensor(grad_norms).detach().to("cpu")
    plt.plot(torch.arange(H) + 1, grad_norms)
    plt.savefig("pendulum_grad_norms.pdf")
