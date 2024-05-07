import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from forl.models.mlp import mlp, SimNorm
from torch.optim import Adam
import torch.nn as nn

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

sns.set()
torch.manual_seed(0)

# gravity, height, width
g, h, w = -9.81, 0.0, 5


def f(x, v, th, a, t):
    ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
    y = v * torch.sin(th) * ty + g / 2 * ty**2
    out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
    out = torch.where((h > y) & (ty < t), w, out)
    return out


fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# simulation variables
samples = 1000
xx = torch.linspace(-torch.pi, torch.pi, samples)
x, v, a, t = 0, 10, 1, 2
yy = -f(x, v, xx, a, t)
std = 0.1  # noise for policy
N = 5000  # data samples
epochs = 100  # for optimization
batch_size = 56
lr = 2e-3

# train simply MLP
torch.manual_seed(0)
model0 = mlp(1, [32, 32], 1, last_layer="linear", last_layer_kwargs={})
opt = Adam(model0.parameters(), lr=lr)
steps = samples // batch_size
print("Training...")
model0.train()
with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
    for epoch in tepoch:
        epoch_loss = 0
        for step in range(steps):
            idx = torch.randint(0, samples, (batch_size,))
            _xx = xx[idx].unsqueeze(1)
            _yy = yy[idx].unsqueeze(1)
            pred = model0(_xx)
            loss = torch.mean((pred - _yy) ** 2)
            model0.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= steps
        tepoch.set_postfix(loss=epoch_loss)


# train TDMPC model
torch.manual_seed(0)
model = mlp(
    1,
    [32],
    32,
    last_layer="normedlinear",
    last_layer_kwargs={"act": SimNorm(8)},
)
decoder = mlp(32, [], 1, last_layer="linear", last_layer_kwargs={})
opt = Adam([{"params": model.parameters()}, {"params": decoder.parameters()}], lr=lr)
print("Training...")
model.train()
with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
    for epoch in tepoch:
        epoch_loss = 0
        for step in range(steps):
            idx = torch.randint(0, samples, (batch_size,))
            _xx = xx[idx].unsqueeze(1)
            _yy = yy[idx].unsqueeze(1)
            pred = decoder(model(_xx))
            loss = torch.mean((pred - _yy) ** 2)
            model.zero_grad()
            # decoder.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= steps
        tepoch.set_postfix(loss=epoch_loss)

model1 = lambda x: decoder(model(x))

# Spectrum normalized MLP
torch.manual_seed(0)
model2 = mlp(
    1,
    [32, 32],
    1,
    last_layer="spectralnormlinear",
    last_layer_kwargs={"layer_norm": False},
)
opt = Adam(model2.parameters(), lr=lr)
print("Training...")
model2.train()
with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
    for epoch in tepoch:
        epoch_loss = 0
        for step in range(steps):
            idx = torch.randint(0, samples, (batch_size,))
            _xx = xx[idx].unsqueeze(1)
            _yy = yy[idx].unsqueeze(1)
            pred = model2(_xx)
            loss = torch.mean((pred - _yy) ** 2)
            model2.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= steps
        tepoch.set_postfix(loss=epoch_loss)


print("Plotting the problem landscape")
ax.plot(xx, -f(x, v, xx, a, t), label=r"$J(\theta)$")
models = {0: "MLP", 1: "SimNorm MLP", 2: "Spectral MLP"}
for i, m in enumerate([model0, model1, model2]):
    est = m(xx.unsqueeze(1)).detach().numpy()
    ax.plot(xx, est, label=models[i])

ax.set_xlabel(r"$\theta$")
ax.legend()
plt.tight_layout()
plt.savefig("ball_wall.pdf", bbox_inches="tight")
plt.show()
