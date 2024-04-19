import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self):
        super().__init__()
        self.dim = 8  # cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


def mlp(
    in_dim,
    mlp_dims,
    out_dim,
    layer=nn.Linear,
    act=nn.Mish,
    norm=False,
    spectral=False,
    last=False,
):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(layer(dims[i], dims[i + 1]))
        # if spectral:
        #     mlp[-1] = spectral_norm(mlp[-1])
        if norm:
            mlp.append(nn.LayerNorm(dims[i + 1]))
        mlp.append(act())
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=last())
        if last
        else nn.Linear(dims[-2], dims[-1])
    )
    if spectral:
        mlp[-1] = spectral_norm(mlp[-1])
    return nn.Sequential(*mlp)
