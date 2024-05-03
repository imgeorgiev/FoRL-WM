import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
from typing import List, Optional
from hydra.utils import instantiate


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

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        units: List[int],
        activation_cfg: dict = {"_target_": "torch.nn.Mish", "inplace": True},
        activation_cfg_last: Optional[dict] = None,
        spectral: bool = False,
    ):
        super(MLP, self).__init__()
        self.layer_dims = [input_dim] + units + [output_dim]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
                modules.append(instantiate(activation_cfg))

        if spectral:
            modules[-1] = spectral_norm(modules[-1])

        if activation_cfg_last:
            modules.append(activation_cfg_last)

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
