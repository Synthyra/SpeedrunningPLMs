"""Linear layers and free parameters with Xavier-normal weights, and layer normalization without a bias.

These are the rules the annotation transformer's checkpoints were built by, and
`foundry.layers.attention` and `foundry.layers.feedforward` build their layers with them. A linear
layer can keep `nn.Linear`'s default initialization instead, and a free parameter its standard
normal values, as DSM's and synth's translators' layers do.
"""

from __future__ import annotations

import torch

from functools import partial
from typing import Any
from torch import nn


__all__ = ["LayerNorm", "Linear", "LinearLayer", "ParameterLayer"]


def LinearLayer(
    input_size: int,
    output_size: int,
    bias: bool = False,
    *,
    spectral_norm: bool = False,
    xavier_init: bool = True,
) -> nn.Linear:
    """Build a linear layer with a Xavier-normal weight and, with `bias`, a zero bias.

    `xavier_init=False` keeps `nn.Linear`'s default initialization of both, and draws no random
    numbers beyond it. With `spectral_norm`, `torch.nn.utils.spectral_norm` divides the weight by
    its largest singular value at each forward pass. The layer then holds the weight as
    `weight_orig`, beside the power iteration's `weight_u` and `weight_v`.
    """
    layer = nn.Linear(input_size, output_size, bias=bias)  # weight: (d_out, d_in)
    if xavier_init:
        nn.init.xavier_normal_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)

    if spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


def ParameterLayer(
    size: tuple[int, ...], *, xavier_init: bool = True, **kwargs: Any
) -> nn.Parameter:
    """Build a free parameter of shape `size` with Xavier-normal values.

    `xavier_init=False` keeps the standard normal values `torch.randn` draws, and draws no random
    numbers beyond them. `kwargs` go to `torch.randn`, such as `dtype` and `device`.
    """
    parameter = nn.Parameter(torch.randn(size, **kwargs))  # shape: size
    if xavier_init:
        nn.init.xavier_normal_(parameter)
    return parameter


Linear = partial(LinearLayer, bias=False)
LayerNorm = partial(nn.LayerNorm, bias=False)
