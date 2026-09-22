import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Protocol


class MLPConfig(Protocol):
    hidden_size: int
    expansion_ratio: float


def norm(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d), with any leading dimensions.
    return F.rms_norm(x, (x.size(-1),))  # (..., d)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in); weight: (d_out, d_in).
        return F.linear(x, self.weight.to(x.dtype))  # (..., d_out)
    

def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        corrected_dim = correction_fn(config.expansion_ratio, config.hidden_size)  # d_mlp
        self.up = Linear(config.hidden_size, corrected_dim)
        self.down = Linear(corrected_dim, config.hidden_size)
        self.down.weight.data.zero_()  # (d, d_mlp); start with a zero MLP residual.
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d); the intermediate projection has width d_mlp.
        return self.down(self.relu(self.up(x)).square())  # (..., d)


class BottleneckMLP(nn.Module):
    """Residual MLP for a UNet bottleneck with sequence length one."""

    def __init__(
        self,
        hidden_size: int,
        expansion_ratio: float,
        base_hidden_size: Optional[int] = None,
        embedding_residual: bool = True,
    ) -> None:
        super().__init__()
        corrected_dim = correction_fn(expansion_ratio, hidden_size)  # d_mlp
        self.up = Linear(hidden_size, corrected_dim)
        self.down = Linear(corrected_dim, hidden_size)
        self.down.weight.data.zero_()  # (d, d_mlp)
        self.relu = nn.ReLU()
        self.embedding_residual = embedding_residual
        if self.embedding_residual:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))  # (2,)
        
        # Projection layer for x0 if hidden sizes differ (for Conv1D UNet)
        if self.embedding_residual and base_hidden_size is not None and base_hidden_size != hidden_size:
            self.x0_projection = Linear(base_hidden_size, hidden_size)
        else:
            self.x0_projection = None
    
    def forward(
            self,
            x: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            **kwargs: object,
        ) -> torch.Tensor:
        # x: (b, 1, d); x0: (b, 1, d_base) before optional projection.
        if self.embedding_residual and x0 is not None:
            if self.x0_projection is not None:
                x0 = self.x0_projection(x0)  # (b, 1, d)
            x = self.lambdas[0] * x + self.lambdas[1] * x0  # (b, 1, d)
        out = self.down(self.relu(self.up(norm(x))).square())  # (b, 1, d)
        return x + out  # (b, 1, d)
