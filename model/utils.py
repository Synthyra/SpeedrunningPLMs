import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))
    

def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        corrected_dim = correction_fn(config.expansion_ratio, config.hidden_size)
        self.up = Linear(config.hidden_size, corrected_dim)
        self.down = Linear(corrected_dim, config.hidden_size)
        self.down.weight.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.relu(self.up(x)).square())


class BottleneckMLP(nn.Module):
    """MLP block used when sequence is a vector (length 1) in Conv1D UNet.
    Replaces transformer blocks at depths where sequence length = 1.
    Takes hidden_size directly instead of config to support variable sizes per layer.
    """
    def __init__(self, hidden_size: int, expansion_ratio: float, base_hidden_size: int = None):
        super().__init__()
        corrected_dim = correction_fn(expansion_ratio, hidden_size)
        self.up = Linear(hidden_size, corrected_dim)
        self.down = Linear(corrected_dim, hidden_size)
        self.down.weight.data.zero_()
        self.relu = nn.ReLU()
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        
        # Projection layer for x0 if hidden sizes differ (for Conv1D UNet)
        if base_hidden_size is not None and base_hidden_size != hidden_size:
            self.x0_projection = Linear(base_hidden_size, hidden_size)
        else:
            self.x0_projection = None
    
    def forward(
            self,
            x: torch.Tensor,
            x0: torch.Tensor = None,
            **kwargs,
        ) -> torch.Tensor:
        # Apply residual mixing with x0 if provided (for UNet skip connections)
        if x0 is not None:
            if self.x0_projection is not None:
                x0 = self.x0_projection(x0)
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        # Two-layer MLP with squared ReLU
        out = self.down(self.relu(self.up(norm(x))).square())
        return x + out
