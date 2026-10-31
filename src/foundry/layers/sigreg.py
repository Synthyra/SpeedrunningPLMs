"""SIGReg: the sketched isotropic Gaussian regularizer of LeJEPA and LeWorldModel.

Embeddings are projected onto random unit directions, and each one-dimensional projection is
scored by the Epps-Pulley statistic against a standard normal: `n` times the integral, weighted by
the normal's characteristic function, of the squared distance between the empirical
characteristic function and `exp(-t^2 / 2)`. By the Cramer-Wold theorem, matching every
one-dimensional marginal matches the joint distribution, so the mean over directions pulls the
embeddings toward an isotropic Gaussian and away from collapse (Balestriero and LeCun 2025,
arXiv 2511.08544; Maes et al. 2026, arXiv 2603.19312).

The integrand is even in `t`, so the integral over `[-t_max, t_max]` is twice the one over
`[0, t_max]`, which the trapezoid weights carry. The defaults are LeWorldModel's: 17 knots on
`[0, 3]` and 1,024 directions, resampled at every call.
"""

from __future__ import annotations

import torch

from torch import nn


__all__ = ["SIGReg"]


class SIGReg(nn.Module):
    """The Epps-Pulley statistic of `z` along `num_slices` random directions, averaged.

    `forward(z)` takes `(..., n, d)`: `n` samples of dimension `d`, with any leading dimensions
    scored separately (LeWorldModel scores each time step) and averaged. Under the null the
    statistic is order one whatever `n`, and it grows with `n` when the samples are not Gaussian.
    """

    def __init__(self, knots: int = 17, num_slices: int = 1024, t_max: float = 3.0) -> None:
        super().__init__()
        assert knots >= 2, f"knots must be at least 2, got {knots}"
        assert num_slices > 0, f"num_slices must be positive, got {num_slices}"
        assert t_max > 0.0, f"t_max must be positive, got {t_max}"
        self.num_slices = num_slices
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)  # (k,)
        spacing = t_max / (knots - 1)
        trapezoid = torch.full((knots,), 2.0 * spacing, dtype=torch.float32)  # (k,)
        trapezoid[[0, -1]] = spacing
        gaussian_cf = torch.exp(-t.square() / 2.0)  # (k,)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("gaussian_cf", gaussian_cf, persistent=False)
        self.register_buffer("weights", trapezoid * gaussian_cf, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., n, d)
        assert z.ndim >= 2, f"SIGReg expects (..., n, d), got {tuple(z.shape)}"
        n, d = z.shape[-2], z.shape[-1]
        assert n >= 2, f"SIGReg needs at least two samples, got {n}"
        directions = torch.randn(d, self.num_slices, device=z.device, dtype=torch.float32)  # (d, s)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True)  # (d, s)
        projected = z.float() @ directions  # (..., n, s)
        arguments = projected.unsqueeze(-1) * self.t  # (..., n, s, k)
        real_error = arguments.cos().mean(dim=-3) - self.gaussian_cf  # (..., s, k)
        imaginary = arguments.sin().mean(dim=-3)  # (..., s, k)
        squared_error = real_error.square() + imaginary.square()  # (..., s, k)
        statistic = (squared_error @ self.weights) * n  # (..., s)
        return statistic.mean()  # ()
