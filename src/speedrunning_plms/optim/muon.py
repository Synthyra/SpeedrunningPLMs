"""Muon updates for hidden matrices whose gradients are already synchronized."""

from __future__ import annotations

import math
import torch

from collections.abc import Callable, Iterable


# Five-step schedule with a 2% safety factor from modded-nanogpt's Polar Express:
# https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
# Method: https://arxiv.org/abs/2505.16932
POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _orthogonalize(
    G: torch.Tensor, coefficients: tuple[tuple[float, float, float], ...],
    safety_factor: float, epsilon: float,
) -> torch.Tensor:
    # G: (m, n); r = min(m, n), s = max(m, n).
    if G.ndim != 2 or not G.is_floating_point():
        raise ValueError("Orthogonalization requires a floating-point matrix")
    with torch.autocast(device_type=G.device.type, enabled=False):
        X = G.float()  # (m, n); normalize in FP32 before any BF16 conversion.
        X = X / (X.norm() * (1 + safety_factor) + epsilon)  # (m, n)
        if G.device.type == "cuda":
            X = X.bfloat16()  # (m, n)
        if G.shape[0] > G.shape[1]:
            X = X.T  # (n, m); X is now (r, s).
        for a, b, c in coefficients:
            A = X @ X.T  # (r, r)
            B = b * A + c * (A @ A)  # (r, r)
            X = a * X + B @ X  # (r, s)
        if G.shape[0] > G.shape[1]:
            X = X.T  # (m, n)
        return X  # (m, n); BF16 on CUDA and FP32 elsewhere.


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Apply quintic Newton-Schulz steps to approximately orthogonalize a matrix."""
    # G and return: (m, n).
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    return _orthogonalize(G, ((3.4445, -4.7750, 2.0315),) * steps, 0.0, 1e-7)  # (m, n)


def zeropower_via_polar_express(G: torch.Tensor) -> torch.Tensor:
    """Apply the fixed five-step Polar Express coefficient schedule."""
    # G and return: (m, n).
    return _orthogonalize(G, POLAR_EXPRESS_COEFFICIENTS, 0.02, 1e-6)  # (m, n)


class Muon(torch.optim.Optimizer):
    """Apply momentum, orthogonalization, and decoupled decay to 2D weights.

    DDP must synchronize gradients before ``step``. This optimizer performs no
    communication and supports arbitrary parameter counts on CPU and CUDA.
    Updates scale by sqrt(max(1, rows / columns)); ``lr`` is a Muon-specific
    rate, independent of the AdamW rate used for embeddings and output heads.
    """

    def __init__(
        self, params: Iterable[torch.Tensor], lr: float = 0.02,
        momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
        weight_decay: float = 0.0, orthogonalization: str = "newton_schulz",
    ) -> None:
        for name, value in (("lr", lr), ("weight_decay", weight_decay)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        if type(ns_steps) is not int or ns_steps < 1:
            raise ValueError("ns_steps must be a positive integer")
        if orthogonalization not in {"newton_schulz", "polar_express"}:
            raise ValueError("orthogonalization must be newton_schulz or polar_express")
        if orthogonalization == "polar_express" and ns_steps != 5:
            raise ValueError("polar_express requires ns_steps=5 for its coefficient schedule")
        parameters = list(params)  # Each parameter is (m, n).
        if any(p.ndim != 2 or not p.is_floating_point() for p in parameters):
            raise ValueError("Muon requires floating-point matrix parameters")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        orthogonalization=orthogonalization)
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None  # Optional scalar ().
        if closure is not None:
            with torch.enable_grad():
                loss = closure()  # ()
        for group in self.param_groups:
            for parameter in group["params"]:  # parameter: (m, n).
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                gradient = parameter.grad.float()  # (m, n)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)  # (m, n), FP32.
                # Optimizer.load_state_dict may cast buffers to parameter dtype.
                buffer = state["momentum_buffer"].float()  # (m, n), FP32.
                state["momentum_buffer"] = buffer  # (m, n)
                buffer.lerp_(gradient, 1 - group["momentum"])  # (m, n)
                update = gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer  # (m, n)
                if group["orthogonalization"] == "polar_express":
                    update = zeropower_via_polar_express(update)  # (m, n)
                else:
                    update = zeropower_via_newtonschulz5(update, group["ns_steps"])  # (m, n)
                scale = max(1, parameter.shape[0] / parameter.shape[1]) ** 0.5
                parameter.mul_(1 - group["lr"] * group["weight_decay"])  # (m, n)
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"] * scale)  # (m, n)
        return loss  # Optional scalar ().
