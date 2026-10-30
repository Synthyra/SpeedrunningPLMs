"""The backprop-free family: pieces of the Dust, noise and Boost line (`ideas/backprop_free.md`) beside backprop.

- `residual_es` (B8, `backprop_free.md` 1.7 and 5.4): a scale on every sublayer's output, 72 in all, trained by
  two-sided evolution strategies instead of gradients. Every `interval` steps two extra forwards score the batch at
  s + sigma eps and s - sigma eps; s moves by -es_lr (L+ - L-) / (2 sigma) eps, times the step's lr multiplier. The
  forwards run inside the clock.
- `unit_select` (B9, recipe 3): every `interval` steps a forward-only statistic ranks each block's 2,048 MLP units by
  mean activation times the norm of the unit's down-projection column, read from the training forward itself; only
  the top `keep` share of units (rows of the up projection, columns of the down) receive gradients until the next
  ranking. Boost's hidden-unit credit is replaced by this saliency, which needs no extra pass. Muon's momentum still
  carries a masked unit's earlier updates, decaying at its momentum rate.
- `unit_random`: the control for `unit_select`, the same share of units drawn at random each interval.
- `readout_warmstart` (C2, recipe 1, in part): the shared head reads the stream after every `stride`-th block and
  each readout's masked loss joins the training loss, weighted 1 for the first `fraction` of the clock and 0 after,
  so every depth gets a direct signal early. This is Boost's per-block shared readout trained by backprop, not its
  orthogonal-nudge estimator: the estimator's 36-block cost is unmeasured and its one comparison trails backprop.
- `readout_boost` (`backprop_free.md` 3.2 and 3.7): Boost's readout sites, read by backprop. The shared head reads the
  stream right after every attention update and every MLP update (Boost's xmid and xout, 72 readouts, captured by the
  `updated_stream` hook in a training forward), and their mean masked loss, times `readout_weight`, joins the training
  loss for the first `switch` share of the clock (Boost's phase switch, 0.3), then stops. The weight lives in a 0-d
  buffer that `after_step` refills from the clock's progress, so the compiled graph never sees a changing Python float.
- `unit_route` (`backprop_free.md` 3.4 and 3.8): Cut's routes as a gradient mask. In each forward, each token's
  `route_k` units per block with the largest relu^2(pre-activation) times the norm of the unit's down-projection
  column pass gradient back into the up projection; the other units keep their forward values and pass none, as
  Cut's unrouted units stay at their clean outputs. The forward is unchanged and the down projection's gradient is
  whole. `unit_select` differs in what it ranks and how long a choice lasts: one set of units for every token, ranked
  by the batch's mean activation times the column norm every `interval` steps, masking the weight gradients until the
  next ranking; `unit_route` ranks per token, per step, from the live pre-activations.

Symbols: b windows or proteins, l tokens, d = 768, f = 2,048, v = 33 tokens, k = route_k.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, StepState, Strategy, StrategyConfig


SUBLAYERS = ("attention", "mlp")


@dataclass(frozen=True)
class BackpropFreeConfig(StrategyConfig):
    variant: str = "none"
    interval: int = 4  # steps between ES updates or unit rankings
    sigma: float = 0.05  # ES perturbation scale
    es_lr: float = 0.02
    keep: float = 0.5  # share of MLP units that train between rankings
    stride: int = 6  # readout after every stride-th block
    fraction: float = 0.3  # share of the clock with the readouts on
    readout_weight: float = 0.3  # readout_boost: the readouts' mean loss weight while on
    switch: float = 0.3  # readout_boost: share of the clock with the readouts on, Boost's phase switch
    route_k: int = 192  # unit_route: units a token routes per block (Cut-lite's 192)
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "residual_es", "unit_select", "unit_random", "readout_warmstart",
                                           "readout_boost", "unit_route")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "interval": (1, 50), "sigma": (0.005, 0.3), "es_lr": (0.001, 0.2), "keep": (0.05, 0.95), "stride": (1, 18),
        "fraction": (0.05, 0.8), "readout_weight": (0.05, 1.0), "switch": (0.1, 1.0), "route_k": (64, 1024),
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.route_k, int):
            raise ValueError(f"route_k must be an integer, not {self.route_k!r}")


class BackpropFree(Strategy):
    family = "backprop_free"
    Config = BackpropFreeConfig
    config: BackpropFreeConfig

    def overrides(self, hook: str) -> bool:
        uses = {"residual_es": ("sublayer_output", "after_step"),
                "unit_select": ("mlp_hidden", "after_backward", "after_step"),
                "unit_random": ("after_backward", "after_step"),
                "readout_warmstart": ("auxiliary_loss", "after_step"),
                "readout_boost": ("updated_stream", "auxiliary_loss", "after_step"),
                "unit_route": ("mlp_hidden",)}
        return hook in uses.get(self.variant, ())

    def build(self, model: nn.Module, max_tokens: int) -> None:
        holder = nn.Module()
        blocks, f = benchmark.BLOCKS, benchmark.INTERMEDIATE
        if self.variant == "residual_es":
            holder.register_buffer("scales", torch.ones(blocks, 2))  # (36, 2): attention, MLP; kept in best.pt
        elif self.variant in ("unit_select", "unit_random"):
            holder.register_buffer("masks", torch.ones(blocks, f), persistent=False)  # (36, f), 1 where a unit trains
            holder.register_buffer("activity", torch.zeros(blocks, f), persistent=False)  # (36, f), the last forward's means
        elif self.variant == "readout_warmstart":
            holder.register_buffer("weight", torch.ones(()), persistent=False)  # (), the readouts' weight; a tensor, so no recompile
        elif self.variant == "readout_boost":
            holder.register_buffer("weight", torch.tensor(self.config.readout_weight), persistent=False)  # (), refilled each step
        if self.active:
            model.extensions[self.family] = holder  # type: ignore[index]
            self.holder = holder
            self.generator: torch.Generator | None = None

    def _generator(self, device: torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(20261029)
        return self.generator

    # Residual-scalar ES.
    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d).
        return Y * self.holder.scales[index, 0 if sublayer == "attention" else 1].to(Y.dtype)  # (b, l, d)

    def _es_update(self, state: StepState) -> None:
        scales = self.holder.scales  # (36, 2)
        eps = torch.randn(scales.shape, generator=self._generator(scales.device), device=scales.device)  # (36, 2)
        count = (state.labels != benchmark.IGNORE_LABEL).sum().clamp(min=1)  # ()
        base = scales.clone()  # (36, 2)
        scales.copy_(base + self.config.sigma * eps)
        loss_plus = state.loss_without_grad() / count  # ()
        scales.copy_(base - self.config.sigma * eps)
        loss_minus = state.loss_without_grad() / count  # ()
        gradient = (loss_plus - loss_minus) / (2 * self.config.sigma) * eps  # (36, 2)
        scales.copy_(base - self.config.es_lr * state.lr_scale * gradient)

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "unit_route":
            self.down_weights = [block.mlp.down.weight for block in model.blocks]  # type: ignore[union-attr]  # each (d, f)

    # Unit selection and routes.
    def mlp_hidden(self, index: int, H: Tensor) -> Tensor:
        # H: (b, l, f), relu^2 of the pre-activations under the record's activation.
        if self.variant == "unit_route":
            down_norms = self.down_weights[index].detach().float().norm(dim=0)  # (f,)
            scores = H.detach().float() * down_norms  # (b, l, f)
            routes = scores.topk(self.config.route_k, dim=-1).indices  # (b, l, k)
            routed = torch.zeros_like(H, dtype=torch.bool).scatter_(-1, routes, True)  # (b, l, f)
            return torch.where(routed, H, H.detach())  # (b, l, f); the same values, gradient through routed units only
        # unit_select: each unit's mean activation, a forward-only statistic.
        self.holder.activity[index].copy_(H.detach().float().mean(dim=(0, 1)))  # (f,)
        return H  # (b, l, f)

    # Boost readouts.
    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d); kept for the readouts in a training forward only, so evaluation holds no extra streams.
        if context.labels is not None:
            context.stored[f"readout_{index}_{sublayer}"] = X
        return X  # (b, l, d)

    def _rank_units(self, model: nn.Module) -> None:
        masks = self.holder.masks  # (36, f)
        keep = int(self.config.keep * masks.shape[1])
        if self.variant == "unit_random":
            scores = torch.rand(masks.shape, generator=self._generator(masks.device), device=masks.device)  # (36, f)
        else:
            down_norms = torch.stack([block.mlp.down.weight.detach().float().norm(dim=0) for block in model.blocks])  # (36, f)
            scores = self.holder.activity * down_norms  # (36, f)
        chosen = scores.topk(keep, dim=1).indices  # (36, keep)
        masks.zero_().scatter_(1, chosen, 1.0)  # (36, f)

    def after_backward(self, state: StepState) -> None:
        masks = self.holder.masks  # (36, f)
        for index, block in enumerate(state.model.blocks):  # type: ignore[union-attr]
            mask = masks[index]  # (f,)
            if block.mlp.up.weight.grad is not None:
                block.mlp.up.weight.grad.mul_(mask[:, None].to(block.mlp.up.weight.grad.dtype))  # (f, d)
                block.mlp.down.weight.grad.mul_(mask[None, :].to(block.mlp.down.weight.grad.dtype))  # (d, f)

    # Readout warm start.
    def auxiliary_loss(self, model: nn.Module, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        # states: (b, l, d) each, 37 of them; labels: (b, l); returns (), the readouts' summed loss averaged over readouts.
        if self.variant == "readout_boost":
            streams = [context.stored[f"readout_{index}_{sublayer}"] for index in range(benchmark.BLOCKS) for sublayer in SUBLAYERS]
        else:
            streams = [states[index] for index in range(self.config.stride, benchmark.BLOCKS, self.config.stride)]
        total = sum(benchmark.masked_cross_entropy(model.logits(X), labels) for X in streams)  # ()
        return self.holder.weight * total / len(streams)  # ()

    def after_step(self, state: StepState) -> None:
        if self.variant == "residual_es" and state.step % self.config.interval == 0:
            self._es_update(state)
        elif self.variant in ("unit_select", "unit_random") and state.step % self.config.interval == 0:
            self._rank_units(state.model)
        elif self.variant == "readout_warmstart":
            self.holder.weight.fill_(1.0 if state.progress < self.config.fraction else 0.0)
        elif self.variant == "readout_boost":
            self.holder.weight.fill_(self.config.readout_weight if state.progress < self.config.switch else 0.0)
