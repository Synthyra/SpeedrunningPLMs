"""The strategy interface: what one family of a genome adds to the candidate and when it acts.

A strategy is a family's choice in a genome, configured by a frozen dataclass whose `variant` names the choice. It may
add modules to the model at construction (`build`), initialize them after the record's own initialization
(`initialize`), claim optimizer parameter groups (`parameter_groups`), transform tensors at the forward's hook points,
and act on the training loop through callbacks. Every hook has a no-op default and the model calls a hook only on the
strategies that override it, so a family left at its default leaves the record's arithmetic untouched.

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, f = 2,048, v = 33 tokens.
"""

from __future__ import annotations

import contextlib
import torch

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask


HOOKS = ("embed", "block_input", "normalized_input", "keys", "queries", "values", "attention_output", "mlp_hidden",
         "sublayer_output", "updated_stream", "head_input", "logits", "auxiliary_loss", "training_loss", "after_backward",
         "after_step", "evaluation_weights")


@dataclass(frozen=True)
class StrategyConfig:
    """A family's configuration: the variant and its numeric parameters, each defaulting to the record's value."""

    variant: str
    VARIANTS: ClassVar[tuple[str, ...]] = ()
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {}  # inclusive ranges of the numeric fields the search may move

    def __post_init__(self) -> None:
        if self.variant not in self.VARIANTS:
            raise ValueError(f"{type(self).__name__}: variant {self.variant!r} is not one of {self.VARIANTS}")
        for name, (low, high) in self.BOUNDS.items():
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"{type(self).__name__}.{name} = {value} is outside [{low}, {high}]")

    def as_document(self) -> dict[str, Any]:
        """The JSON form a genome stores: every field."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass
class ParameterGroup:
    """Parameters a strategy hands the optimizer: Adam at a rate, or Muon beside the block matrices."""

    name: str
    parameters: list[nn.Parameter]
    optimizer: Literal["adam", "muon"]
    lr: float


@dataclass
class ForwardContext:
    """What one forward pass shares between hook points: the tokens, the attention masks and the stored tensors."""

    input_ids: Tensor  # (b, l)
    block_masks: tuple[BlockMask, ...] | None  # the kernel's flex masks, in training on the GPU
    dense_mask: Tensor | None  # (b, 1, 1, l) padding or (b, 1, l, l) document mask, bool, for SDPA
    stored: dict[str, Tensor] = field(default_factory=dict)  # what a later block reads, such as the first values
    labels: Tensor | None = None  # (b, l), only in a training forward that computes auxiliary losses


@dataclass
class StepState:
    """The training loop's view of one step, handed to the callbacks after the backward and after the update."""

    model: nn.Module  # the uncompiled candidate
    step: int  # applied steps before this one
    progress: float  # share of the clock spent
    lr_scale: float
    corrupted: Tensor  # (b, l)
    labels: Tensor  # (b, l)
    loss_without_grad: Callable[[], Tensor]  # the summed masked loss of this batch at the current weights, ()
    # The step's forward and backward again at the current weights, replacing every gradient (the loop's train_step on
    # the same batch and masks); None where no loop provides one.
    recompute_gradient: Callable[[], None] | None = None
    # The run's budget over the 10-minute track's 600 s: a number given in steps (an averaging window, a decay's memory,
    # a decay timescale) is read at the 10-minute track's steps and scaled by this, so one value means one share of the
    # run on every track.
    run_scale: float = 1.0


class Strategy:
    """One family's choice. Subclasses set `family` and `Config` and override only the hooks they need."""

    family: ClassVar[str]
    Config: ClassVar[type[StrategyConfig]]

    def __init__(self, config: StrategyConfig) -> None:
        assert isinstance(config, self.Config), f"{self.family} takes {self.Config.__name__}"
        self.config = config

    @property
    def variant(self) -> str:
        return self.config.variant

    @property
    def active(self) -> bool:
        """False for a family switched off (`none`): its hooks are never called."""
        return self.variant != "none"

    def overrides(self, hook: str) -> bool:
        """True when this strategy is active and implements the hook, so the model calls it."""
        return self.active and getattr(type(self), hook) is not getattr(Strategy, hook)

    # Construction.
    def build(self, model: nn.Module, max_tokens: int) -> None:
        """Add modules under `model.extensions[self.family]`; runs after the record's modules exist."""

    def initialize(self, model: nn.Module) -> None:
        """Initialize what `build` added; runs after the record's initialization."""

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        """The parameters this strategy owns and how the optimizer treats them; every other block matrix is Muon's."""
        return []

    # Forward hooks, in the order a forward meets them.
    def embed(self, model: nn.Module, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d), the embedding.
        return X  # (b, l, d)

    def block_input(self, model: nn.Module, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
        # X: (b, l, d) entering block `index`; states: (b, l, d) each, the embedding and every earlier block's output.
        return X  # (b, l, d)

    def normalized_input(self, index: int, sublayer: str, X: Tensor) -> Tensor:
        # X: (b, l, d), block `index`'s normalized stream entering its "attention" or "mlp" sublayer.
        return X  # (b, l, d)

    def keys(self, index: int, K: Tensor, context: ForwardContext) -> Tensor:
        # K: (b, l, h, d_h), block `index`'s raw keys from the key projection, before QK-norm and RoPE.
        return K  # (b, l, h, d_h)

    def queries(self, index: int, Q: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, l, q, d_h), block `index`'s queries after QK-norm and RoPE, before attention.
        return Q  # (b, l, q, d_h)

    def values(self, index: int, V: Tensor, context: ForwardContext) -> Tensor:
        # V: (b, l, h, d_h), block `index`'s values before attention.
        return V  # (b, l, h, d_h)

    def attention_output(self, index: int, X: Tensor, Y: Tensor) -> Tensor:
        # X: (b, l, d), the attention's input; Y: (b, h, l, d_h), the heads' outputs.
        return Y  # (b, h, l, d_h)

    def mlp_hidden(self, index: int, H: Tensor) -> Tensor:
        # H: (b, l, f), block `index`'s MLP activation before the down projection.
        return H  # (b, l, f)

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d), the attention's or the MLP's output before the residual update.
        return Y  # (b, l, d)

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream right after block `index`'s "attention" or "mlp" residual update.
        return X  # (b, l, d)

    def head_input(self, model: nn.Module, X: Tensor) -> Tensor:
        # X: (b, l, d), the normalized state the 33-way head reads.
        return X  # (b, l, d)

    def logits(self, model: nn.Module, logits: Tensor) -> Tensor:
        # logits: (b, l, v).
        return logits  # (b, l, v)

    def auxiliary_loss(self, model: nn.Module, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        """A summed loss added to the training loss; called only in training, with the stream after each block and
        the forward's context (what the hooks stored)."""
        # states: (b, l, d) each, 37 of them; labels: (b, l); returns ().
        raise NotImplementedError

    def training_loss(self, logits: Tensor, labels: Tensor, loss: Tensor) -> Tensor:
        """The summed objective the step differentiates, from the record's summed masked cross entropy `loss`; called
        outside the compiled model on both training paths. The logged loss stays the record's."""
        # logits: (b, l, v); labels: (b, l) with IGNORE_LABEL where unscored; loss: (); returns ().
        return loss  # ()

    # Training callbacks.
    def after_backward(self, state: StepState) -> None:
        """Edit gradients before the optimizers step."""

    def after_step(self, state: StepState) -> None:
        """Edit weights or running statistics after the optimizers step."""

    @property
    def evaluation_options(self) -> int:
        """How many weight options each validation pass scores; the loop keeps the lowest loss, sets `option` to each
        in turn before entering `evaluation_weights`, and reads only val_select."""
        return 1

    option: int = 0  # the evaluation option `evaluation_weights` gives, set by the loop

    @contextlib.contextmanager
    def evaluation_weights(self, model: nn.Module) -> Iterator[None]:
        """The weights a validation pass scores; the live weights return afterwards."""
        yield


def norm(X: Tensor) -> Tensor:
    # X: (..., k); RMS normalization over the last dimension, no affine.
    return torch.nn.functional.rms_norm(X, (X.shape[-1],))  # (..., k)


def extension(model: nn.Module, family: str) -> nn.Module:
    """The module a family's `build` registered on the model."""
    extensions: nn.ModuleDict = model.extensions  # type: ignore[assignment]
    return extensions[family]


def config_from(config_type: type[StrategyConfig], document: dict[str, Any]) -> StrategyConfig:
    """A family's config from its genome entry; unknown keys raise, missing keys take the record's defaults."""
    known = {item.name for item in fields(config_type)}
    unknown = sorted(set(document) - known)
    if unknown:
        raise ValueError(f"{config_type.__name__} has no field {unknown}; it has {sorted(known)}")
    return config_type(**document)
