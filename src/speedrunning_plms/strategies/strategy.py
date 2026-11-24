"""The strategy interface: what one family of a genome adds to the candidate and when it acts.

A strategy is a family's choice in a genome, configured by a frozen dataclass whose `variant` names the choice. It may
add modules to the model at construction (`build`), initialize them after the record's own initialization
(`initialize`), claim optimizer parameter groups (`parameter_groups`), transform tensors at the forward's hook points,
and act on the training loop through callbacks. Every hook has a no-op default and the model calls a hook only on the
strategies that override it (`overrides`), in registry order, so a family left at its default leaves the record's
arithmetic untouched. The model a hook receives is `train.py`'s `Candidate`, typed here as `CandidateModel`.

Symbols: b windows or proteins, l tokens, d the stream width (768 in the record), h heads, q query heads, d_h per-head
width, f the MLP width (2,048 in the record), v = 33 tokens.
"""

from __future__ import annotations

import contextlib
import torch

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask


if TYPE_CHECKING:  # the registry imports every strategy module, this one included
    from speedrunning_plms.strategies import Strategies


HOOKS = ("embed", "block_input", "normalized_input", "keys", "queries", "values", "attention_output", "mlp_hidden",
         "sublayer_output", "updated_stream", "head_input", "logits", "auxiliary_loss", "training_loss", "after_backward",
         "after_step", "evaluation_weights")


@dataclass(frozen=True)
class StrategyConfig:
    """A family's configuration: the variant and its numeric parameters, each defaulting to the record's value.

    A stored genome is a diff from these defaults, so a field's name and default never change once a run used them."""

    variant: str
    VARIANTS: ClassVar[tuple[str, ...]] = ()
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {}  # inclusive ranges of the numeric fields the search may move
    INTEGERS: ClassVar[tuple[str, ...]] = ()  # fields a genome must give as integers, such as step counts and block indices

    def __post_init__(self) -> None:
        if self.variant not in self.VARIANTS:
            raise ValueError(f"{type(self).__name__}: variant {self.variant!r} is not one of {self.VARIANTS}")
        for name, (low, high) in self.BOUNDS.items():
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"{type(self).__name__}.{name} = {value} is outside [{low}, {high}]")
        for name in self.INTEGERS:
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, not {value!r}")

    def as_document(self) -> dict[str, Any]:
        """The JSON form a genome stores: every field."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


class FadingWeight:
    """A config mixin for an auxiliary loss weight that falls linearly from `weight` at the start of the clock to 0 at
    `until`. The config declares both fields itself, with its own defaults and bounds."""

    weight: float
    until: float

    def weight_at(self, progress: float) -> float:
        """The weight at this share of the clock."""
        return self.weight * max(0.0, 1 - progress / self.until)


class CandidateModel(Protocol):
    """The candidate as a strategy sees it: `train.py`'s `Candidate`, named by exactly what the strategies read.

    `build` runs inside the candidate's construction, when only `strategies`, `extensions` and `embedding` exist;
    `blocks` and `head` exist from `initialize` on. A family registers what it adds under `extensions[family]`, which
    puts its parameters in the state dict as `extensions.<family>.*`. Each block in `blocks` holds `attention` (an
    `attention.Attention`) and `mlp` (`up` and `down`, `nn.Linear`). torch types a `ModuleList` item's submodules as
    `Tensor | Module`, so a type checker cannot follow code that reads inside a block until `train.py` types its
    blocks."""

    @property
    def strategies(self) -> Strategies:
        """One strategy per family; another family's config is `model.strategies.<family>.config`."""
        ...

    @property
    def extensions(self) -> nn.ModuleDict: ...

    @property
    def embedding(self) -> nn.Embedding: ...

    @property
    def blocks(self) -> nn.ModuleList: ...

    @property
    def head(self) -> nn.Linear: ...

    def head_features(self, X: Tensor) -> Tensor:
        """What the head reads: the normalized state through the head-input hooks."""
        ...

    def logits(self, X: Tensor) -> Tensor:
        """The head on a state, through the logit hooks."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True) -> Iterator[tuple[str, nn.Parameter]]: ...


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

    model: CandidateModel  # the uncompiled candidate
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
    """One family's choice. Subclasses set `family` and `Config`, annotate `config` with their config class, and
    override only the hooks they need.

    The forward hooks (`embed` through `auxiliary_loss`) run inside the model `torch.compile` compiles on the GPU, so a
    number they read from Python is baked into the graph: a value that changes during the run lives in a 0-d buffer a
    callback refills. `training_loss` and the callbacks run outside the compiled model, in the training loop."""

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
        """True when this strategy is active and implements the hook, so the model calls it. Read once, when the
        registry is built; a strategy whose hooks depend on its numbers as well as its variant overrides this."""
        return self.active and getattr(type(self), hook) is not getattr(Strategy, hook)

    # Construction. Every strategy's `build` and `initialize` run, in registry order, whether or not it is active; a
    # random draw in either shifts the initialization of everything built after it.
    def build(self, model: CandidateModel, max_tokens: int) -> None:
        """Add modules under `model.extensions[self.family]`. Runs inside the candidate's construction, after the
        embedding exists and before the blocks and the head do."""

    def initialize(self, model: CandidateModel) -> None:
        """Initialize what `build` added, or change the record's initialization; runs after the record's
        initialization, on the whole model."""

    def conflicts(self, strategies: Strategies) -> list[str]:
        """What this family's config cannot be built with, given the rest of the genome: a layout past the depth, a width
        the heads do not divide, a family it excludes. `resolve` raises on any (`GenomeConflict`), and the search skips
        such a child instead of spending a run on a build error."""
        return []

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        """The parameters this strategy owns and how the optimizer treats them; every other block matrix is Muon's.
        Called each time the loop builds its optimizers (before the compile warmup and again after it), and every
        trainable parameter must land in exactly one group."""
        return []

    # Forward hooks, in the order a forward meets them.
    def embed(self, model: CandidateModel, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d), the embedding.
        return X  # (b, l, d)

    def block_input(self, model: CandidateModel, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
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

    def mlp_hidden(self, index: int, H: Tensor, X: Tensor) -> Tensor:
        # H: (b, l, f), block `index`'s MLP activation before the down projection; X: (b, l, d), the MLP's normalized
        # input, which a router reads.
        return H  # (b, l, f)

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d), the attention's or the MLP's output before the residual update.
        return Y  # (b, l, d)

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream right after block `index`'s "attention" or "mlp" residual update.
        return X  # (b, l, d)

    def head_input(self, model: CandidateModel, X: Tensor) -> Tensor:
        # X: (b, l, d), the normalized state the 33-way head reads.
        return X  # (b, l, d)

    def logits(self, model: CandidateModel, logits: Tensor) -> Tensor:
        """The head's logits; not called on the loop's CPU training forward, which keeps the record's arithmetic."""
        # logits: (b, l, v).
        return logits  # (b, l, v)

    def auxiliary_loss(self, model: CandidateModel, states: list[Tensor], labels: Tensor,
                       context: ForwardContext) -> Tensor:
        """A summed loss added to the training loss, with the stream after each block and the forward's context (what
        the hooks stored); called only in a forward given `labels`, which the loop passes on the flex path (the GPU)
        alone."""
        # states: (b, l, d) each, 37 of them; labels: (b, l); returns ().
        raise NotImplementedError

    def training_loss(self, logits: Tensor, labels: Tensor, loss: Tensor) -> Tensor:
        """The summed objective the step differentiates, from the record's summed masked cross entropy `loss`; called
        outside the compiled model on both training paths. The logged loss stays the record's."""
        # logits: (b, l, v); labels: (b, l) with IGNORE_LABEL where unscored; loss: (); returns ().
        return loss  # ()

    # Training callbacks, on the uncompiled model inside the clock. They also run on the compile warmup's synthetic steps
    # (steps 0 to 2 of each window shape, progress 0), whose weights the loop then restores, so a callback that keeps
    # state resets it at step 0.
    def after_backward(self, state: StepState) -> None:
        """Edit gradients before the optimizers step."""

    def after_step(self, state: StepState) -> None:
        """Edit weights or running statistics after the optimizers step and the gradients are cleared."""

    @property
    def evaluation_options(self) -> int:
        """How many weight options each validation pass scores; the loop keeps the lowest loss, sets `option` to each
        in turn before entering `evaluation_weights`, and reads only val_select."""
        return 1

    option: int = 0  # the evaluation option `evaluation_weights` gives, set by the loop

    @contextlib.contextmanager
    def evaluation_weights(self, model: CandidateModel) -> Iterator[None]:
        """The weights a validation pass scores, entered around each in-run pass outside the clock; the live weights
        return afterwards."""
        yield


def norm(X: Tensor) -> Tensor:
    # X: (..., k); RMS normalization over the last dimension, no affine.
    return torch.nn.functional.rms_norm(X, (X.shape[-1],))  # (..., k)


def extension(model: CandidateModel, family: str) -> nn.Module:
    """The module a family's `build` registered on the model."""
    return model.extensions[family]


def config_from(config_type: type[StrategyConfig], document: dict[str, Any]) -> StrategyConfig:
    """A family's config from its genome entry; unknown keys raise, missing keys take the record's defaults."""
    known = {item.name for item in fields(config_type)}
    unknown = sorted(set(document) - known)
    if unknown:
        raise ValueError(f"{config_type.__name__} has no field {unknown}; it has {sorted(known)}")
    return config_type(**document)
