"""The loop family: a looped core at the record's depth, the looped-transformer recipe of 2025 and 2026 in the speedrun.

The 36 blocks are read as positions: a prelude of `prelude` blocks run once, a core of `core` blocks run `passes`
times, and a coda of the 36 - prelude - core x passes left, run once (Huginn, arXiv 2502.05171; recAMPLIFY, arXiv
2610.26747, a (2, 8, 2) layout over 6 cycles; Inkling 2 Mini, its middle 12 of 24 layers three times; Dory1.0-4B, 8-20-8
blocks). Every position stays a block of the record's shape, so a forward costs what the record's does and the shape
rule holds: a loop here trades stored parameters for repeated ones at the same compute, it never adds depth.

- `tied`: each core position past the first pass shares every matrix of its attention and its MLP (each `nn.Linear`
  weight inside them) with the same block of the first pass, so prelude + core + coda blocks are stored (20 of 36 at
  the defaults). What other families keep per block (norm scalings, value-residual shares, gates) stays one per
  position, as in Relaxed Recursive Transformers (arXiv 2410.20672). Muon orthogonalizes fewer matrices a step.
- `untied`: the same positions and pieces with nothing shared: the control that separates what tying does from what
  re-injection, scaling, sampled depth and readouts do.

The pieces, all off at their defaults but LOOM's re-injection:

- `inject`: at the start of pass t >= 2 the stream becomes (1 - g_t) h + g_t e, e the prelude's output. `loom`:
  g_t = `inject_lambda` / (t sqrt(core)) (LOOM, arXiv 2610.01153, Eq. 2, lambda 0.5; LOOM has no prelude and re-injects
  the embedding); `learned`: g per pass and channel, the sigmoid of a parameter that starts at LOOM's value, AdamW at
  the scalar rate, the nearest form here to Inkling's gated cross-pass message; `none`.
  `read`: the loop-native residual read (InfiLoop, arXiv 2610.11570; Loopback, arXiv 2611.07226; the per-channel-head
  query of Multi-Head Attention Residuals, arXiv 2607.27230; SGLR's `residual_read`, 2026-11-23_residual_routing). The
  prelude's output e stays on its own path, and after pass t the stream is e + sum_j w_tj u_j over the passes' updates
  u_j (a pass's output minus its input), with w_tj = n softmax_j(q_t . RMSNorm(u_j) - lambda (t - j)) per token and
  channel head (the attention's heads), q_t one learned query per pass, lambda a learned decay per head that starts at
  `read_decay` (InfiLoop starts at 0.55), AdamW at the scalar rate in float32. At q = 0 and lambda = 0 every weight is
  n x 1/n = 1 and the stream is the plain carried sum, so it starts as `none`. Carrying only the last state lets a
  noisy pass overwrite a good one ("latent overthinking"); the read lets a token weigh its passes.
- `write_gate`: 1 scales each pass's update per token and channel head by 2 sigmoid(w_t . RMSNorm(u) + b_t), its own
  pass's weights from zero, so every gate starts at exactly one and a pass can learn to decline a write (mHC's post map,
  arXiv 2512.24880; the zero-initialized write gates of arXiv 2611.24567; SGLR's `residual_write`).
- `residual_lambda`: when not 0, every core sublayer's output is scaled by lambda / (passes sqrt(core)) before its
  residual update, LOOM's gamma (Eq. 1, lambda 0.5), which bounds the stream's growth over passes. LOOM also
  normalizes its MoE output; that norm is left out, since a zero-initialized output projection has no direction.
- `sample`: in training each window draws its own depth k, uniform over 1 to `passes`, each step (recAMPLIFY samples
  per minibatch, Dory per step; one compiled graph cannot change its pass count, so here the draw is per window). The
  passes past k still run and their updates are discarded at every core block, so the window's coda reads its stream
  after pass k and the pass count becomes a test-time dial. A sampled depth saves no compute.
- `grad_passes`: when not 0, only the last n passes carry gradient: the stream is detached where pass `passes` - n
  ends, and the prelude still learns through e (Huginn's truncated backpropagation; LOOM's segments of 3 passes). The
  cut passes skip their backward, which buys steps on the clock. Tied only, with re-injection, without sampled depth.
- `readout_weight`: the shared head reads the stream at the end of every pass but the last (past the cut, when
  truncated), and the readouts' mean masked loss times the weight joins the training loss for the whole run, a fixed
  per-depth prior. Popescu et al. (arXiv 2607.20519) find that such depth supervision with a post-hoc confidence
  readout matches or beats a jointly trained halting gate, which therefore is not here.
- Evaluation: `eval_passes` (0: every pass) sets the dial, and `exit` stops a token after a pass: `confidence` when
  the shared head's readout puts at least `exit_threshold` of its 20 residue probabilities on one residue,
  `fixed_point` when the pass moved the token's state by less than `exit_threshold` of its norm (FPRM-style, arXiv
  2606.18206). A stopped or undialed token keeps its state and still serves as a key and value; its compute is not
  saved, and the strategy counts the passes each real token ran (`pass_tokens`) for the dial study. The run's own
  evaluations follow the config; a study sets `eval_passes`, `exit` and `exit_threshold` on the strategy.

The loop needs a prelude of at least one block and the single stream: its pass boundaries live in `updated_stream`
after the MLP of the prelude's last block and of each pass's last block, where the forward's context holds what a later
pass reads (the prelude's output, each pass's start, the running tokens).

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens, c = 20 residue columns.
"""

from __future__ import annotations

import math
import torch

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


if TYPE_CHECKING:  # the registry imports this module
    from speedrunning_plms.strategies import Strategies


INJECTIONS = ("none", "loom", "learned", "read")
EXITS = ("none", "confidence", "fixed_point")
PRELUDE_OUTPUT = "loop_prelude_output"  # e, re-injected at every pass after the first
PASS_INPUT = "loop_pass_input"  # the stream the current pass began from, after any injection
UPDATES = "loop_updates"  # the passes' updates so far, (b, l, d) each, for the residual read
HELD = "loop_held"  # the stream where the current pass began, before its injection: what a stopped token keeps
RUNNING = "loop_running"  # (b, l or 1, 1) bool, the tokens the current pass updates; absent when every token runs
DEPTH = "loop_depth"  # (b, 1, 1), each window's sampled pass count
STOPPED = "loop_stopped"  # (b, l, 1) bool, the tokens an exit has stopped, in evaluation
PASS_END = "loop_pass_end_"  # + pass number: the stream a readout reads


@dataclass(frozen=True)
class LoopConfig(StrategyConfig):
    variant: str = "none"
    prelude: int = 6
    core: int = 8
    passes: int = 3
    inject: str = "loom"
    inject_lambda: float = 0.5  # LOOM's lambda for the re-injection share; the starting share when `learned`
    residual_lambda: float = 0.0  # 0 leaves the core's sublayer outputs alone; LOOM's lambda is 0.5
    sample: int = 0  # 1 draws each window's depth in training
    grad_passes: int = 0  # 0 backpropagates through every pass
    readout_weight: float = 0.0
    read_decay: float = 0.0  # the residual read's starting decay per head, read only with `inject` "read", as
    # `inject_lambda` is only with "loom" and "learned"; 0 starts the read at the plain sum
    write_gate: int = 0  # 1 gates each pass's update per token and channel head, from one
    eval_passes: int = 0  # the evaluation's dial; 0 runs every pass
    exit: str = "none"
    exit_threshold: float = 0.9
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "tied", "untied")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "prelude": (1, 34), "core": (1, 35), "passes": (1, 35), "inject_lambda": (0.05, 4.0), "residual_lambda": (0.0, 16.0),
        "sample": (0, 1), "grad_passes": (0, 34), "readout_weight": (0.0, 1.0), "read_decay": (0.0, 2.0), "write_gate": (0, 1),
        "eval_passes": (0, 35), "exit_threshold": (0.0, 1.0),
    }
    INTEGERS: ClassVar[tuple[str, ...]] = ("prelude", "core", "passes", "sample", "grad_passes", "write_gate", "eval_passes")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.inject not in INJECTIONS:
            raise ValueError(f"inject must be one of {INJECTIONS}, not {self.inject!r}")
        if self.grad_passes and self.inject == "read":
            raise ValueError("the residual read reads every pass's update, so it cannot cut the passes before grad_passes")
        if self.exit not in EXITS:
            raise ValueError(f"exit must be one of {EXITS}, not {self.exit!r}")
        if self.eval_passes > self.passes:
            raise ValueError(f"eval_passes {self.eval_passes} exceeds passes {self.passes}")
        if self.grad_passes:
            if self.grad_passes >= self.passes:
                raise ValueError(f"grad_passes {self.grad_passes} must be under passes {self.passes} (0 keeps every pass)")
            if self.variant == "untied" or self.sample or self.inject == "none":
                raise ValueError("truncated passes need tied blocks (the cut blocks would never train), no sampled depth "
                                 "and a re-injection (the prelude learns only through it)")
        if self.readout_weight and not self.readout_passes:
            raise ValueError("readout_weight needs a pass to read before the last one (and past the cut, when truncated)")

    def coda(self, blocks: int) -> int:
        """The blocks after the last pass, in a model of `blocks`."""
        return blocks - self.prelude - self.core * self.passes

    def stored_blocks(self, blocks: int) -> int:
        """The blocks whose matrices are stored once: all of them untied, prelude + core + coda tied."""
        return self.prelude + self.core + self.coda(blocks) if self.variant == "tied" else blocks

    @property
    def readout_passes(self) -> tuple[int, ...]:
        """The passes, 1-based, whose ends the readouts read: every one but the last, past the cut when truncated."""
        first = self.passes - self.grad_passes + 1 if self.grad_passes else 1
        return tuple(range(first, self.passes))

    @property
    def residual_scale(self) -> float:
        """LOOM's gamma on the core's sublayer outputs, or 1."""
        return self.residual_lambda / (self.passes * math.sqrt(self.core)) if self.residual_lambda else 1.0

    def share(self, start: int) -> float:
        """LOOM's re-injection share g_t at the start of pass t (1-based, t >= 2)."""
        return self.inject_lambda / (start * math.sqrt(self.core))

    def is_core(self, position: int) -> bool:
        return self.prelude <= position < self.prelude + self.core * self.passes

    def source(self, position: int) -> int:
        """The block of the first pass a core position repeats; any other position is its own."""
        return self.prelude + (position - self.prelude) % self.core if self.is_core(position) else position


def tie_matrices(target: nn.Module, source: nn.Module) -> None:
    """Point every `nn.Linear` weight (and bias) in `target`'s attention and MLP at the same layer's in `source`."""
    for sublayer in ("attention", "mlp"):
        targets = {name: layer for name, layer in getattr(target, sublayer).named_modules() if isinstance(layer, nn.Linear)}
        sources = {name: layer for name, layer in getattr(source, sublayer).named_modules() if isinstance(layer, nn.Linear)}
        if targets.keys() != sources.keys():
            raise ValueError(f"cannot tie {sublayer} layers {sorted(targets)} to {sorted(sources)}: the blocks differ "
                             "(attention.kv_share gives blocks different projections)")
        for name, layer in targets.items():
            assert layer.weight.shape == sources[name].weight.shape, f"{sublayer}.{name}: shapes differ"
            layer.weight = sources[name].weight
            if layer.bias is not None:
                layer.bias = sources[name].bias


def reinjected(X: Tensor, E: Tensor, share: Tensor | float) -> Tensor:
    """(1 - g) X + g E: the stream pulled toward the prelude's output by the share g."""
    # X, E: (b, l, d); share: () or (d,); returns (b, l, d).
    return X + share * (E - X)  # (b, l, d)


def read_updates(E: Tensor, updates: list[Tensor], query: Tensor, decay: Tensor) -> Tensor:
    """The residual read: e plus the updates weighted per token and channel head by n softmax(q . RMSNorm(u) - lambda age)."""
    # E: (b, l, d); updates: n of (b, l, d), oldest first; query: (h, d_h); decay: (h,); returns (b, l, d).
    b, l, d = E.shape
    n, heads = len(updates), query.shape[0]
    U = torch.stack(updates, dim=2).view(b, l, n, heads, d // heads)  # (b, l, n, h, d_h)
    age = torch.arange(n - 1, -1, -1, device=E.device, dtype=torch.float32)  # (n,): the newest update is 0 passes old
    with torch.autocast(E.device.type, enabled=False):
        scores = torch.einsum("blnhc,hc->blnh", norm(U.float()), query.float())  # (b, l, n, h)
        scores = scores - age[:, None] * decay.float()  # (b, l, n, h)
        weights = n * scores.softmax(dim=2)  # (b, l, n, h): every weight is exactly 1 at q = 0 and lambda = 0
    read = (weights[..., None].to(U.dtype) * U).sum(dim=2)  # (b, l, h, d_h)
    return E + read.reshape(b, l, d)  # (b, l, d)


def gated_update(update: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """A pass's update scaled per token and channel head by 2 sigmoid(w . RMSNorm(u) + b), one at w = 0 and b = 0."""
    # update: (b, l, d); weight: (h, d_h); bias: (h,); returns (b, l, d).
    b, l, d = update.shape
    U = update.view(b, l, weight.shape[0], -1)  # (b, l, h, d_h)
    with torch.autocast(update.device.type, enabled=False):
        gates = 2 * torch.sigmoid(torch.einsum("blhc,hc->blh", norm(U.float()), weight.float()) + bias.float())  # (b, l, h)
    return (U * gates[..., None].to(U.dtype)).reshape(b, l, d)  # (b, l, d)


def valid_tokens(context: ForwardContext, X: Tensor) -> Tensor:
    """The real tokens of an evaluation forward: its padding mask, or every token when it has none."""
    # X: (b, l, d); returns (b, l) bool.
    mask = context.dense_mask
    if mask is not None and mask.shape[2] == 1:  # (b, 1, 1, l), the padding mask the evaluator passes
        return mask[:, 0, 0, :]  # (b, l)
    return torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)  # (b, l)


class Loop(Strategy):
    family = "loop"
    Config = LoopConfig
    config: LoopConfig

    def __init__(self, config: LoopConfig) -> None:
        super().__init__(config)
        # The block whose MLP update ends the prelude (0) or pass j (j): where the pass boundaries act.
        self.boundaries = {config.prelude - 1 + passes_done * config.core: passes_done for passes_done in range(config.passes + 1)}
        # The evaluation's dial and exit, from the config; a dial study sets these on the strategy.
        self.eval_passes = config.eval_passes or config.passes
        self.exit = config.exit
        self.exit_threshold = config.exit_threshold

    def conflicts(self, strategies: Strategies) -> list[str]:
        if not self.active:
            return []
        blocks, reasons = strategies.shape.config.blocks, []
        if self.config.coda(blocks) < 0:
            reasons.append(f"prelude {self.config.prelude} + core {self.config.core} x passes {self.config.passes} exceeds {blocks} blocks")
        if strategies.stream.variant != "single":
            reasons.append("the pass boundaries act on the stream itself, so the loop needs stream.variant 'single'")
        return reasons

    def overrides(self, hook: str) -> bool:
        uses = {"updated_stream"}
        if self.config.residual_lambda:
            uses.add("sublayer_output")
        if self.config.readout_weight:
            uses.add("auxiliary_loss")
        return self.active and hook in uses

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if not self.active:
            return
        config = self.config
        holder = nn.Module()
        if config.inject == "learned" and config.passes > 1:
            shares = torch.tensor([config.share(start) for start in range(2, config.passes + 1)])  # (passes - 1,)
            holder.inject_logits = nn.Parameter(torch.logit(shares)[:, None].repeat(1, model.strategies.shape.config.hidden))  # (passes - 1, d)
        heads = model.strategies.attention.config.heads
        head_width = model.strategies.shape.config.hidden // heads  # the attention's heads split the channels
        if config.inject == "read":
            holder.read_queries = nn.Parameter(torch.zeros(config.passes, heads, head_width))  # (passes, h, d_h)
            holder.read_decay = nn.Parameter(torch.full((heads,), config.read_decay))  # (h,)
        if config.write_gate:
            holder.write_weights = nn.Parameter(torch.zeros(config.passes, heads, head_width))  # (passes, h, d_h)
            holder.write_bias = nn.Parameter(torch.zeros(config.passes, heads))  # (passes, h)
        # Evaluation tokens: [0] every real token, [j] the real tokens pass j updated; a dial study zeroes and reads it.
        holder.register_buffer("pass_tokens", torch.zeros(config.passes + 1, dtype=torch.long), persistent=False)  # (passes + 1,)
        model.extensions[self.family] = holder
        self.holder = holder

    def initialize(self, model: CandidateModel) -> None:
        if not self.active:
            return
        self.model = model  # the head an exit reads, in evaluation
        if self.variant != "tied":
            return
        for position in range(len(model.blocks)):
            source = self.config.source(position)
            if source != position:
                tie_matrices(model.blocks[position], model.blocks[source])

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if not self.active:
            return []
        names = ("inject_logits", "read_queries", "read_decay", "write_weights", "write_bias")
        parameters = [getattr(self.holder, name) for name in names if hasattr(self.holder, name)]
        if not parameters:
            return []
        return [ParameterGroup("loop_injection", parameters, "adam", model.strategies.optimizer.config.scalar_lr)]

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d); the scale is a constant, so the compiled graph bakes it in.
        if not self.config.is_core(index):
            return Y  # (b, l, d)
        return Y * self.config.residual_scale  # (b, l, d)

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream after block `index`'s attention or MLP update.
        if sublayer != "mlp":
            return X  # (b, l, d)
        if self.config.is_core(index) and RUNNING in context.stored:
            X = torch.where(context.stored[RUNNING], X, context.stored[HELD])  # (b, l, d): a stopped token keeps its state
        passes_done = self.boundaries.get(index)
        if passes_done is None:
            return X  # (b, l, d)
        if passes_done == 0:
            return self._enter(X, context)  # (b, l, d)
        return self._end_pass(passes_done, X, context)  # (b, l, d)

    def _enter(self, X: Tensor, context: ForwardContext) -> Tensor:
        """The prelude's end: keep e, and in training draw each window's depth; pass 1 updates every token."""
        # X: (b, l, d), the prelude's output; returns it unchanged.
        context.stored[PRELUDE_OUTPUT] = X  # (b, l, d)
        context.stored[HELD] = X  # (b, l, d)
        context.stored[PASS_INPUT] = X  # (b, l, d)
        context.stored[UPDATES] = []
        if self.holder.training and self.config.sample:
            context.stored[DEPTH] = torch.randint(1, self.config.passes + 1, (X.shape[0], 1, 1), device=X.device)  # (b, 1, 1)
        if not self.holder.training:
            real = valid_tokens(context, X).sum()  # ()
            self.holder.pass_tokens[0] += real
            self.holder.pass_tokens[1] += real
        return X  # (b, l, d)

    def _end_pass(self, passes_done: int, X: Tensor, context: ForwardContext) -> Tensor:
        """The end of pass `passes_done`: keep a readout, decide who runs the next pass, cut the gradient, re-inject."""
        # X: (b, l, d), the stream after the pass, stopped tokens already restored; returns the next pass's start.
        config = self.config
        if config.inject == "read" or config.write_gate:
            X = self._written(passes_done, X, context)  # (b, l, d)
        if self.holder.training and context.labels is not None and passes_done in config.readout_passes:
            context.stored[f"{PASS_END}{passes_done}"] = X  # (b, l, d)
        if passes_done == config.passes:
            return X  # (b, l, d), the coda's input
        start = passes_done + 1
        running = self._running(start, X, context)  # (b, l or 1, 1) bool, or None when every token runs
        if config.grad_passes and passes_done == config.passes - config.grad_passes:
            X = X.detach()  # (b, l, d): the passes so far skip their backward; e keeps its gradient
        context.stored[HELD] = X  # (b, l, d)
        injected = self._inject(X, context.stored[PRELUDE_OUTPUT], start)  # (b, l, d)
        if running is None:
            context.stored.pop(RUNNING, None)
            following = injected  # (b, l, d)
        else:
            context.stored[RUNNING] = running
            following = torch.where(running, injected, X)  # (b, l, d): a stopped token is not re-injected
        context.stored[PASS_INPUT] = following
        return following  # (b, l, d)

    def _written(self, passes_done: int, X: Tensor, context: ForwardContext) -> Tensor:
        """The pass's update gated and the stream read over every update so far, for the tokens the pass updated."""
        # X: (b, l, d), the stream after pass `passes_done`; returns it rewritten (b, l, d).
        start = context.stored[PASS_INPUT]  # (b, l, d)
        update = X - start  # (b, l, d), zero for a token the pass did not update
        if self.config.write_gate:
            update = gated_update(update, self.holder.write_weights[passes_done - 1], self.holder.write_bias[passes_done - 1])  # (b, l, d)
        if self.config.inject == "read":
            context.stored[UPDATES] = [*context.stored[UPDATES], update]
            written = read_updates(context.stored[PRELUDE_OUTPUT], context.stored[UPDATES],
                                   self.holder.read_queries[passes_done - 1], self.holder.read_decay)  # (b, l, d)
        else:
            written = start + update  # (b, l, d)
        if RUNNING not in context.stored:
            return written  # (b, l, d)
        return torch.where(context.stored[RUNNING], written, X)  # (b, l, d): a token the pass skipped keeps its state

    def _inject(self, X: Tensor, E: Tensor, start: int) -> Tensor:
        # X, E: (b, l, d); returns the stream entering pass `start` (b, l, d).
        if self.config.inject in ("none", "read"):  # the read keeps e on its own path instead of mixing it in
            return X  # (b, l, d)
        if self.config.inject == "learned":
            share = torch.sigmoid(self.holder.inject_logits[start - 2])  # (d,)
            return reinjected(X, E, share.to(X.dtype))  # (b, l, d)
        return reinjected(X, E, self.config.share(start))  # (b, l, d)

    def _running(self, start: int, X: Tensor, context: ForwardContext) -> Tensor | None:
        """The tokens pass `start` updates: in training a window's sampled depth; in evaluation the dial and the exits."""
        # X: (b, l, d), the stream after pass start - 1; returns (b, l or 1, 1) bool, or None for every token.
        if self.holder.training:
            return context.stored[DEPTH] >= start if self.config.sample else None  # (b, 1, 1)
        if start > self.eval_passes:
            running = torch.zeros(X.shape[0], 1, 1, dtype=torch.bool, device=X.device)  # (b, 1, 1): the dial stops here
        else:
            stopped = self._stopped(X, context)  # (b, l, 1) bool or None
            if stopped is not None and STOPPED in context.stored:
                stopped = stopped | context.stored[STOPPED]  # (b, l, 1): a stopped token stays stopped
            if stopped is None:
                running = None
            else:
                context.stored[STOPPED] = stopped
                running = ~stopped  # (b, l, 1)
        real = valid_tokens(context, X)  # (b, l)
        self.holder.pass_tokens[start] += real.sum() if running is None else (running[..., 0] & real).sum()
        return running  # (b, l or 1, 1) bool, or None

    def _stopped(self, X: Tensor, context: ForwardContext) -> Tensor | None:
        """The tokens the exit stops after this pass, or None when no exit is set."""
        # X: (b, l, d), the stream after the pass; returns (b, l, 1) bool.
        if self.exit == "none":
            return None
        if self.exit == "confidence":
            logits = self.model.logits(X)[..., benchmark.ESM2_RESIDUE_COLUMNS.to(X.device)]  # (b, l, c)
            confidence = logits.float().softmax(dim=-1).amax(dim=-1, keepdim=True)  # (b, l, 1)
            return confidence >= self.exit_threshold  # (b, l, 1)
        moved = (X - context.stored[HELD]).float().norm(dim=-1, keepdim=True)  # (b, l, 1)
        return moved < self.exit_threshold * X.float().norm(dim=-1, keepdim=True)  # (b, l, 1)

    def auxiliary_loss(self, model: CandidateModel, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        """The readouts at the ends of the passes, each the record's summed masked loss, averaged over the readouts."""
        # states: (b, l, d) each; labels: (b, l); returns ().
        readouts = [context.stored[f"{PASS_END}{passes_done}"] for passes_done in self.config.readout_passes]
        total = sum(benchmark.masked_cross_entropy(model.logits(X), labels) for X in readouts)  # ()
        return self.config.readout_weight * total / len(readouts)  # ()
