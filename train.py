"""The candidate: a skeleton every genome builds on, the training loop and the result, around the fixed benchmark.

The skeleton is a pre-norm transformer of 36 blocks on a 768-wide stream, each block one attention sublayer and a
ReLU-squared MLP of 2,048, between an embedding and an untied head over the ESM2 alphabet, trained with bf16
autocast, `torch.compile`, flex attention over packed windows with a document mask, the collapse-free cross entropy on
masked residues, and validation on `val_select` every 90 s of clock to select the weights. Everything else comes from
the genome (`--genome`, a JSON file; `speedrunning_plms.strategies`): the attention kernel and its positional prior,
the residual form and stream, skips, tables, gates, the head's cap, the optimizer and its schedule, the batch, the
window curriculum, weight averaging and the backprop-free pieces. Without a genome every family takes its default,
which is the record (022).

What the benchmark fixes and this file may not touch: the data and its corruption, the clock and its caps, the
evaluator, the shape rule, the result schema (`speedrunning_plms.benchmark`).

    python train.py --data-dir /data/afdb_v6 --output-dir runs/029_x --name 029_x [--genome genomes/x.json]

Symbols: b windows or proteins, l tokens, d = 768, f = 2,048, v = 33 tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
import numpy as np
import torch

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from speedrunning_plms import benchmark
from speedrunning_plms.strategies import Genome, Strategies, default_genome, genome_document, genome_sha256, load_genome, sources_sha256
from speedrunning_plms.strategies.activation import Activation
from speedrunning_plms.strategies.attention import Attention, document_dense_mask
from speedrunning_plms.strategies.strategy import ForwardContext, StepState, Strategy, norm


EXPERIMENT_GROUP = "2026-10-28_afdb_v6_speedrun"
WANDB_PROJECT = "speedrunning-plms"
EVAL_INTERVAL_SECONDS = 90.0  # about 4 s each on the GH200, under the 60 s cap with room
EVAL_TARGET_SECONDS = 50.0  # the in-run evals' total a slower genome's spacing aims at, under the 60 s cap
WARMUP_STEPS = 3  # compile warmup on synthetic windows a shape, outside the clock
FINAL_SPLITS = ("val_select", "validation", "test")


class MLP(nn.Module):
    def __init__(self, index: int, d: int, f: int, activation: Activation, hidden_hooks: list[Strategy]) -> None:
        super().__init__()
        self.index = index
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)
        self.activation = activation
        self.hidden_hooks = hidden_hooks

    def forward(self, X: Tensor) -> Tensor:
        # X: (b, l, d)
        H = self.activation.apply(self.index, self.up(X))  # (b, l, f), ReLU squared in the record
        for hook in self.hidden_hooks:
            H = hook.mlp_hidden(self.index, H)  # (b, l, f)
        return self.down(H)  # (b, l, d)


class Block(nn.Module):
    def __init__(self, index: int, attention: Attention, strategies: Strategies) -> None:
        super().__init__()
        self.index = index
        self.attention = attention
        self.mlp = MLP(index, benchmark.HIDDEN, benchmark.INTERMEDIATE, strategies.activation, strategies.hooked("mlp_hidden"))
        self.residual = strategies.residual
        self.normalized_hooks = strategies.hooked("normalized_input")
        self.output_hooks = strategies.hooked("sublayer_output")
        self.stream_hooks = strategies.hooked("updated_stream")

    def sublayer_input(self, sublayer: str, X: Tensor) -> Tensor:
        # X: (b, l, d), the stream; returns the sublayer's normalized input (b, l, d).
        H = self.residual.sublayer_input(X)  # (b, l, d)
        for hook in self.normalized_hooks:
            H = hook.normalized_input(self.index, sublayer, H)  # (b, l, d)
        return H  # (b, l, d)

    def forward(self, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d)
        A = self.attention(self.sublayer_input("attention", X), context)  # (b, l, d)
        for hook in self.output_hooks:
            A = hook.sublayer_output(self.index, "attention", A)  # (b, l, d)
        X = self.residual.update(self.index, "attention", X, A)  # (b, l, d)
        for hook in self.stream_hooks:
            X = hook.updated_stream(self.index, "attention", X, context)  # (b, l, d)
        M = self.mlp(self.sublayer_input("mlp", X))  # (b, l, d)
        for hook in self.output_hooks:
            M = hook.sublayer_output(self.index, "mlp", M)  # (b, l, d)
        X = self.residual.update(self.index, "mlp", X, M)  # (b, l, d)
        for hook in self.stream_hooks:
            X = hook.updated_stream(self.index, "mlp", X, context)  # (b, l, d)
        return X  # (b, l, d)


class Candidate(nn.Module):
    def __init__(self, strategies: Strategies, max_tokens: int) -> None:
        super().__init__()
        d = benchmark.HIDDEN
        self.strategies = strategies
        self.extensions = nn.ModuleDict()  # what the strategies add, by family
        self.embedding = nn.Embedding(benchmark.VOCAB_SIZE, d)
        for strategy in strategies:
            strategy.build(self, max_tokens)
        self.blocks = nn.ModuleList(Block(index, strategies.attention.attention(self, index), strategies)
                                    for index in range(benchmark.BLOCKS))
        self.head = nn.Linear(d, benchmark.VOCAB_SIZE, bias=False)
        self.residue_columns = benchmark.ESM2_RESIDUE_COLUMNS  # (c,), the evaluator's view of this vocabulary
        self.embed_hooks = strategies.hooked("embed")
        self.input_hooks = strategies.hooked("block_input")
        self.head_input_hooks = strategies.hooked("head_input")
        self.logit_hooks = strategies.hooked("logits")
        self.auxiliary_hooks = strategies.hooked("auxiliary_loss")
        self.reset_parameters()
        for strategy in strategies:
            strategy.initialize(self)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)  # (v, d)
        nn.init.normal_(self.head.weight, std=0.02)  # (v, d)
        for block in self.blocks:
            for linear in (block.attention.query, block.attention.key, block.attention.value, block.mlp.up):
                nn.init.normal_(linear.weight, std=0.02)  # (d or 2d, d) or (f, d)
            nn.init.zeros_(block.attention.output.weight)  # (d, d): a block starts as the identity
            nn.init.zeros_(block.mlp.down.weight)  # (d, f)

    def _states(self, input_ids: Tensor, context: ForwardContext) -> tuple[list[Tensor], Tensor]:
        # input_ids: (b, l); returns the embedding and every block's output, each (b, l, d), and the head's input (b, l, d).
        X = self.embedding(input_ids)  # (b, l, d)
        for hook in self.embed_hooks:
            X = hook.embed(self, input_ids, X)  # (b, l, d)
        mixer = self.strategies.stream
        stream = mixer.begin(X)
        states = [X]
        for index, block in enumerate(self.blocks):
            X = mixer.read(stream, index)  # (b, l, d)
            for hook in self.input_hooks:
                X = hook.block_input(self, index, X, states)  # (b, l, d)
            stream = mixer.write(stream, index, X, block(X, context))
            states.append(mixer.view(stream))  # (b, l, d)
        return states, mixer.end(stream)  # 37 of (b, l, d), (b, l, d)

    def head_features(self, X: Tensor) -> Tensor:
        """What the 33-way head reads: the normalized state, through the strategies' head-input hooks."""
        # X: (b, l, d); returns (b, l, d).
        H = norm(X)  # (b, l, d)
        for hook in self.head_input_hooks:
            H = hook.head_input(self, H)  # (b, l, d)
        return H  # (b, l, d)

    def logits(self, X: Tensor) -> Tensor:
        """The head on a state, through the strategies' logit hooks."""
        # X: (b, l, d); returns (b, l, v).
        logits = self.head(self.head_features(X))  # (b, l, v)
        for hook in self.logit_hooks:
            logits = hook.logits(self, logits)  # (b, l, v)
        return logits  # (b, l, v)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, block_masks: tuple[BlockMask, ...] | None = None,
                labels: Tensor | None = None, dense_mask: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Logits over the ESM2 alphabet; `attention_mask` (b, l) bool is the padding mask the evaluator passes. With
        `labels`, training also returns the strategies' summed auxiliary loss."""
        # input_ids, attention_mask, labels: (b, l); dense_mask: (b, 1, l or 1, l); returns (b, l, v), and () with labels.
        if dense_mask is None and block_masks is None and attention_mask is not None:
            dense_mask = attention_mask[:, None, None, :]  # (b, 1, 1, l)
        context = ForwardContext(input_ids, block_masks, dense_mask, labels=labels)
        states, final = self._states(input_ids, context)
        logits = self.logits(final)  # (b, l, v)
        if labels is None:
            return logits  # (b, l, v)
        auxiliary = sum(hook.auxiliary_loss(self, states, labels, context) for hook in self.auxiliary_hooks)  # ()
        return logits, auxiliary  # (b, l, v), ()

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """The 37 states the 3Di probe reads."""
        # input_ids, attention_mask: (b, l); each state (b, l, d).
        return self._states(input_ids, ForwardContext(input_ids, None, attention_mask[:, None, None, :]))[0]  # 37 of (b, l, d)


def build_candidate(genome: Genome, max_tokens: int) -> Candidate:
    """The candidate a genome describes; the probe rebuilds a run's model through this."""
    return Candidate(Strategies(genome), max_tokens)


def windows(steps: Iterator[np.ndarray], step_tokens: int, window_of_step: Callable[[], int]) -> Iterator[Tensor]:
    """Each step's block of the batch family's reader as windows; the record reads consecutive blocks of the shuffled
    stream (`batch.contiguous_steps`)."""
    # Each yield: (step_tokens / window, window) uint8 on the CPU, the window read from `window_of_step` at each step.
    for chunk in steps:
        window = window_of_step()
        yield torch.from_numpy(chunk).view(step_tokens // window, window)  # (b, l)


def train_step(model: nn.Module, corrupted: Tensor, labels: Tensor, block_masks: tuple[BlockMask, ...] | None,
               dense_mask: Tensor | None, device: torch.device, auxiliary: bool, loss_hooks: list[Strategy]) -> tuple[Tensor, Tensor]:
    """One forward and backward; returns the summed masked loss (without auxiliary terms or the loss hooks' changes)
    and the masked count. The loss hooks turn that loss into the objective the backward differentiates."""
    # corrupted, labels: (b, l); dense_mask: (b, 1, l, l) or None.
    extra: Tensor | None = None
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if block_masks is not None and auxiliary:
            logits, extra = model(corrupted, block_masks=block_masks, labels=labels)  # (b, l, v), ()
        elif block_masks is not None:
            logits = model(corrupted, block_masks=block_masks)  # (b, l, v)
        else:
            assert dense_mask is not None
            logits = _dense_forward(model, corrupted, dense_mask)  # (b, l, v)
    count = (labels != benchmark.IGNORE_LABEL).sum()  # ()
    loss = benchmark.masked_cross_entropy(logits, labels)  # ()
    objective = loss  # ()
    for hook in loss_hooks:
        objective = hook.training_loss(logits, labels, objective)  # ()
    total = objective if extra is None else objective + extra  # ()
    (total / count.clamp(min=1)).backward()
    return loss.detach(), count  # (), ()


def _dense_forward(model: nn.Module, corrupted: Tensor, dense_mask: Tensor) -> Tensor:
    """The CPU smoke's training forward, with the record's arithmetic: the head without logit hooks or auxiliary
    losses (the record's CPU path skipped its softcap; kept so a CPU smoke of the default genome equals 022's). The
    head-input hooks run, since they are part of the head."""
    # corrupted: (b, l); dense_mask: (b, 1, l, l). A compiled module keeps the original's methods under `_orig_mod`.
    candidate = getattr(model, "_orig_mod", model)
    final = candidate._states(corrupted, ForwardContext(corrupted, None, dense_mask))[1]  # (b, l, d)
    return candidate.head(candidate.head_features(final))  # (b, l, v)


def eval_interval(eval_seconds: float, budget: float) -> float:
    """The spacing of in-run evals after the first: the record's 90 s, or wider when one eval costs more, so the evals
    left and the final one stay inside EVAL_TARGET_SECONDS. A record eval costs about 4 s and keeps 90 s; a Dual
    Triangle eval cost 12 s in its pilot, and seven of them would break the 60 s cap."""
    left = math.floor((EVAL_TARGET_SECONDS - 2 * eval_seconds) / eval_seconds)  # after the first and before the final
    if left < 0:
        return math.inf
    return max(EVAL_INTERVAL_SECONDS, (budget - EVAL_INTERVAL_SECONDS) / (left + 1))


def candidate_digest(genome: Genome) -> tuple[str, dict[str, str]]:
    """What defines a candidate: this file's bytes, the strategy modules' sources and the genome; one digest of all."""
    parts = {"train.py": benchmark.candidate_sha256(Path(__file__)), "strategies": sources_sha256(),
             "genome": genome_sha256(genome)}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest(), parts


class NullRun:
    """What `--no-wandb` logs to: nothing."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def finish(self, status: str, summary: dict[str, float]) -> None:
        pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the run's name, also its W&B name")
    parser.add_argument("--genome", type=Path, default=None, help="a genome JSON; without one, the record's genome")
    parser.add_argument("--hypothesis", default="", help="one sentence: what this candidate changes and why it should help")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=benchmark.BUDGET_SECONDS, help="training seconds on the clock")
    parser.add_argument("--max-steps", type=int, default=None, help="a smoke run: stop after this many applied steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="time the first steps with foundry.pilot and go on")
    parser.add_argument("--eval-tokens-per-batch", type=int, default=131_072)  # outside the clock; fewer, larger batches cut the final evals
    parser.add_argument("--skip-final-evals", action="store_true", help="a smoke run: no full validation and test")
    parser.add_argument("--final-splits", nargs="+", choices=FINAL_SPLITS, default=list(FINAL_SPLITS),
                        help="the splits the selected weights are scored on; a short search rung needs only val_select")
    parser.add_argument("--window", type=int, default=None, help="override the genome's window, for a smoke run")
    parser.add_argument("--windows-per-step", type=int, default=None, help="override the genome's windows a step, for a smoke run")
    return parser.parse_args()


def genome_from(arguments: argparse.Namespace) -> Genome:
    """The run's genome: the file's, or the record's, with the smoke's batch overrides."""
    genome = load_genome(arguments.genome) if arguments.genome is not None else default_genome()
    overrides = {name: value for name, value in (("window", arguments.window), ("windows_per_step", arguments.windows_per_step))
                 if value is not None}
    return {**genome, "batch": {**genome["batch"], **overrides}}


def main() -> None:
    arguments = parse_arguments()
    genome = genome_from(arguments)
    strategies = Strategies(genome)
    batch, curriculum, schedule = strategies.batch, strategies.curriculum, strategies.schedule
    device = torch.device(arguments.device)
    torch.manual_seed(arguments.seed)
    output_dir = arguments.output_dir
    if (output_dir / "result.json").exists():
        raise FileExistsError(f"{output_dir} already holds a result")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(UTC)

    manifest = benchmark.load_manifest(arguments.data_dir)
    data_id = benchmark.benchmark_id(manifest)
    stream = benchmark.load_train(arguments.data_dir)
    val_select = benchmark.load_held_out(arguments.data_dir, "val_select")
    max_tokens = max(manifest["max_tokens"].values())
    # The reading order and the whole-protein packing, once on the CPU before the warmup; the record needs neither.
    order = curriculum.protein_order(stream.offsets, arguments.data_dir, arguments.seed)  # (n,) or None
    packing = batch.packing(stream.offsets, order, arguments.seed) if batch.variant == "whole" else None
    if packing is not None:
        print(json.dumps({"packing": {"windows": packing.windows, "padding_share": round(packing.padding_share, 5)}}), flush=True)

    model = Candidate(strategies, max_tokens).to(device)
    architecture = benchmark.check_architecture(model)
    compile_options = {**curriculum.compile_options(), **strategies.kernel.compile_options()}
    compiled = model if arguments.no_compile or device.type == "cpu" else torch.compile(model, **compile_options)
    mark_steps = compiled is not model and strategies.kernel.marks_steps  # CUDA graph trees need each step's start
    use_flex = device.type == "cuda"
    clock = benchmark.Clock(device, budget=arguments.budget)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    auxiliary = bool(strategies.hooked("auxiliary_loss"))
    loss_hooks = strategies.hooked("training_loss")
    backward_hooks, step_hooks = strategies.hooked("after_backward"), strategies.hooked("after_step")
    evaluation_hooks = strategies.hooked("evaluation_weights")

    flat_genome = {f"{family}.{key}": value for family, fields in strategies.genome.items() for key, value in fields.items()}
    config = {**flat_genome, "seed": arguments.seed, "budget": arguments.budget, "name": arguments.name,
              "hypothesis": arguments.hypothesis, "benchmark_id": data_id, "experiment_id": EXPERIMENT_GROUP}
    run: Any
    if arguments.no_wandb:
        run = NullRun()
    else:
        from foundry.logging import Run

        run = Run(output_dir / "wandb", config, project=WANDB_PROJECT, group=EXPERIMENT_GROUP, job_type="train", name=arguments.name)

    def masks_of(corrupted: Tensor) -> tuple[tuple[BlockMask, ...] | None, Tensor | None]:
        # corrupted: (b, l); the kernel's flex masks on the GPU, the document mask (b, 1, l, l) otherwise.
        return (strategies.attention.training_masks(corrupted), None) if use_flex else (None, document_dense_mask(corrupted))  # masks, or (b, 1, l, l)

    def step_state(step: int, progress: float, lr_scale: float, corrupted: Tensor, labels: Tensor,
                   block_masks: tuple[BlockMask, ...] | None, dense_mask: Tensor | None) -> StepState:
        # corrupted, labels: (b, l); dense_mask: (b, 1, l, l) or None.
        def loss_without_grad() -> Tensor:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if block_masks is not None:
                    logits = compiled(corrupted, block_masks=block_masks)  # (b, l, v)
                else:
                    logits = _dense_forward(compiled, corrupted, dense_mask)  # (b, l, v)
            return benchmark.masked_cross_entropy(logits, labels)  # ()

        return StepState(model, step, progress, lr_scale, corrupted, labels, loss_without_grad)

    # Compile warmup on synthetic windows of every shape the run trains on, callbacks included, then back to the fresh
    # initialization and fresh optimizers.
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    optimizers = strategies.optimizer.build_optimizers(model, schedule.config.momentum)
    with clock.warming_up():
        for window in curriculum.windows(batch.config.window):
            for warmup_step in range(WARMUP_STEPS):
                synthetic = torch.randint(4, 24, (batch.step_tokens // window, window), device=device)  # (b, l) residues
                synthetic[:, 0] = benchmark.CLS_ID
                corrupted, labels = benchmark.corrupt(synthetic, generator)
                block_masks, dense_mask = masks_of(corrupted)
                if mark_steps:
                    torch.compiler.cudagraph_mark_step_begin()
                train_step(compiled, corrupted, labels, block_masks, dense_mask, device, auxiliary, loss_hooks)
                state = step_state(warmup_step, 0.0, 1.0, corrupted, labels, block_masks, dense_mask)
                for hook in backward_hooks:
                    hook.after_backward(state)
                for optimizer in optimizers:
                    optimizer.step()
                model.zero_grad(set_to_none=True)
                for hook in step_hooks:
                    hook.after_step(state)
        # The evaluation path's kernels warm once here (its first pass costs four times the later ones).
        model.eval()
        benchmark.evaluate_split(model, val_select, device, tokens_per_batch=arguments.eval_tokens_per_batch, keep_probabilities=False)
        model.train()
    model.load_state_dict(initial_state)
    optimizers = strategies.optimizer.build_optimizers(model, schedule.config.momentum)
    base_lrs = [[group["lr"] for group in optimizer.param_groups] for optimizer in optimizers]
    generator.manual_seed(arguments.seed)

    curve: list[dict[str, Any]] = []
    best_loss, best_step, best_state = math.inf, -1, None
    next_eval = EVAL_INTERVAL_SECONDS

    def evaluate_now(step: int) -> None:
        # Each evaluation option (one in the record) is scored on val_select; the lowest loss is the point and selects.
        nonlocal best_loss, best_step, best_state
        options = max((hook.evaluation_options for hook in evaluation_hooks), default=1)
        scored: list[dict[str, Any]] = []
        with clock.paused():
            for option in range(options):
                with contextlib.ExitStack() as weights:
                    for hook in evaluation_hooks:
                        hook.option = option
                        weights.enter_context(hook.evaluation_weights(model))
                    model.eval()
                    metrics = benchmark.evaluate_split(model, val_select, device, tokens_per_batch=arguments.eval_tokens_per_batch,
                                                       keep_probabilities=False)
                    model.train()
                    if metrics["loss"] < best_loss:  # inside the evaluation weights, which are what was scored
                        best_loss, best_step = metrics["loss"], step
                        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                scored.append(metrics)
        metrics = min(scored, key=lambda option_metrics: option_metrics["loss"])
        point = {"step": step, "train_seconds": clock.train_seconds, "loss": metrics["loss"], "accuracy": metrics["accuracy"]}
        if options > 1:
            point["option_losses"] = [option_metrics["loss"] for option_metrics in scored]
        curve.append(point)
        run.log({"val/select_loss": metrics["loss"], "val/select_accuracy": metrics["accuracy"]}, step=step)
        print(json.dumps({"eval": point}), flush=True)

    def window_of_step() -> int:
        return curriculum.window(batch.config.window, clock.progress)

    batches = windows(batch.steps(stream.tokens, stream.offsets, order, packing, arguments.seed), batch.step_tokens, window_of_step)
    if arguments.pilot:
        from foundry.pilot import Pilot

        pilot = Pilot(steps=20, warmup=3, stop=False, items=batch.step_tokens, item_unit="tokens",
                      budget_seconds=arguments.budget, receipt=output_dir / "pilot.json")
        batches = pilot.watch(batches)

    model.train()
    step = 0
    for window_batch in batches:
        if arguments.max_steps is not None and step >= arguments.max_steps:
            break
        clock.start_step()
        input_ids = window_batch.to(device, non_blocking=True).long()  # (b, l)
        corrupted, labels = benchmark.corrupt(input_ids, generator)  # each (b, l)
        block_masks, dense_mask = masks_of(corrupted)
        if mark_steps:
            torch.compiler.cudagraph_mark_step_begin()
        loss, count = train_step(compiled, corrupted, labels, block_masks, dense_mask, device, auxiliary, loss_hooks)
        if clock.expired():
            model.zero_grad(set_to_none=True)
            break
        progress = clock.progress
        scale = schedule.lr_scale(progress, cooldown=not strategies.optimizer.schedule_free)
        state = step_state(step, progress, scale, corrupted, labels, block_masks, dense_mask)
        for hook in backward_hooks:
            hook.after_backward(state)
        for optimizer, lrs in zip(optimizers, base_lrs, strict=True):
            for group, base_lr in zip(optimizer.param_groups, lrs, strict=True):
                group["lr"] = base_lr * scale
                if "momentum" in group:
                    group["momentum"] = schedule.momentum(progress)
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        for hook in step_hooks:
            hook.after_step(state)
        clock.applied()
        step += 1
        if step % 50 == 0 or step == 1:
            run.log({"train/loss": float(loss / count.clamp(min=1)), "train/lr_scale": scale,
                     "cost/train_seconds": clock.train_seconds}, step=step)
            print(json.dumps({"step": step, "loss": float(loss / count.clamp(min=1)), "train_seconds": round(clock.train_seconds, 1)}), flush=True)
        if clock.train_seconds >= next_eval:
            evaluate_now(step)
            next_eval += eval_interval(clock.eval_seconds / clock.eval_count, arguments.budget)
    if not curve or curve[-1]["step"] != step:
        evaluate_now(step)
    if best_state is None:
        raise RuntimeError("no validation point was taken")

    model.load_state_dict(best_state)
    model.eval()
    torch.save({name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor for name, tensor in best_state.items()},
               output_dir / "best.pt")
    metrics: dict[str, dict[str, Any]] = {}
    final_eval_seconds = 0.0
    if not arguments.skip_final_evals:
        for split in arguments.final_splits:
            held_out = val_select if split == "val_select" else benchmark.load_held_out(arguments.data_dir, split)
            started = time.perf_counter()
            metrics[split] = benchmark.evaluate_split(model, held_out, device, tokens_per_batch=arguments.eval_tokens_per_batch)
            final_eval_seconds += time.perf_counter() - started
            print(json.dumps({split: {key: metrics[split][key] for key in ("loss", "accuracy", "mcc")}}), flush=True)
    clock_receipt = clock.receipt()
    evaluator = benchmark.evaluator_sha256()
    digest, parts = candidate_digest(strategies.genome)
    run_result = {
        "schema_version": 2, "name": arguments.name, "hypothesis": arguments.hypothesis, "seed": arguments.seed,
        "status": "completed" if clock_receipt["valid"] else "invalid",
        "benchmark_id": data_id, "evaluator_sha256": evaluator,
        "comparison_key": benchmark.comparison_key(data_id, evaluator, clock_receipt),
        "candidate_sha256": digest, "candidate_parts": parts, "candidate_file": Path(__file__).name,
        "genome": strategies.genome, "genome_changes": genome_document(strategies.genome)["families"],
        "max_steps": arguments.max_steps, "compiled": compiled is not model,
        "clock": clock_receipt, "architecture": architecture, "curve": curve,
        "selected_step": best_step, "selected_val_select_loss": best_loss, "metrics": metrics,
        "final_eval_seconds": final_eval_seconds, "environment": benchmark.environment(device),
        "wandb": None if arguments.no_wandb else run.run.url,
        "started_utc": started_utc.isoformat(), "finished_utc": datetime.now(UTC).isoformat(),
    }
    benchmark.write_result(output_dir, run_result)
    summary = {f"{split}/loss": values["loss"] for split, values in metrics.items()}
    run.finish(status="completed", summary={**summary, "cost/steps": clock_receipt["steps"]})
    print(json.dumps(benchmark.ledger_row(run_result)), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "torchinductor"))
    main()
