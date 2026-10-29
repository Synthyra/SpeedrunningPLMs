"""The candidate: everything a run may change, in one file, around the fixed benchmark.

The baseline recipe (the first incumbent): a pre-norm transformer of 36 blocks on a 768-wide stream, 12 heads of 64
with QK-norm and rotary positions, a ReLU-squared MLP of 2,048, zero-initialized output projections, an untied head over
the ESM2 alphabet, bf16 autocast, `torch.compile`, flex attention with a document mask over packed 1,024-token windows,
Muon with Polar Express on the block matrices and AdamW on the embedding and head, a clock-fraction schedule, and the
collapse-free cross entropy on masked residues. Validation on `val_select` every minute of clock selects the weights.

What the benchmark fixes and this file may not touch: the data and its corruption, the clock and its caps, the
evaluator, the shape rule, the result schema (`speedrunning_plms.benchmark`).

    python train.py --data-dir /data/afdb_v6 --output-dir runs/000_baseline --name 000_baseline

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, f = 2,048, v = 33 tokens, c = 20
classes, m masked residues.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from speedrunning_plms import benchmark
from speedrunning_plms.optim.muon import POLAR_EXPRESS_COEFFICIENTS


EXPERIMENT_GROUP = "2026-10-28_afdb_v6_speedrun"
WANDB_PROJECT = "speedrunning-plms"


@dataclass(frozen=True)
class Recipe:
    """The editable hyperparameters."""

    window: int = 1024  # tokens a window
    windows_per_step: int = 64  # 65,536 tokens a step
    heads: int = 12
    rope_base: float = 10_000.0
    muon_lr: float = 0.04
    muon_momentum: float = 0.95
    muon_momentum_start: float = 0.85
    muon_momentum_warmup: float = 0.15  # share of the budget
    embed_lr: float = 1e-3
    head_lr: float = 3e-4
    logit_softcap: float = 30.0  # tanh cap on the head's logits, modded-nanogpt records 9 and 18
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_fraction: float = 0.02
    cooldown_fraction: float = 0.6
    final_lr_ratio: float = 0.0
    eval_interval_seconds: float = 90.0  # about 4 s each on the GH200, under the 60 s cap with room
    warmup_steps: int = 3  # compile warmup on synthetic windows, outside the clock


def norm(X: Tensor) -> Tensor:
    # X: (..., k); RMS normalization over the last dimension, no affine.
    return F.rms_norm(X, (X.shape[-1],))


class Rotary(nn.Module):
    def __init__(self, d_h: int, base: float, max_tokens: int) -> None:
        super().__init__()
        inverse_frequency = base ** (-torch.arange(0, d_h, 2).float() / d_h)  # (d_h / 2,)
        angles = torch.outer(torch.arange(max_tokens).float(), inverse_frequency)  # (max_tokens, d_h / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_tokens, d_h / 2)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (max_tokens, d_h / 2)

    def forward(self, X: Tensor) -> Tensor:
        # X: (b, l, h, d_h); rotate the two halves by position.
        l = X.shape[1]
        cos, sin = self.cos[:l, None, :], self.sin[:l, None, :]  # each (l, 1, d_h / 2)
        first, second = X.float().chunk(2, dim=-1)  # each (b, l, h, d_h / 2)
        rotated = torch.cat((first * cos + second * sin, second * cos - first * sin), dim=-1)  # (b, l, h, d_h)
        return rotated.type_as(X)  # (b, l, h, d_h)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, rotary: Rotary) -> None:
        super().__init__()
        self.heads = heads  # h
        self.d_h = d // heads
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.rotary = rotary
        slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)  # (h,), ALiBi's geometric slopes
        self.register_buffer("slopes", slopes, persistent=False)

    def score_mod(self) -> Any:
        slopes = self.slopes

        def local_bias(score: Tensor, batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
            return score - slopes[head] * (query - key).abs()

        return local_bias

    def forward(self, X: Tensor, block_mask: BlockMask | None, dense_mask: Tensor | None) -> Tensor:
        # X: (b, l, d); block_mask for flex attention, or dense_mask (b, 1, l or 1, l) bool for SDPA.
        b, l, d = X.shape
        Q = norm(self.query(X).view(b, l, self.heads, self.d_h))  # (b, l, h, d_h)
        K = norm(self.key(X).view(b, l, self.heads, self.d_h))  # (b, l, h, d_h)
        V = self.value(X).view(b, l, self.heads, self.d_h)  # (b, l, h, d_h)
        Q, K = self.rotary(Q), self.rotary(K)  # each (b, l, h, d_h)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # each (b, h, l, d_h)
        if block_mask is not None:
            Y = flex_attention(Q, K, V, score_mod=self.score_mod(), block_mask=block_mask)  # (b, h, l, d_h)
        else:
            positions = torch.arange(l, device=X.device)  # (l,)
            distance = (positions[:, None] - positions[None, :]).abs().to(Q.dtype)  # (l, l)
            bias = -self.slopes.to(Q.dtype).view(1, self.heads, 1, 1) * distance  # (1, h, l, l)
            attention_mask = bias.masked_fill(~dense_mask, float("-inf"))  # (b, h, l, l)
            Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)  # (b, h, l, d_h)
        return self.output(Y.transpose(1, 2).reshape(b, l, d))  # (b, l, d)


class MLP(nn.Module):
    def __init__(self, d: int, f: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        # X: (b, l, d)
        return self.down(F.relu(self.up(X)).square())  # (b, l, d)


class Block(nn.Module):
    def __init__(self, d: int, f: int, heads: int, rotary: Rotary) -> None:
        super().__init__()
        self.attention = Attention(d, heads, rotary)
        self.mlp = MLP(d, f)

    def forward(self, X: Tensor, block_mask: BlockMask | None, dense_mask: Tensor | None) -> Tensor:
        # X: (b, l, d)
        X = X + self.attention(norm(X), block_mask, dense_mask)  # (b, l, d)
        return X + self.mlp(norm(X))  # (b, l, d)


class Candidate(nn.Module):
    def __init__(self, recipe: Recipe, max_tokens: int) -> None:
        super().__init__()
        d, f = benchmark.HIDDEN, benchmark.INTERMEDIATE
        self.embedding = nn.Embedding(benchmark.VOCAB_SIZE, d)
        rotary = Rotary(d // recipe.heads, recipe.rope_base, max_tokens)
        self.blocks = nn.ModuleList(Block(d, f, recipe.heads, rotary) for _ in range(benchmark.BLOCKS))
        self.head = nn.Linear(d, benchmark.VOCAB_SIZE, bias=False)
        self.residue_columns = benchmark.ESM2_RESIDUE_COLUMNS  # (c,), the evaluator's view of this vocabulary
        self.softcap = recipe.logit_softcap
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)  # (v, d)
        nn.init.normal_(self.head.weight, std=0.02)  # (v, d)
        for block in self.blocks:
            for linear in (block.attention.query, block.attention.key, block.attention.value, block.mlp.up):
                nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(block.attention.output.weight)  # (d, d): a block starts as the identity
            nn.init.zeros_(block.mlp.down.weight)  # (d, f)

    def _states(self, input_ids: Tensor, block_mask: BlockMask | None, dense_mask: Tensor | None) -> list[Tensor]:
        # input_ids: (b, l); returns the embedding and every block's output, each (b, l, d).
        X = self.embedding(input_ids)  # (b, l, d)
        states = [X]
        for block in self.blocks:
            X = block(X, block_mask, dense_mask)  # (b, l, d)
            states.append(X)
        return states

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, block_mask: BlockMask | None = None) -> Tensor:
        """Logits over the ESM2 alphabet; `attention_mask` (b, l) bool is the padding mask the evaluator passes."""
        # input_ids: (b, l); returns (b, l, v).
        dense_mask = attention_mask[:, None, None, :] if block_mask is None and attention_mask is not None else None  # (b, 1, 1, l)
        X = self._states(input_ids, block_mask, dense_mask)[-1]  # (b, l, d)
        return self.softcap * torch.tanh(self.head(norm(X)) / self.softcap)  # (b, l, v)

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """The 37 states the 3Di probe reads."""
        # input_ids, attention_mask: (b, l); each state (b, l, d).
        return self._states(input_ids, None, attention_mask[:, None, None, :])


def document_block_mask(input_ids: Tensor) -> BlockMask:
    """Tokens attend within their own protein: the document index is the count of CLS tokens up to each position."""
    # input_ids: (b, l) of packed windows; a window holds whole and cut proteins back to back.
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)  # (b, l)

    def same_document(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        return document[batch, query] == document[batch, key]

    b, l = input_ids.shape
    return create_block_mask(same_document, b, None, l, l, device=input_ids.device, _compile=True)


def document_dense_mask(input_ids: Tensor) -> Tensor:
    """The same rule as a dense bool mask (b, 1, l, l), for devices without flex attention."""
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)  # (b, l)
    return (document[:, :, None] == document[:, None, :])[:, None]  # (b, 1, l, l)


def windows(stream: benchmark.TrainStream, recipe: Recipe, seed: int) -> Iterator[Tensor]:
    """Contiguous windows of the shuffled training stream, each step a block of consecutive windows; wraps at the end."""
    # Each yield: (windows_per_step, window) uint8 on the CPU.
    step_tokens = recipe.windows_per_step * recipe.window
    total = (len(stream.tokens) // step_tokens) * step_tokens
    start = (seed * step_tokens) % total
    while True:
        chunk = np.asarray(stream.tokens[start : start + step_tokens])  # (step_tokens,)
        yield torch.from_numpy(chunk.copy()).view(recipe.windows_per_step, recipe.window)  # (b, l)
        start = (start + step_tokens) % total


def learning_rate_scale(progress: float, recipe: Recipe) -> float:
    """Linear warmup, a constant plateau, then a linear cooldown, by share of the clock."""
    if progress < recipe.warmup_fraction:
        return progress / recipe.warmup_fraction
    if progress > 1 - recipe.cooldown_fraction:
        phase = (progress - (1 - recipe.cooldown_fraction)) / recipe.cooldown_fraction
        return 1 - (1 - recipe.final_lr_ratio) * phase
    return 1.0


def polar_express(G: Tensor, safety_factor: float = 0.02, epsilon: float = 1e-6) -> Tensor:
    """The five-step Polar Express of modded-nanogpt, on a bank of matrices at once."""
    # G: (k, m, n) in fp32; returns (k, m, n), bf16 on CUDA. r = min(m, n), s = max(m, n).
    norms = G.flatten(1).norm(dim=1).view(-1, 1, 1)  # (k, 1, 1)
    X = G / (norms * (1 + safety_factor) + epsilon)  # (k, m, n)
    if G.device.type == "cuda":
        X = X.bfloat16()
    transposed = G.shape[1] > G.shape[2]
    if transposed:
        X = X.mT  # (k, r, s)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        A = X @ X.mT  # (k, r, r)
        B = b * A + c * (A @ A)  # (k, r, r)
        X = a * X + B @ X  # (k, r, s)
    return X.mT if transposed else X  # (k, m, n)


class Muon(torch.optim.Optimizer):
    """Muon with Polar Express, one batched orthogonalization per parameter shape.

    The matrices of one shape (144 of 768 x 768, 36 of 2,048 x 768, 36 of 768 x 2,048) share a stacked momentum buffer
    and one batched Polar Express, so a step launches tens of kernels instead of the thousands a matrix-by-matrix loop
    does. The update of each matrix equals the per-matrix one (checked on CPU to zero difference).
    """

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float, weight_decay: float) -> None:
        parameters = list(params)
        assert all(parameter.ndim == 2 and parameter.is_floating_point() for parameter in parameters)
        super().__init__(parameters, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        for group in self.param_groups:
            by_shape: dict[tuple[int, ...], list[Tensor]] = {}
            for parameter in group["params"]:
                by_shape.setdefault(tuple(parameter.shape), []).append(parameter)
            group["banks"] = [(members, torch.zeros(len(members), *shape, dtype=torch.float32, device=members[0].device))
                              for shape, members in by_shape.items()]  # momentum buffer: (k, m, n)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        assert closure is None
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for members, buffer in group["banks"]:
                if members[0].grad is None:
                    continue
                G = torch.stack([member.grad for member in members]).float()  # (k, m, n)
                buffer.lerp_(G, 1 - momentum)  # (k, m, n)
                U = polar_express(G.lerp(buffer, momentum))  # (k, m, n), Nesterov
                scale = max(1, members[0].shape[0] / members[0].shape[1]) ** 0.5
                if group["weight_decay"]:
                    torch._foreach_mul_(members, 1 - lr * group["weight_decay"])
                torch._foreach_add_(members, list(U.to(members[0].dtype).unbind(0)), alpha=-lr * scale)


def build_optimizers(model: Candidate, recipe: Recipe) -> list[torch.optim.Optimizer]:
    block_matrices = [parameter for block in model.blocks for parameter in block.parameters() if parameter.ndim == 2]
    return [
        Muon(block_matrices, lr=recipe.muon_lr, momentum=recipe.muon_momentum, weight_decay=recipe.weight_decay),
        torch.optim.AdamW([{"params": [model.embedding.weight], "lr": recipe.embed_lr},
                           {"params": [model.head.weight], "lr": recipe.head_lr}],
                          betas=recipe.adam_betas, weight_decay=recipe.weight_decay, fused=model.embedding.weight.is_cuda),
    ]


def train_step(model: nn.Module, corrupted: Tensor, labels: Tensor, block_mask: BlockMask | None, dense_mask: Tensor | None,
               device: torch.device) -> tuple[Tensor, Tensor]:
    """One forward and backward; returns the summed loss and the masked count, both on the device."""
    # corrupted, labels: (b, l).
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(corrupted, block_mask=block_mask) if block_mask is not None else model(corrupted, attention_mask=None) \
            if dense_mask is None else _dense_forward(model, corrupted, dense_mask)  # (b, l, v)
    count = (labels != benchmark.IGNORE_LABEL).sum()  # ()
    loss = benchmark.masked_cross_entropy(logits, labels)  # ()
    (loss / count.clamp(min=1)).backward()
    return loss.detach(), count


def _dense_forward(model: nn.Module, corrupted: Tensor, dense_mask: Tensor) -> Tensor:
    # A compiled module keeps the original's methods under `_orig_mod`.
    candidate = getattr(model, "_orig_mod", model)
    X = candidate._states(corrupted, None, dense_mask)[-1]  # (b, l, d)
    return candidate.head(norm(X))  # (b, l, v)


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
    parser.add_argument("--window", type=int, default=None, help="override the recipe's window, for a smoke run")
    parser.add_argument("--windows-per-step", type=int, default=None, help="override the recipe's windows a step, for a smoke run")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    overrides = {name: value for name, value in (("window", arguments.window), ("windows_per_step", arguments.windows_per_step))
                 if value is not None}
    recipe = replace(Recipe(), **overrides)
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

    model = Candidate(recipe, max_tokens).to(device)
    architecture = benchmark.check_architecture(model)
    compiled = model if arguments.no_compile or device.type == "cpu" else torch.compile(model)
    use_flex = device.type == "cuda"
    clock = benchmark.Clock(device, budget=arguments.budget)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)

    config = {**asdict(recipe), "seed": arguments.seed, "budget": arguments.budget, "name": arguments.name,
              "hypothesis": arguments.hypothesis, "benchmark_id": data_id, "experiment_id": EXPERIMENT_GROUP}
    run: Any
    if arguments.no_wandb:
        run = NullRun()
    else:
        from foundry.logging import Run

        run = Run(output_dir / "wandb", config, project=WANDB_PROJECT, group=EXPERIMENT_GROUP, job_type="train", name=arguments.name)

    # Compile warmup on synthetic windows, then back to the fresh initialization and fresh optimizers.
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    optimizers = build_optimizers(model, recipe)
    with clock.warming_up():
        for _ in range(recipe.warmup_steps):
            synthetic = torch.randint(4, 24, (recipe.windows_per_step, recipe.window), device=device)  # (b, l) residues
            synthetic[:, 0] = benchmark.CLS_ID
            corrupted, labels = benchmark.corrupt(synthetic, generator)
            block_mask = document_block_mask(corrupted) if use_flex else None
            dense_mask = None if use_flex else document_dense_mask(corrupted)
            train_step(compiled, corrupted, labels, block_mask, dense_mask, device)
            for optimizer in optimizers:
                optimizer.step()
            model.zero_grad(set_to_none=True)
        # The evaluation path's kernels warm once here (its first pass costs four times the later ones).
        model.eval()
        benchmark.evaluate_split(model, val_select, device, tokens_per_batch=arguments.eval_tokens_per_batch, keep_probabilities=False)
        model.train()
    model.load_state_dict(initial_state)
    optimizers = build_optimizers(model, recipe)
    base_lrs = [[group["lr"] for group in optimizer.param_groups] for optimizer in optimizers]
    generator.manual_seed(arguments.seed)

    curve: list[dict[str, float | int]] = []
    best_loss, best_step, best_state = math.inf, -1, None
    next_eval = recipe.eval_interval_seconds

    def evaluate_now(step: int) -> None:
        nonlocal best_loss, best_step, best_state
        with clock.paused():
            model.eval()
            metrics = benchmark.evaluate_split(model, val_select, device, tokens_per_batch=arguments.eval_tokens_per_batch,
                                               keep_probabilities=False)
            model.train()
        point = {"step": step, "train_seconds": clock.train_seconds, "loss": metrics["loss"], "accuracy": metrics["accuracy"]}
        curve.append(point)
        run.log({"val/select_loss": metrics["loss"], "val/select_accuracy": metrics["accuracy"]}, step=step)
        print(json.dumps({"eval": point}), flush=True)
        if metrics["loss"] < best_loss:
            best_loss, best_step = metrics["loss"], step
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    batches = windows(stream, recipe, arguments.seed)
    if arguments.pilot:
        from foundry.pilot import Pilot

        pilot = Pilot(steps=20, warmup=3, stop=False, items=recipe.windows_per_step * recipe.window, item_unit="tokens",
                      budget_seconds=arguments.budget, receipt=output_dir / "pilot.json")
        batches = pilot.watch(batches)

    model.train()
    step = 0
    for batch in batches:
        if arguments.max_steps is not None and step >= arguments.max_steps:
            break
        clock.start_step()
        input_ids = batch.to(device, non_blocking=True).long()  # (b, l)
        corrupted, labels = benchmark.corrupt(input_ids, generator)  # each (b, l)
        block_mask = document_block_mask(corrupted) if use_flex else None
        dense_mask = None if use_flex else document_dense_mask(corrupted)
        loss, count = train_step(compiled, corrupted, labels, block_mask, dense_mask, device)
        if clock.expired():
            model.zero_grad(set_to_none=True)
            break
        progress = clock.progress
        scale = learning_rate_scale(progress, recipe)
        for optimizer, lrs in zip(optimizers, base_lrs, strict=True):
            for group, base_lr in zip(optimizer.param_groups, lrs, strict=True):
                group["lr"] = base_lr * scale
                if "momentum" in group:
                    phase = min(1.0, progress / recipe.muon_momentum_warmup)
                    group["momentum"] = recipe.muon_momentum_start + phase * (recipe.muon_momentum - recipe.muon_momentum_start)
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        clock.applied()
        step += 1
        if step % 50 == 0 or step == 1:
            run.log({"train/loss": float(loss / count.clamp(min=1)), "train/lr_scale": scale,
                     "cost/train_seconds": clock.train_seconds}, step=step)
            print(json.dumps({"step": step, "loss": float(loss / count.clamp(min=1)), "train_seconds": round(clock.train_seconds, 1)}), flush=True)
        if clock.train_seconds >= next_eval:
            evaluate_now(step)
            next_eval += recipe.eval_interval_seconds
    if not curve or curve[-1]["step"] != step:
        evaluate_now(step)
    if best_state is None:
        raise RuntimeError("no validation point was taken")

    model.load_state_dict(best_state)
    model.eval()
    torch.save({name: tensor.to(torch.bfloat16) for name, tensor in best_state.items()}, output_dir / "best.pt")
    metrics: dict[str, dict[str, object]] = {}
    final_eval_seconds = 0.0
    if not arguments.skip_final_evals:
        for split in ("val_select", "validation", "test"):
            held_out = val_select if split == "val_select" else benchmark.load_held_out(arguments.data_dir, split)
            started = time.perf_counter()
            metrics[split] = benchmark.evaluate_split(model, held_out, device, tokens_per_batch=arguments.eval_tokens_per_batch)
            final_eval_seconds += time.perf_counter() - started
            print(json.dumps({split: {key: metrics[split][key] for key in ("loss", "accuracy", "mcc")}}), flush=True)
    clock_receipt = clock.receipt()
    evaluator = benchmark.evaluator_sha256()
    candidate_file = Path(__file__)
    run_result = {
        "schema_version": 2, "name": arguments.name, "hypothesis": arguments.hypothesis, "seed": arguments.seed,
        "status": "completed" if clock_receipt["valid"] else "invalid",
        "benchmark_id": data_id, "evaluator_sha256": evaluator,
        "comparison_key": benchmark.comparison_key(data_id, evaluator, clock_receipt),
        "candidate_sha256": benchmark.candidate_sha256(candidate_file), "candidate_file": candidate_file.name,
        "recipe": asdict(recipe), "max_steps": arguments.max_steps, "compiled": compiled is not model,
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
