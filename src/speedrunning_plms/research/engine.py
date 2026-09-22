"""Time-bounded, fixed-objective protein masked-language-model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from speedrunning_plms.models import PLM, PLMConfig
from speedrunning_plms.research import benchmark


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str = "data/uniref50"
    output_dir: str = "runs/baseline"
    time_budget: float = 300.0
    max_steps: int | None = None
    device: str = "auto"
    seed: int = 42
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    architecture: str = "standard"
    hidden_size: int = 256
    heads: int = 4
    layers: int = 6
    patch_layers: int = 4
    compile: bool = False
    bf16: bool = False
    cpu_threads: int = 1
    evaluate_only: str | None = None
    split: str = "valid"


def _validate(config: ExperimentConfig) -> None:
    integer_fields = (
        "batch_size", "grad_accum", "hidden_size", "heads", "layers", "patch_layers", "cpu_threads",
    )
    for name in integer_fields:
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    for name in ("time_budget", "learning_rate", "weight_decay"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if config.time_budget == 0 or config.learning_rate == 0:
        raise ValueError("time_budget and learning_rate must be positive")
    if config.max_steps is not None and (type(config.max_steps) is not int or config.max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if type(config.seed) is not int or not -(2**63) <= config.seed < 2**64:
        raise ValueError("seed must be an integer between -2**63 and 2**64 - 1")

    if config.architecture not in {"standard", "unet", "patch_unet"}:
        raise ValueError("architecture must be standard, unet, or patch_unet")
    if config.hidden_size % config.heads or (config.hidden_size // config.heads) % 2:
        raise ValueError("hidden_size must be divisible by heads with an even head dimension")
    if config.architecture == "unet" and config.layers % 2:
        raise ValueError("unet requires an even number of layers")
    if config.architecture == "patch_unet" and config.patch_layers % 2:
        raise ValueError("patch_unet requires an even number of patch_layers")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config.split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    if config.split == "test" and not config.evaluate_only:
        raise ValueError("The test split is only available with --evaluate-only")

    for name in ("compile", "bf16"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")


def _distributed_environment() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Require WORLD_SIZE >= 1, 0 <= RANK < WORLD_SIZE, and LOCAL_RANK >= 0")
    return rank, world_size, local_rank


def training_batches(
    tokens: Tensor, batch_size: int, seed: int, rank: int, world_size: int,
) -> Iterator[Tensor]:
    """Cycle shuffled global batches, giving each rank equally many sequences."""
    # tokens: (n, l); each yield: (b, l). Repeats occur only across epochs.
    if len(tokens) == 0:
        raise ValueError("Training split is empty")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.empty(0, dtype=torch.long)  # (0,)
    global_batch_size = batch_size * world_size
    while True:
        while len(indices) < global_batch_size:
            indices = torch.cat((indices, torch.randperm(len(tokens), generator=generator)))  # (remaining,)
        local_indices = indices[rank * batch_size : (rank + 1) * batch_size]  # (b,)
        yield tokens[local_indices]  # (b, l)
        indices = indices[global_batch_size:]  # (remaining,)


def _loss_sum(logits: Tensor, labels: Tensor) -> Tensor:
    # logits: (b, l, c); labels: (b, l). Sum remains zero for an unmasked batch.
    return F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), reduction="sum", ignore_index=-100,
    )  # ()


def _verify_distributed_benchmark(fingerprint: str, code_sha256: str, world_size: int) -> None:
    if world_size == 1:
        return
    local = (fingerprint, code_sha256)
    identities: list[tuple[str, str] | None] = [None] * world_size
    dist.all_gather_object(identities, local)
    if any(identity != local for identity in identities):
        raise ValueError("Distributed ranks have different benchmark data or evaluator code")


def _verify_distributed_config(config: ExperimentConfig, world_size: int) -> None:
    if world_size == 1:
        return
    # Nodes may mount identical datasets and outputs at different local paths.
    local = asdict(config)
    del local["data_dir"], local["output_dir"]
    configurations: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(configurations, local)
    if any(configuration != local for configuration in configurations):
        raise ValueError("Distributed ranks have different experiment configurations")


def _deadline_reached(
    start: float, budget: float, device: torch.device, rank: int, world_size: int,
) -> bool:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    expired = rank == 0 and time.perf_counter() - start >= budget
    stop = torch.tensor(int(expired), device=device)  # ()
    if world_size > 1:
        dist.broadcast(stop, src=0)  # ()
    return bool(stop.item())


def _train(
    model: torch.nn.Module,
    tokens: Tensor,
    config: ExperimentConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[int, int, float]:
    # tokens: (n, l). DDP averages gradients; scale to the global masked-token mean.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches = training_batches(tokens, config.batch_size, config.seed, rank, world_size)
    generator = torch.Generator().manual_seed((config.seed + 1 + rank) % 2**64)
    model.train()
    steps = attempts = total_masked = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    while True:
        if config.max_steps is not None and attempts >= config.max_steps:
            break
        if _deadline_reached(start, config.time_budget, device, rank, world_size):
            break

        optimizer.zero_grad(set_to_none=True)
        masked_count = torch.zeros((), dtype=torch.long, device=device)  # ()
        finite = torch.ones((), dtype=torch.long, device=device)  # ()
        interrupted = False
        for _ in range(config.grad_accum):
            if _deadline_reached(start, config.time_budget, device, rank, world_size):
                interrupted = True
                break
            inputs, labels = benchmark.corrupt_tokens(next(batches), generator=generator)  # each (b, l)
            inputs, labels = inputs.to(device), labels.to(device)  # each (b, l)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16):
                attention_mask = inputs != benchmark.PAD_TOKEN_ID  # (b, l)
                logits = model(input_ids=inputs, attention_mask=attention_mask).logits  # (b, l, c)
                loss = _loss_sum(logits, labels)  # ()
            finite *= torch.isfinite(loss).long()  # ()
            loss.backward()
            masked_count += (labels != -100).sum()  # ()
        if interrupted:
            optimizer.zero_grad(set_to_none=True)
            break

        if world_size > 1:
            dist.all_reduce(masked_count)  # ()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)  # ()
        if not finite.item():
            raise ValueError("Training produced a non-finite loss")

        count = masked_count.item()
        if count:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(world_size / count)  # same shape as parameter
            # Discard overtime work so long microbatches or accumulation cannot
            # buy extra updates. CUDA synchronization also meters gradient scaling.
            if _deadline_reached(start, config.time_budget, device, rank, world_size):
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            steps += 1
        attempts += 1
        total_masked += count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return steps, total_masked, time.perf_counter() - start


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train or evaluate locally; only rank zero writes checkpoint and result.json."""
    _validate(config)
    rank, world_size, local_rank = _distributed_environment()
    wall_start = time.perf_counter()
    output_dir = Path(config.output_dir)
    if (output_dir / "result.json").exists() or (output_dir / "checkpoint").exists():
        raise FileExistsError(f"Experiment artifacts already exist in {output_dir}")

    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    device_type = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device_type = "cpu" if device_type == "auto" else device_type
    device = torch.device("cuda", local_rank) if device_type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config.bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16")
        torch.cuda.reset_peak_memory_stats(device)

    initialized = False
    try:
        if world_size > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
            initialized = True
        _verify_distributed_config(config, world_size)

        manifest = benchmark.load_manifest(Path(config.data_dir))
        fingerprint = benchmark.benchmark_id(Path(config.data_dir))
        benchmark_code_sha256 = hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()
        _verify_distributed_benchmark(fingerprint, benchmark_code_sha256, world_size)
        evaluation_tokens = benchmark.load_split(Path(config.data_dir), config.split)  # (n_eval, l)
        if config.evaluate_only:
            model = PLM.from_pretrained(config.evaluate_only, local_files_only=True).to(device)
            if not model.config.mlm or model.config.masked_diffusion:
                raise ValueError("Checkpoint must use the fixed MLM objective")
        else:
            length = manifest["max_length"]
            if config.architecture == "patch_unet" and length & (length - 1):
                raise ValueError("patch_unet requires a power-of-two benchmark max_length")
            model_config = PLMConfig(
                hidden_size=config.hidden_size, num_attention_heads=config.heads,
                num_hidden_layers=config.layers, num_unet_layers=config.patch_layers,
                max_sequence_length=manifest["max_length"], vocab_size=benchmark.VOCAB_SIZE,
                unet=config.architecture == "unet", patch_unet=config.architecture == "patch_unet",
                mlm=True, masked_diffusion=False, token_dropout=False,
                compile_flex_attention=False, tokenizer_name=None,
                cls_token_id=benchmark.CLS_TOKEN_ID, eos_token_id=benchmark.EOS_TOKEN_ID,
                pad_token_id=benchmark.PAD_TOKEN_ID, mask_token_id=benchmark.MASK_TOKEN_ID,
            )
            model = PLM(model_config).to(device)

        training_model = torch.compile(model) if config.compile else model
        if world_size > 1 and not config.evaluate_only:
            training_model = DistributedDataParallel(
                training_model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        steps, train_masked, train_seconds = 0, 0, 0.0
        if not config.evaluate_only:
            training_tokens = benchmark.load_split(Path(config.data_dir), "train")  # (n_train, l)
            steps, train_masked, train_seconds = _train(
                training_model, training_tokens, config, device, rank, world_size,
            )

        # Unequal evaluation shards must not pass through DDP forward collectives.
        metrics = benchmark.evaluate_model(
            model, evaluation_tokens, config.batch_size, device, rank=rank, world_size=world_size,
        )
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        gpu_names = [gpu_name] * world_size
        if world_size > 1:
            dist.all_gather_object(gpu_names, gpu_name)

        architecture = "standard"
        if model.config.patch_unet:
            architecture = "patch_unet"
        elif model.config.unet:
            architecture = "unet"
        metric_prefix = "val" if config.split == "valid" else "test"
        result: dict[str, Any] = {
            "schema_version": 1, "status": "completed", "split": config.split,
            "objective": "masked15", "dataset": manifest["dataset"], "benchmark_id": fingerprint,
            "data_fingerprint": fingerprint, "config": asdict(config), "seed": config.seed,
            "benchmark_code_sha256": benchmark_code_sha256,
            "architecture": architecture,
            "model_config": model.config.to_dict(),
            "time_budget": config.time_budget, "max_steps": config.max_steps,
            "optimizer_steps": steps, "train_masked_tokens": train_masked,
            "train_seconds": train_seconds, "world_size": world_size, "device": str(device),
            "gpu_name": gpu_name, "gpu_names": gpu_names,
            "cpu_name": platform.processor(), "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "eval_dtype": "float32",
            "train_dtype": "bfloat16" if config.bf16 else "float32",
            "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
            "masked_tokens": metrics["masked_tokens"], "masked_accuracy": metrics["masked_accuracy"],
            f"{metric_prefix}_loss": metrics["loss"],
            f"{metric_prefix}_bits_per_masked_residue": metrics["bits_per_masked_residue"],
        }
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not config.evaluate_only:
                model.save_pretrained(output_dir / "checkpoint")
            result["wall_seconds"] = time.perf_counter() - wall_start
            temporary = output_dir / "result.json.tmp"
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(output_dir / "result.json")
        return result
    finally:
        if initialized:
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Protein MLM speedrun: fixed 15% masking")
    parser.add_argument(
        "--config", type=Path, help="JSON experiment hyperparameters; explicit flags take precedence",
    )
    defaults = ExperimentConfig()
    for field in fields(ExperimentConfig):
        default = getattr(defaults, field.name)
        kwargs: dict[str, Any] = {"default": argparse.SUPPRESS}
        if field.name in {"compile", "bf16"}:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            value_type = str if default is None else type(default)
            kwargs["type"] = int if field.name == "max_steps" else value_type
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)

    arguments = vars(parser.parse_args(argv))
    config_path = arguments.pop("config")
    configured = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    if not isinstance(configured, dict):
        parser.error("Config must be a JSON object")
    unknown = configured.keys() - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        parser.error(f"Unknown config fields: {', '.join(sorted(unknown))}")

    result = run_experiment(ExperimentConfig(**(configured | arguments)))
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
