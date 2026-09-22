"""Check accumulated distributed Muon updates against a global-batch reference."""

import json
import os
import socket
import subprocess
import sys
import pytest
import torch

from dataclasses import asdict
from pathlib import Path

from speedrunning_plms.models import PLM
from speedrunning_plms.optim.factory import build_optimizers
from speedrunning_plms.research import benchmark, engine


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch lacks Gloo")
def test_two_process_muon_accumulation_matches_global_masked_mean(tmp_path: Path) -> None:
    train = torch.tensor(benchmark.encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 8, 16))  # (n, 16)
    data_dir = tmp_path / "data"
    benchmark.write_dataset({"train": train, "valid": train}, data_dir)
    config = engine.ExperimentConfig(
        data_dir=str(data_dir), output_dir=str(tmp_path / "run"), device="cpu",
        architecture="standard", hidden_size=8, heads=2, layers=2, batch_size=2,
        time_budget=30, max_steps=1, grad_accum=2, optimizer="muon",
        attention_backend="sdpa", fused_qkv=True, value_embeddings=True,
        embedding_residual=True, prefetch=True, lr_schedule="constant",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ | {
        "USE_LIBUV": "0", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    command = [sys.executable, "-m", "speedrunning_plms.research.engine", "--config", str(config_path)]
    # Direct workers support Windows PyTorch builds without libuv rendezvous.
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment | {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}) for rank in range(2)]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=60)
            assert worker.returncode == 0, stdout + stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    result = json.loads((tmp_path / "run/result.json").read_text(encoding="utf-8"))
    assert result["world_size"] == 2
    assert result["optimizer_steps"] == 1
    distributed_model = PLM.from_pretrained(tmp_path / "run/checkpoint", local_files_only=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    reference = PLM(distributed_model.config)
    optimizers = build_optimizers(
        reference, optimizer=config.optimizer, learning_rate=config.learning_rate,
        muon_lr=config.muon_lr, weight_decay=config.weight_decay,
        muon_momentum=config.muon_momentum, muon_steps=config.muon_steps,
        muon_backend=config.muon_backend, fused_adam=config.fused_adam,
    )
    reference.train()
    masked_counts = []
    for rank in range(2):
        batches = engine.training_batches(train, config.batch_size, config.seed, rank, 2)
        generator = torch.Generator().manual_seed(config.seed + 1 + rank)
        for _ in range(config.grad_accum):
            inputs, labels = benchmark.corrupt_tokens(next(batches), generator=generator)  # each (b, 16)
            logits = reference(input_ids=inputs, attention_mask=inputs != benchmark.PAD_TOKEN_ID).logits  # (b, 16, 33)
            engine._loss_sum(logits, labels).backward()
            masked_counts.append((labels != -100).sum().item())

    # Unequal counts distinguish a global residue mean from microbatch means.
    assert len(set(masked_counts)) > 1
    assert result["train_masked_tokens"] == sum(masked_counts)
    for parameter in reference.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sum(masked_counts))  # same shape as parameter
    for optimizer in optimizers:
        optimizer.step()
    for name, actual in distributed_model.named_parameters():
        expected = reference.get_parameter(name)  # same shape as actual
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5, msg=name)
