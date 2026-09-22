"""Select AdamW or Muon plus AdamW for protein model training."""

from __future__ import annotations

import torch

from speedrunning_plms.optim.muon import Muon


def build_optimizers(
    model: torch.nn.Module, *, optimizer: str, learning_rate: float,
    muon_lr: float, weight_decay: float, muon_momentum: float,
    muon_steps: int, muon_backend: str, fused_adam: bool,
) -> list[torch.optim.Optimizer]:
    """Use Muon for hidden projections, keeping tied weights and gates in AdamW.

    Module traversal also handles DDP and compiled wrappers. Fused AdamW is
    enabled only when requested and every AdamW parameter is on CUDA; CPU
    uses ordinary AdamW. The AdamW-only path preserves the original group.
    """
    if optimizer not in {"adamw", "muon"}:
        raise ValueError("optimizer must be adamw or muon")
    parameters = list(model.parameters())  # Parameter shapes vary by module.
    if optimizer == "adamw":
        return [torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay,
                                 fused=True if fused_adam and all(p.is_cuda for p in parameters) else None)]

    excluded: set[int] = set()
    linear_weights: set[int] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_weights.add(id(module.weight))
        if isinstance(module, torch.nn.Embedding) or {"lm_head", "value_gate"} & set(name.split(".")):
            excluded.update(id(p) for p in module.parameters())
        get_output = getattr(module, "get_output_embeddings", None)
        if callable(get_output):
            output = get_output()
            if output is not None:
                excluded.update(id(p) for p in output.parameters())

    muon_parameters = [p for p in parameters if p.requires_grad and p.ndim == 2
                       and id(p) in linear_weights and id(p) not in excluded]  # Each (m, n).
    muon_ids = {id(p) for p in muon_parameters}
    adam_parameters = [p for p in parameters if p.requires_grad and id(p) not in muon_ids]
    optimizers: list[torch.optim.Optimizer] = []
    if muon_parameters:
        optimizers.append(Muon(muon_parameters, lr=muon_lr, momentum=muon_momentum,
                               ns_steps=muon_steps, weight_decay=weight_decay,
                               orthogonalization=muon_backend))
    if adam_parameters:
        optimizers.append(torch.optim.AdamW(adam_parameters, lr=learning_rate,
                          weight_decay=weight_decay,
                          fused=True if fused_adam and all(p.is_cuda for p in adam_parameters) else None))
    if not optimizers:
        raise ValueError("The model has no trainable parameters")
    return optimizers
