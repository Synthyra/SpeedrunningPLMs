import random
import time

import numpy as np
import torch
import yaml

from collections.abc import Callable
from os import PathLike
from typing import Any, ParamSpec, TypeVar


_Params = ParamSpec("_Params")
_Return = TypeVar("_Return")


def _get_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0
    for parameter in model.parameters():  # parameter: arbitrary parameter shape (...)
        if parameter.grad is not None:  # gradient: same shape (...)
            param_norm = parameter.grad.data.norm(2)  # ()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class AutoGradClipper:
    """Clip at a percentile of observed gradient norms after ten observations."""

    # adapted from https://github.com/pseeth/autoclip/tree/master

    def __init__(
        self,
        model: torch.nn.Module,
        clip_percentile: float = 10,
        history_length: int = 1000000,
    ) -> None:
        self.model = model
        self.clip_percentile = clip_percentile
        self.history_length = history_length
        self.grad_history: list[float] = []

    def clip_gradients(self) -> np.float64 | None:
        """Clip gradients based on percentile of gradient history."""
        obs_grad_norm = _get_grad_norm(self.model)
        self.grad_history.append(obs_grad_norm)

        if len(self.grad_history) > self.history_length:
            self.grad_history = self.grad_history[-self.history_length:]

        if len(self.grad_history) >= 10:
            clip_value = np.percentile(self.grad_history, self.clip_percentile)  # ()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)  # gradients retain (...)
            return clip_value
        return None


def load_config_from_yaml(yaml_path: str | PathLike[str]) -> Any:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_param_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for _, parameter in model.named_parameters())


class LerpTensor:
    def __init__(self, start_val: float, end_val: float, precision: int | float) -> None:
        self.start, self.end, self.prec = start_val, end_val, precision
        self.prev_val: float | None = None
        dtype = torch.int32 if isinstance(precision, int) else torch.float
        self.gpu_val = torch.tensor(0, dtype=dtype, device="cuda")  # ()

    def __call__(self, frac_done: float) -> torch.Tensor:
        val = (max((1 - frac_done), 0) * self.start + min(frac_done, 1) * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.gpu_val.fill_(val)  # (); update the existing device scalar
            self.prev_val = val
        return self.gpu_val  # ()


class LerpFloat:
    def __init__(self, start_val: float, end_val: float, precision: float) -> None:
        self.start, self.end, self.prec = start_val, end_val, precision
        self.prev_val: float | None = None

    def __call__(self, frac_done: float) -> float:
        val = (max((1 - frac_done), 0) * self.start + min(frac_done, 1) * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.prev_val = val
        return self.prev_val


class GlobalTimer:
    """Track elapsed wall time with CUDA synchronization at each measurement."""

    def __init__(self) -> None:
        self.total_time = 0.0
        self.start_time: float | None = None
        self.is_running = False

    def start(self) -> None:
        """Start the timer."""
        if not self.is_running:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            self.is_running = True

    def pause(self) -> None:
        """Pause the timer and add elapsed time to total."""
        if self.is_running:
            torch.cuda.synchronize()
            self.total_time += time.perf_counter() - self.start_time
            self.is_running = False

    def resume(self) -> None:
        """Resume the timer."""
        self.start()

    def get_time(self) -> float:
        """Get total elapsed time including current session if running."""
        current_time = self.total_time
        if self.is_running:
            torch.cuda.synchronize()
            current_time += time.perf_counter() - self.start_time
        return current_time

    def reset(self) -> None:
        """Reset the timer to zero."""
        self.total_time = 0.0
        self.start_time = None
        self.is_running = False


def exclude_from_timer(timer: GlobalTimer) -> Callable[[Callable[_Params, _Return]], Callable[_Params, _Return]]:
    """Decorator that pauses the timer during function execution."""
    def decorator(func: Callable[_Params, _Return]) -> Callable[_Params, _Return]:
        def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _Return:
            timer.pause()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.resume()
            return result
        return wrapper
    return decorator
