"""Check scalar training schedules without initializing CUDA."""

import pytest
import torch

from speedrunning_plms.training.utils import LerpTensor


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_lerp_schedule_updates_the_existing_tensor(dtype: torch.dtype) -> None:
    schedule = LerpTensor.__new__(LerpTensor)
    schedule.start = 0
    schedule.end = 10
    schedule.prec = 2
    schedule.prev_val = None
    schedule.gpu_val = torch.tensor(0, dtype=dtype)  # ()
    original = schedule.gpu_val  # ()

    assert schedule(0.5) is original
    assert original.item() == 4
    assert schedule(0.5) is original
    assert original.item() == 4
    assert schedule(1.0) is original
    assert original.item() == 10
