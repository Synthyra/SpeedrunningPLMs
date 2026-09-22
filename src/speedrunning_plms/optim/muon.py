import os
import torch
import torch.distributed as dist

from collections.abc import Iterable


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Apply quintic Newton-Schulz steps to approximately orthogonalize a matrix."""
    # G: (m, n); r = min(m, n), s = max(m, n).
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()  # (m, n)
    if G.size(0) > G.size(1):
        X = X.T  # (n, m), so X is (r, s) after this branch

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)  # (r, s); norm is ()
    for _ in range(steps):
        A = X @ X.T  # (r, r)
        B = b * A + c * A @ A  # (r, r); coefficients from @jxbz, @leloykun, @YouJiacheng
        X = a * X + B @ X  # (r, s)

    if G.size(0) > G.size(1):
        X = X.T  # (m, n)
    return X  # (m, n)


class Muon(torch.optim.Optimizer):
    """Apply momentum and approximate orthogonalization to 2D CUDA parameters.

    Embeddings, output heads, and scalar/vector parameters need another optimizer.
    Equal-sized parameter groups must divide evenly across distributed ranks.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        # Each parameter is (m, n); size = m * n within each group.
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [
            {
                'params': [p for p in params if p.numel() == size],
                'update_buffer': [
                    torch.empty(size, device='cuda', dtype=torch.bfloat16)  # (size,)
                    for _ in range(self.world_size)
                ],
            }
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']  # world_size tensors, each (size,)
            params = group['params']
            assert len(params) % self.world_size == 0
            handle = None
            params_world = None

            def update_prev() -> None:
                if params_world is None:
                    return
                if handle is not None:
                    handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    # p_world: (m, n); g_world: (size,), where size = m * n.
                    p_world.data.add_(
                        g_world.view_as(p_world),  # (m, n)
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )  # (m, n), updated in place

            for base_i in range(len(params))[::self.world_size]:
                parameter = params[base_i + self.rank]  # (m, n)
                gradient = parameter.grad  # (m, n) or None
                assert gradient is not None
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(gradient)  # (m, n)
                buffer = state['momentum_buffer']  # (m, n)
                buffer.lerp_(gradient, 1 - momentum)  # (m, n)
                gradient = gradient.lerp_(buffer, momentum) if nesterov else buffer  # (m, n)
                gradient = zeropower_via_newtonschulz5(gradient, steps=ns_steps).flatten()  # (size,)
                update_prev()
                if self.world_size > 1:
                    handle = dist.all_gather(update_buffers, gradient, async_op=True)  # each buffer: (size,)
                else:
                    update_buffers[0].copy_(gradient)  # (size,)
                    handle = None
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
