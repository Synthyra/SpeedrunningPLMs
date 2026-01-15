import os
import torch
import torch.distributed as dist


### Muon optimizer
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile
def polar_express(G, steps=5, eps=1e-6):
    assert len(G.shape) == 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() * (1 + 2e-2) + eps)
    X = X.contiguous()
    A = torch.empty((X.size(0), X.size(0)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    for a, b, c in polar_express_coeffs[:steps]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        C = a * X + B @ X
        X, C = C, X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def apply_normuon_variance_reduction(v, second_momentum_buffer, beta2, red_dim):
    v_mean = v.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v.mul_(final_scale.type_as(v))


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
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
                    torch.empty(size, device='cuda', dtype=torch.bfloat16)
                    for _ in range(self.world_size)
                ],
            }
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            assert len(params) % self.world_size == 0
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                if handle is not None:
                    handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                p = params[base_i + self.rank]
                g = p.grad
                assert g is not None
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = polar_express(g, steps=ns_steps).flatten()
                update_prev()
                if self.world_size > 1:
                    handle = dist.all_gather(update_buffers, g, async_op=True)
                else:
                    update_buffers[0].copy_(g)
                    handle = None
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


class NorMuon(torch.optim.Optimizer):
    """
    NorMuon optimizer with Polar Express orthogonalization and cautious weight decay.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                if "second_momentum_buffer" not in state:
                    red_dim = -1 if p.shape[-2] >= p.shape[-1] else -2
                    state["second_momentum_buffer"] = (
                        torch.zeros_like(grad[..., :, :1])
                        if red_dim == -1
                        else torch.zeros_like(grad[..., :1, :])
                    )

                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                v = grad.lerp_(buf, momentum)
                v = polar_express(v)

                red_dim = -1 if p.shape[-2] >= p.shape[-1] else -2
                v = apply_normuon_variance_reduction(v, state["second_momentum_buffer"], beta2, red_dim)

                if weight_decay != 0:
                    mask = (v * p) >= 0
                    p.add_(p * mask, alpha=-weight_decay * lr)
                p.add_(v, alpha=-lr)
