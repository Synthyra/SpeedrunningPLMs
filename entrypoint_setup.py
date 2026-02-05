import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only error/warning messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'true'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# if on a linux machine, set HF_HOME to the directory of the script
if os.name == 'linux' and "HF_HOME" not in os.environ:
    os.environ['HF_HOME'] = os.path.dirname(os.path.abspath(__file__))


# === PyTorch Performance Optimizations ===
try:
    import torch
    import atexit
    # Enable TensorFloat32 tensor cores for float32 matmul (Ampere+ GPUs)
    # Provides significant speedup with minimal precision loss
    torch.set_float32_matmul_precision('high')
    
    # Enable TF32 for matrix multiplications and cuDNN operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN autotuner - finds fastest algorithms for your hardware
    # Best when input sizes are consistent; may slow down first iterations
    torch.backends.cudnn.benchmark = True
    
    # Deterministic operations off for speed (set True if reproducibility needed)
    torch.backends.cudnn.deterministic = False


    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm_backends = "ATEN,CUTLASS,FBGEMM"

    try:
        import torch._dynamo as dynamo
        dynamo.config.capture_scalar_outputs = True
    except Exception:
        print("Failed to import torch._dynamo")

    # Ensure DDP process groups are destroyed on exit to avoid NCCL warnings.
    try:
        import torch.distributed as dist
        def _cleanup_ddp():
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        atexit.register(_cleanup_ddp)
    except Exception:
        pass


    
except ImportError:
    pass


try:
    import wandb
    os.environ["WANDB_AVAILABLE"] = 'true'
except ImportError:
    os.environ["WANDB_AVAILABLE"] = 'false'