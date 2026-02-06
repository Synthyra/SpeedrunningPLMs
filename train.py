import entrypoint_setup

import os
import sys

code = open(sys.argv[0]).read()
code += open('entrypoint_setup.py', 'r', encoding='utf-8').read()
code += open('optimizer.py', 'r', encoding='utf-8').read()
code += open('data/dataloading.py', 'r', encoding='utf-8').read()
code += open('model/utils.py', 'r', encoding='utf-8').read()
code += open('model/attention.py', 'r', encoding='utf-8').read()
code += open('model/model.py', 'r', encoding='utf-8').read()

import uuid
import contextlib
import subprocess
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist

from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from transformers import EsmTokenizer, get_scheduler
from tqdm import tqdm
from pathlib import Path

from data.download_data import get as ensure_hf_file
from model.model import PLM, PLMConfig
from data.dataloading import (
    OptimizedTrainLoader,
    OptimizedEvalLoader,
    ChunkedTrainLoader,
    ChunkedEvalLoader,
    AsyncBatchPipeline,
    apply_masking_gpu,
)
from optimizer import Muon
from utils import (
    set_seed,
    load_config_from_yaml,
    exclude_from_timer,
    GlobalTimer,
    LerpTensor,
    LerpFloat,
    AutoGradClipper
)


if os.environ['WANDB_AVAILABLE'] == 'true':
    import wandb


def arg_parser():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to YAML file")

    # CLI-specific arguments (always from CLI for security)
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases API token")
    parser.add_argument("--log_name", type=str, default=None, help="Name of the log file, else will be randomly generated")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    
    # All other arguments with defaults (can be overridden by YAML)
    parser.add_argument("--save_path", type=str, default="Synthyra/speedrun_test", help="Path to save the model and report to wandb")
    parser.add_argument("--data_name", type=str, default="uniref50", help="Dataset name: uniref50, omg_prot50, or og_prot90")
    parser.add_argument("--num_chunks", type=int, default=100, help="Number of training chunks to ensure are downloaded")
    
    # Distributed training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clear_cache_every", type=int, default=1000, help="Clear CUDA cache every N steps")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (0 to disable)")
    parser.add_argument("--auto_grad_clip", action="store_true", help="Enable auto gradient clipping")
    parser.add_argument("--auto_grad_clip_p", type=float, default=10.0, help="Percentile for auto gradient clipping")
    
    # Model hyperparams
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--num_attention_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=24, help="Number of hidden layers (for non-unet)")
    parser.add_argument("--num_unet_layers", type=int, default=0, help="Number of Conv1D UNet layers (encoder + decoder)")
    parser.add_argument("--num_extra_layers", type=int, default=0, help="Number of extra transformer layers after UNet")
    parser.add_argument("--vocab_size", type=int, default=33, help="Vocabulary size")
    parser.add_argument("--expansion_ratio", type=float, default=2.0, help="Expansion ratio for MLP")
    parser.add_argument("--soft_logit_cap", type=float, default=32.0, help="Soft logit cap")
    parser.add_argument("--tie_embeddings", action="store_true", help="Tie embeddings")
    parser.add_argument("--unet", type=bool, default=True, help="Use UNet architecture (skip connections only)")
    parser.add_argument("--patch_unet", action="store_true", help="Use Patch UNet with downsampling (Swin-style)")
    parser.add_argument("--token_dropout", type=bool, default=True, help="Use token dropout")
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16")
    parser.add_argument("--compile_model", type=bool, default=True, help="Use torch.compile on the full model")
    parser.add_argument("--compile_flex_attention", type=bool, default=True, help="Compile flex_attention for fused attention")
    parser.add_argument("--dynamo_recompile_limit", type=int, default=32, help="Dynamo recompile limit for torch.compile")
    
    # Data hyperparams
    parser.add_argument("--mlm", action="store_true", help="Use masked language modeling")
    parser.add_argument("--masked_diffusion", action="store_true", help="Use masked diffusion")
    parser.add_argument("--mask_rate", type=float, default=0.2, help="Mask rate for masked language modeling")
    parser.add_argument("--starting_mask_rate", type=float, default=0.1, help="Starting mask rate for masked language modeling")
    parser.add_argument("--mask_rate_steps", type=int, default=2500, help="Number of steps to reach mask rate")
    parser.add_argument("--mask_rate_schedule", action="store_true", help="Use mask rate schedule")
    
    # Optimization hyperparams
    parser.add_argument("--batch_size", type=int, default=8*64*1024, help="Total batch size in tokens")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--cooldown_steps", type=int, default=5000, help="Number of cooldown steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--scheduler_type", type=str, default='cosine', help="Scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of warmup steps")

    # Adam optimizer params
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer when not using Muon")
    parser.add_argument("--lr_embed", type=float, default=0.001, help="Learning rate for embeddings")
    parser.add_argument("--lr_head", type=float, default=0.001, help="Learning rate for head")
    parser.add_argument("--lr_scalar", type=float, default=0.001, help="Learning rate for scalar params")
    
    # Muon optimizer params
    parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer")
    parser.add_argument("--lr_hidden", type=float, default=0.001, help="Learning rate for hidden layers (Muon)")
    parser.add_argument("--muon_momentum_warmup_steps", type=int, default=300, help="Steps for warmup momentum (0.85 -> 0.95)")
    
    # Evaluation and logging hyperparams
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate on validation set every N steps")
    parser.add_argument("--hf_model_name", type=str, default='lhallee/speedrun', help="Huggingface model name for saving")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N steps")
    
    # Dataloader params
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for optimized dataloader")
    parser.add_argument("--prefetch_factor", type=int, default=8, help="Prefetch factor for optimized dataloader")
    
    # Parse CLI args first
    args = parser.parse_args()
    
    # Load YAML config if provided
    if args.yaml_path:
        yaml_config = load_config_from_yaml(args.yaml_path)
        
        # Security: Never load tokens from YAML files
        cli_only_params = {'hf_token', 'wandb_token', 'yaml_path'}
        
        # Override defaults with YAML values, but preserve CLI overrides
        for key, value in yaml_config.items():
            if key not in cli_only_params and hasattr(args, key):
                # Only override if the argument wasn't explicitly provided via CLI
                # Check if the current value is the default by comparing with parser defaults
                action = next((action for action in parser._actions if action.dest == key), None)
                if action and getattr(args, key) == action.default:
                    # Convert boolean strings to boolean values
                    if isinstance(action.default, bool) and isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    setattr(args, key, value)
    
    # Align input patterns to dataset if not already pointing at it
    args.input_bin = f"data/{args.data_name}/{args.data_name}_train_*.bin"
    args.input_valid_bin = f"data/{args.data_name}/{args.data_name}_valid_*.bin"
    args.input_test_bin = f"data/{args.data_name}/{args.data_name}_test_*.bin"
    return args


class Trainer:
    def __init__(self, args, model_config):
        self.args = args
        self.model_config = model_config

        self.wandb_initialized = False
        
        # Initialize global timer
        self.train_timer = GlobalTimer()
        
        # Initialize mask rate tracking (used directly for patch_unet GPU-side masking)
        self.current_mask_rate = args.mask_rate if args.mlm else 1.0
        
        # Initialize auto gradient clipper
        self.auto_grad_clipper = None
        self.last_clip_value = None
        
        if 'RANK' in os.environ:
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.ddp_local_rank}')
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend='nccl', device_id=self.device)
            dist.barrier()
            self.master_process = (self.ddp_rank == 0)
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
            self.master_process = True

        set_seed(self.args.seed)
        
        print(f'Process {self.ddp_rank}: using device: {self.device}')

    def print0(self, s, logonly=False):
        if self.master_process:
            with open(self.logfile, 'a', encoding='utf-8') as f:
                if not logonly:
                    print(s)
                print(s, file=f)
    
    def log_wandb(self, log_dict, prefix='train'):
        if self.master_process and self.wandb_initialized:
            wandb.log({f'{prefix}/{k}': v for k, v in log_dict.items()})

    @staticmethod
    def _update_confusion(confusion: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor):
        valid_mask = labels != -100
        if not valid_mask.any():
            return
        valid_preds = preds[valid_mask].view(-1).to(dtype=torch.int64, device='cpu')
        valid_labels = labels[valid_mask].view(-1).to(dtype=torch.int64, device='cpu')
        num_classes = confusion.shape[0]
        indices = valid_labels * num_classes + valid_preds
        counts = torch.bincount(indices, minlength=num_classes * num_classes)
        confusion += counts.view(num_classes, num_classes)

    @staticmethod
    def _calculate_metrics_from_confusion(confusion: torch.Tensor):
        total = int(confusion.sum().item())
        if total == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "num_tokens": 0,
            }
        confusion_f = confusion.to(dtype=torch.float64)
        tp = torch.diag(confusion_f)
        actual = confusion_f.sum(dim=1)
        predicted = confusion_f.sum(dim=0)
        precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
        recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
        f1 = torch.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall),
            torch.zeros_like(tp),
        )
        weighted_precision = (precision * actual).sum().item() / total
        weighted_recall = (recall * actual).sum().item() / total
        weighted_f1 = (f1 * actual).sum().item() / total
        correct = tp.sum().item()
        numerator = correct * total - (predicted * actual).sum().item()
        denom_left = total * total - (predicted * predicted).sum().item()
        denom_right = total * total - (actual * actual).sum().item()
        if denom_left <= 0 or denom_right <= 0:
            mcc = 0.0
        else:
            mcc = numerator / math.sqrt(denom_left * denom_right)
        return {
            "accuracy": correct / total,
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
            "mcc": mcc,
            "num_tokens": total,
        }

    @staticmethod
    def _read_bin_num_tokens(path):
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=3)
        if header.size < 3:
            raise ValueError(f"Invalid header in {path}")
        return int(header[2])

    def _print_val_preview(self, input_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
        if not self.master_process:
            return
        pad_token_id = self.pad_token_id
        # Flatten batched tensors to 1D for preview
        if input_ids.dim() == 2:
            input_ids = input_ids.view(-1)
        if labels.dim() == 2:
            labels = labels.view(-1)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])
        assert input_ids.dim() == 1, f"Expected input_ids to be 1D (seq_len,) but got: {input_ids.shape}"
        assert labels.dim() == 1, f"Expected labels to be 1D (seq_len,) but got: {labels.shape}"
        assert logits.dim() == 2, f"Expected logits to be 2D (seq_len, vocab_size) but got: {logits.shape}"
        assert input_ids.shape[0] == labels.shape[0], f"input_ids/labels length mismatch: {input_ids.shape[0]} != {labels.shape[0]}"
        assert logits.shape[0] == input_ids.shape[0], f"logits/input_ids length mismatch: {logits.shape[0]} != {input_ids.shape[0]}"
        input_ids = input_ids.cpu()
        labels = labels.cpu()
        logits = logits.cpu()
        masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            self.print0("Validation preview: no masked positions in selected batch.")
            return

        preds = logits.argmax(dim=-1).to(dtype=input_ids.dtype)
        filled = input_ids.clone()
        filled[masked_positions] = preds[masked_positions]

        original = input_ids.clone()
        original[masked_positions] = labels[masked_positions]

        def _strip_pad(ids):
            if (ids == pad_token_id).any():
                last_valid = (ids != pad_token_id).nonzero(as_tuple=True)[0][-1].item()
                return ids[: last_valid + 1]
            return ids

        input_ids = _strip_pad(input_ids)
        original = _strip_pad(original)
        filled = _strip_pad(filled)

        decoded_input = self.tokenizer.decode(input_ids.tolist()[:128], skip_special_tokens=False).replace(" ", "").replace("<mask>", "-")
        decoded_original = self.tokenizer.decode(original.tolist()[:128], skip_special_tokens=False).replace(" ", "")
        decoded_filled = self.tokenizer.decode(filled.tolist()[:128], skip_special_tokens=False).replace(" ", "").replace("<mask>", "-")

        masked_list = masked_positions.tolist()[:10]
        self.print0("=" * 128, logonly=True)
        self.print0("VALIDATION PREVIEW (single example)", logonly=True)
        self.print0(f"Masked positions:\n{masked_list} ...", logonly=True)
        self.print0(f"Raw input ids:\n{input_ids.tolist()[:10]} ...", logonly=True)
        self.print0(f"Raw original ids:\n{original.tolist()[:10]} ...", logonly=True)
        self.print0(f"Raw filled ids:\n{filled.tolist()[:10]} ...", logonly=True)
        self.print0("-" * 128, logonly=True)
        self.print0(f"Decoded input:\n{decoded_input}", logonly=True)
        self.print0(f"Decoded original:\n{decoded_original}", logonly=True)
        self.print0(f"Decoded filled:\n{decoded_filled}", logonly=True)
        self.print0("=" * 128, logonly=True)

    def init_training(self):
        self.logfile = None
        if self.master_process:
            os.makedirs('logs', exist_ok=True)
            
            # Use provided log_name or generate a random UUID
            if self.args.log_name:
                run_id = self.args.log_name
            else:
                run_id = str(uuid.uuid4())
            log_filename = f'{run_id}.txt'
                
            self.logfile = os.path.join('logs', log_filename)
            print(os.path.basename(self.logfile))
            # create the log file
            with open(self.logfile, 'w', encoding='utf-8') as f:
                # begin the log by printing this file (the Python code)
                print(code, file=f)
                print('=' * 100, file=f)

        # Synchronize before initializing wandb
        if self.ddp_world_size > 1:
            dist.barrier()

        if self.master_process and self.wandb_initialized:
            wandb.init(
                project="speedrunning-plms",
                name=run_id,
                config={
                    **vars(self.args),
                    **vars(self.model_config),
                    "ddp_world_size": self.ddp_world_size,
                    "device": str(self.device)
                }
            )

        self.print0(f'Running python {sys.version}')
        self.print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.print0(f'{result.stdout}', logonly=True)
        self.print0('='*100, logonly=True)

        # Log configuration source
        if self.args.yaml_path:
            self.print0(f'Configuration loaded from YAML: {self.args.yaml_path}')
            self.print0('CLI arguments override YAML where provided (tokens always from CLI for security)')
        else:
            self.print0('Configuration from CLI arguments only')
        self.print0('='*50)
        
        self.print0(f'Model config:\n{self.model_config}')
        self.print0('Args:')
        for k, v in self.args.__dict__.items():
            self.print0(f'{k}: {v}')
        self.print0('='*100, logonly=True)

        # calculate local batch size
        self.batch_size = self.args.batch_size // self.args.grad_accum // self.ddp_world_size

        self.print0(f'Train accumulation steps: {self.args.grad_accum}')
        self.print0(f'Adjusted local batch size: {self.batch_size} tokens')
        self.print0(f'Across {self.ddp_world_size} GPUs')
        self.print0(f'Total batch size: {self.args.batch_size} tokens')

        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        # Special tokens tensor for GPU-side masking (moved to GPU lazily)
        self._special_tokens_cpu = torch.tensor(
            [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.pad_token_id],
            dtype=torch.int32,
        )

        # Ensure dataset is available locally (master process only), then sync
        if self.master_process:
            self.print0(f"Ensuring dataset '{self.args.data_name}' is available (num_chunks={self.args.num_chunks})...")
            try:
                ensure_hf_file(f"{self.args.data_name}_valid_%06d.bin" % 0, self.args.data_name)
                ensure_hf_file(f"{self.args.data_name}_test_%06d.bin" % 0, self.args.data_name)
                for i in tqdm(range(0, self.args.num_chunks + 1), desc="Ensuring dataset chunks"):
                    ensure_hf_file(f"{self.args.data_name}_train_%06d.bin" % i, self.args.data_name)
            except Exception as e:
                self.print0(f"Dataset ensure failed: {e}")
        if self.ddp_world_size > 1:
            dist.barrier()

        self.train_loader = self.init_dataloader(self.args.input_bin, training=True)
        self.valid_loader = self.init_dataloader(self.args.input_valid_bin, training=False)
        self.test_loader = self.init_dataloader(self.args.input_test_bin, training=False)

        self.print0(f'Training DataLoader: {len(self.train_loader.files)} files')
        self.print0(f'Validation DataLoader: {len(self.valid_loader.files)} files')
        self.print0(f'Testing DataLoader: {len(self.test_loader.files)} files')
        self.print0('='*100, logonly=True)

        if self.master_process:
            train_files = sorted(Path.cwd().glob(self.args.input_bin))
            self.total_downloaded_tokens = sum(self._read_bin_num_tokens(f) for f in train_files)
        else:
            self.total_downloaded_tokens = 0
        if self.ddp_world_size > 1:
            total_tokens_tensor = torch.tensor(self.total_downloaded_tokens, device=self.device)
            dist.broadcast(total_tokens_tensor, 0)
            self.total_downloaded_tokens = int(total_tokens_tensor.item())
        self.epoch_counter = 1

        self.model = self.init_model()
        self.print0(summary(self.model))
        
        # Initialize auto gradient clipper if enabled
        if self.args.auto_grad_clip:
            model_for_clipper = self.model.module if self.ddp_world_size > 1 else self.model
            self.auto_grad_clipper = AutoGradClipper(
                model=model_for_clipper,
                clip_percentile=self.args.auto_grad_clip_p,
            )
            self.print0(f"Auto gradient clipping enabled with {self.args.auto_grad_clip_p}% percentile")
        
        self.optimizers = self.init_optimizers()
        self.lr_schedulers, self.sliding_window_size_scheduler, self.mask_rate_scheduler = self.init_schedulers()
        self.print0(f"Ready for training!")

        # Push code + config to HF Hub once so the repo is ready for inference
        if self.master_process and self.args.hf_model_name:
            self.print0(f"Pushing code and config to {self.args.hf_model_name}...")
            model_ref = self.model.module if self.ddp_world_size > 1 else self.model
            model_ref.push_code_and_config_to_hub(self.args.hf_model_name)
            self.print0("Code and config pushed to hub.")
        
        # Create decorated versions of methods that should be excluded from timing
        self._run_eval_loader_timed = exclude_from_timer(self.train_timer)(self.run_eval_loader)
        self._save_checkpoint_timed = exclude_from_timer(self.train_timer)(self.save_checkpoint)

    def init_dataloader(self, filename_pattern, training=True):
        if self.args.patch_unet:
            # Chunked loader for batched UNet: yields (B, max_length) raw input_ids
            if training:
                loader = ChunkedTrainLoader(
                    filename_pattern=filename_pattern,
                    max_length=self.args.max_length,
                    micro_batch_tokens=self.batch_size,
                    process_rank=self.ddp_rank,
                    num_processes=self.ddp_world_size,
                    max_epochs=1,
                    tokenizer=self.tokenizer,
                    num_workers=self.args.num_workers,
                    prefetch_factor=self.args.prefetch_factor,
                )
                return AsyncBatchPipeline(loader)
            else:
                loader = ChunkedEvalLoader(
                    filename_pattern=filename_pattern,
                    max_length=self.args.max_length,
                    micro_batch_tokens=self.batch_size,
                    process_rank=self.ddp_rank,
                    num_processes=self.ddp_world_size,
                    tokenizer=self.tokenizer,
                )
                return AsyncBatchPipeline(loader)
        else:
            # Legacy loader for standard/unet: yields (input_ids, labels, mask_rate)
            if training:
                if self.args.mlm:
                    mask_rate = self.args.mask_rate
                else:
                    mask_rate = 1.0
                return OptimizedTrainLoader(
                    filename_pattern=filename_pattern,
                    seq_len=self.batch_size,
                    process_rank=self.ddp_rank,
                    num_processes=self.ddp_world_size,
                    max_epochs=1,
                    tokenizer=self.tokenizer,
                    num_workers=self.args.num_workers,
                    prefetch_factor=self.args.prefetch_factor,
                    mlm=self.args.mlm or self.args.masked_diffusion,
                    mask_rate=mask_rate,
                )
            else:
                return OptimizedEvalLoader(
                    filename_pattern=filename_pattern,
                    seq_len=self.batch_size,
                    process_rank=self.ddp_rank,
                    num_processes=self.ddp_world_size,
                    tokenizer=self.tokenizer,
                )

    def init_model(self):
        self.print0("Initializing model...")
        model = PLM(self.model_config)
        self.print0(model)
        model = model.cuda()
        if self.args.bfloat16:
            model = model.bfloat16()
            
        # Synchronize before compilation
        if self.ddp_world_size > 1:
            dist.barrier()

        if self.args.compile_model:
            self.print0("Calling torch.compile()")
            torch._dynamo.config.recompile_limit = self.args.dynamo_recompile_limit
            model = torch.compile(model)
        else:
            self.print0("Skipping torch.compile()")
        
        if self.ddp_world_size > 1:
            # Use static graph if model architecture doesn't change
            model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        return model

    def init_optimizers(self):
        self.print0("Initializing optimizers...")
        if self.args.use_muon:
            matrix_params = [
                p for n, p in self.model.named_parameters() 
                if p.ndim >= 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
            ]
            embed_params = [
                p for n, p in self.model.named_parameters() if "embed" in n.lower() and p.requires_grad
            ]
            head_params = [
                p for n, p in self.model.named_parameters() if "lm_head" in n.lower() and p.requires_grad
            ]
            scalar_params = [
                p for n, p in self.model.named_parameters() 
                if p.ndim < 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
            ]
            
            # Confirm every parameter is mapped to an optimizer
            all_params = [p for p in self.model.parameters() if p.requires_grad]
            mapped_params = matrix_params + embed_params + head_params + scalar_params
            assert len(all_params) == len(mapped_params), f"Muon parameter mapping mismatch: {len(all_params)} total vs {len(mapped_params)} mapped"
            self.print0(f"Muon optimizer initialized: {len(matrix_params)} matrix, {len(embed_params)} embed, {len(head_params)} head, {len(scalar_params)} scalar params. Total: {len(all_params)}")

            optimizer1 = torch.optim.Adam([
                dict(params=embed_params, lr=self.args.lr_embed),
                dict(params=head_params, lr=self.args.lr_head),
                dict(params=scalar_params, lr=self.args.lr_scalar),
            ], betas=(0.8, 0.95), fused=True)
            optimizer2 = Muon(matrix_params, lr=self.args.lr_hidden, momentum=0.95)
            optimizers = [optimizer1, optimizer2]
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.print0(f"AdamW optimizer initialized with {len(params)} parameters.")
            optimizer = torch.optim.AdamW(params, lr=self.args.lr)
            optimizers = [optimizer]
        return optimizers
    
    def init_schedulers(self):
        self.print0("Initializing schedulers...")
        lr_schedulers = []
        adam_scheduler = get_scheduler(
            self.args.scheduler_type,
            optimizer=self.optimizers[0],
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.num_steps
        )
        lr_schedulers.append(adam_scheduler)
        if self.args.use_muon:
            muon_scheduler = get_scheduler(
                self.args.scheduler_type,
                optimizer=self.optimizers[-1],
                num_warmup_steps=0, # apparently muon does not need a warmup
                num_training_steps=self.args.num_steps
            )
            lr_schedulers.append(muon_scheduler)
        sliding_window_size_scheduler = LerpTensor(start_val=1024, end_val=self.args.max_length, precision=128)
        if self.args.mask_rate_schedule:
            mask_rate_scheduler = LerpFloat(
                start_val=self.args.starting_mask_rate, 
                end_val=self.args.mask_rate,
                precision=0.01
            )
        else:
            mask_rate_scheduler = None
        return lr_schedulers, sliding_window_size_scheduler, mask_rate_scheduler

    @torch.no_grad()
    def run_eval_loader(self, loader, prefix='val'): # returns loss, tokens
        # Synchronize before evaluation
        if self.ddp_world_size > 1:
            dist.barrier()
            
        loader.reset()
        self.model.eval()

        # Move special tokens to GPU once
        special_tokens_gpu = self._special_tokens_cpu.to(self.device)

        losses, total_tokens = [], 0
        confusion = torch.zeros((self.args.vocab_size, self.args.vocab_size), dtype=torch.int64)
        preview_done = False

        if self.args.patch_unet:
            # Chunked loader: yields (B, max_length) raw input_ids on GPU
            raw_ids = loader.next_batch()
        else:
            # Legacy loader: yields (input_ids, labels, mask_rate) on GPU
            input_ids, labels, mask_rate = loader.next_batch()
            raw_ids = input_ids  # Use input_ids for the loop condition
        
        # Only show progress bar on master process
        pbar = tqdm(desc=f'{prefix} set', leave=False, disable=not self.master_process)
        
        while raw_ids.numel():
            if self.args.patch_unet:
                # Apply masking on GPU with fixed eval mask rate
                input_ids, labels, mask_rate = apply_masking_gpu(
                    raw_ids, special_tokens_gpu, self.mask_token_id, mask_rate=0.15, mlm=True,
                )
            batch_valid_tokens = (input_ids != self.pad_token_id).sum()
            total_tokens += batch_valid_tokens
            loss, logits = self.model(
                input_ids=input_ids,
                labels=labels,
                mask_rate=mask_rate,
                sliding_window_size=self.sliding_window_size,
                return_logits=True,
            )
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            self._update_confusion(confusion, preds.detach(), labels.detach())
            if not preview_done:
                self._print_val_preview(input_ids, labels, logits)
                preview_done = True

            if self.args.patch_unet:
                raw_ids = loader.next_batch()
            else:
                input_ids, labels, mask_rate = loader.next_batch()
                raw_ids = input_ids
            pbar.update(1)
        pbar.close()

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        metrics = self._calculate_metrics_from_confusion(confusion)

        if self.ddp_world_size > 1:
            # Convert to tensors before all_reduce
            avg_loss = torch.tensor(avg_loss, device=self.device)
            total_tokens = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            # Ensure all processes finish evaluation
            dist.barrier()

        perplexity = math.e**avg_loss if isinstance(avg_loss, float) else math.e**avg_loss.item()

        self.print0(
            f'{prefix} set: loss: {avg_loss:.4f} perplexity: {perplexity:.4f} '
            f'tokens: {total_tokens.item() if hasattr(total_tokens, "item") else total_tokens:,}'
        )
        self.print0(
            f"{prefix} metrics: acc:{metrics['accuracy']:.4f} prec:{metrics['precision']:.4f} "
            f"rec:{metrics['recall']:.4f} f1:{metrics['f1']:.4f} mcc:{metrics['mcc']:.4f} "
            f"tokens:{metrics['num_tokens']:,}"
        )

        return avg_loss, perplexity, total_tokens, metrics

    def save_checkpoint(self, step):
        # Only master saves, but all processes wait
        if self.master_process:
            self.print0(f'Saving checkpoint at step {step}...')

            if self.ddp_world_size > 1:
                model = self.model.module
            else:
                model = self.model

            # Always save locally
            log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in self.optimizers])
            os.makedirs('logs', exist_ok=True)
            torch.save(log, 'logs/state_step%06d.pt' % step)
            model.save_weights_local('checkpoints', step)
            self.print0(f'Checkpoint saved locally at step {step}')
        
        # Synchronize after saving
        if self.ddp_world_size > 1:
            dist.barrier()

    def train_step(self, step):
        self.model.train()
        
        # Clear cache periodically to prevent memory fragmentation
        if step % self.args.clear_cache_every == 0:
            torch.cuda.empty_cache()
        
        # Move special tokens to GPU once (cached after first call)
        if not hasattr(self, '_special_tokens_gpu'):
            self._special_tokens_gpu = self._special_tokens_cpu.to(self.device)
        
        # Accumulate losses for proper averaging
        accumulated_loss = 0.0
        
        for i in range(self.args.grad_accum):
            with contextlib.ExitStack() as stack:
                # Only sync gradients on last accumulation step
                if self.ddp_world_size > 1 and i < self.args.grad_accum - 1:
                    stack.enter_context(self.model.no_sync())
                
                if self.args.patch_unet:
                    # Chunked pipeline: yields raw (B, max_length) on GPU
                    raw_ids = self.train_loader.next_batch()
                    if raw_ids.numel() == 0:
                        self.train_loader.reset()
                        raw_ids = self.train_loader.next_batch()
                        assert raw_ids.numel() > 0, "Dataloader returned empty batch even after reset"
                    # Apply masking on GPU
                    input_ids, labels, mask_rate = apply_masking_gpu(
                        raw_ids,
                        self._special_tokens_gpu,
                        self.mask_token_id,
                        mask_rate=self.current_mask_rate,
                        mlm=self.args.mlm or (self.args.masked_diffusion and self.current_mask_rate < 1.0),
                    )
                else:
                    # Legacy pipeline: yields (input_ids, labels, mask_rate) on GPU
                    input_ids, labels, mask_rate = self.train_loader.next_batch()
                    if input_ids.numel() == 0:
                        self.train_loader.reset()
                        input_ids, labels, mask_rate = self.train_loader.next_batch()
                        assert input_ids.numel() > 0, "Dataloader returned empty batch even after reset"
                
                loss = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    mask_rate=mask_rate,
                    sliding_window_size=self.sliding_window_size,
                    return_logits=False,
                ) / self.args.grad_accum
                loss.backward()
                accumulated_loss += loss.item()  # Accumulate the scaled loss

        # momentum warmup for Muon
        if self.args.use_muon:
            frac = min(step/self.args.muon_momentum_warmup_steps, 1)
            for group in self.optimizers[-1].param_groups:
                group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        # Apply gradient clipping if specified
        clip_value = None
        if self.args.auto_grad_clip and self.auto_grad_clipper is not None:
            # Use auto gradient clipping
            clip_value = self.auto_grad_clipper.clip_gradients()
        elif self.args.grad_clip > 0:
            # Use regular gradient clipping
            if self.ddp_world_size > 1:
                clip_grad_norm_(self.model.module.parameters(), self.args.grad_clip)
            else:
                clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            clip_value = self.args.grad_clip

        # step the optimizers and schedulers
        for opt, sched in zip(self.optimizers, self.lr_schedulers):
            opt.step()
            sched.step()

        # null the gradients
        self.model.zero_grad(set_to_none=True)
        
        # Store clip value for logging
        self.last_clip_value = clip_value
        
        # Return the total accumulated loss (already properly scaled)
        return accumulated_loss

    def train(self):
        self.init_training()

        train_losses = []

        ### BEGIN TRAINING LOOP ###
        self.print0("Beginning training loop...")
        
        # Synchronize before starting training
        if self.ddp_world_size > 1:
            dist.barrier()
        
        # Show progress only on master
        pbar = tqdm(range(self.args.num_steps + 1), desc='Training steps', disable=not self.master_process)
        
        try:
            for step in pbar:
                if step == 10: # ignore first 10 steps of timing because they are slower
                    self.train_timer.reset()
                    self.train_timer.start()
                timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

                frac_done = step / self.args.num_steps  # training progress
                if frac_done > 1:
                    self.sliding_window_size = self.args.max_length
                else:
                    self.sliding_window_size = self.sliding_window_size_scheduler(frac_done)
                
                if self.mask_rate_scheduler:
                    frac_done_mask = step / self.args.mask_rate_steps
                    if frac_done_mask > 1:
                        mask_rate = self.args.mask_rate
                    else:
                        mask_rate = self.mask_rate_scheduler(frac_done_mask)
                    self.current_mask_rate = mask_rate
                    if self.args.patch_unet:
                        # For patch_unet, mask_rate is applied in train_step via apply_masking_gpu
                        if self.args.masked_diffusion and frac_done_mask > 1:
                            model_for_mlm = self.model.module if self.ddp_world_size > 1 else self.model
                            model_for_mlm.mlm = False
                    else:
                        # Legacy path: push mask_rate to data loader workers
                        self.train_loader.set_mask_rate(mask_rate)
                        if self.args.masked_diffusion and frac_done_mask > 1 and self.train_loader.mlm:
                            self.train_loader.set_mlm(False)
                            model_for_mlm = self.model.module if self.ddp_world_size > 1 else self.model
                            model_for_mlm.mlm = False
                # once in a while evaluate the validation dataset
                if self.args.eval_every > 0 and step % self.args.eval_every == 0:
                    val_loss, val_perplexity, val_tokens, val_metrics = self._run_eval_loader_timed(
                        self.valid_loader, prefix='Validation'
                    )
                    training_time_sec = self.train_timer.get_time()
                    step_avg_ms = 1000 * training_time_sec / (timed_steps - 1) if timed_steps > 1 else 0
                    self.print0(f'step:{step}/{self.args.num_steps} step_avg:{step_avg_ms:.2f}ms')
                    tokens_seen = (step + 1) * self.args.batch_size
                    epoch_progress = tokens_seen / max(self.total_downloaded_tokens, 1)
                    current_epoch = int(epoch_progress) + 1
                    if current_epoch != self.epoch_counter:
                        self.print0(f"(MOVING FROM EPOCH {self.epoch_counter} TO EPOCH {current_epoch})")
                        self.epoch_counter = current_epoch
                    self.print0(
                        f"Epoch progress: {epoch_progress:.4f} "
                        f"({tokens_seen:,}/{self.total_downloaded_tokens:,} tokens)"
                    )
                    self.log_wandb(
                        {
                            'loss': val_loss,
                            'perplexity': val_perplexity,
                            'tokens': val_tokens,
                            'sliding_window_size': self.sliding_window_size,
                            'accuracy': val_metrics['accuracy'],
                            'precision': val_metrics['precision'],
                            'recall': val_metrics['recall'],
                            'f1': val_metrics['f1'],
                            'mcc': val_metrics['mcc'],
                            'epoch_progress': epoch_progress,
                        },
                        prefix='val'
                    )

                # save checkpoint every `save_every` steps
                if self.args.save_every:
                    if step % self.args.save_every == 0:
                        self._save_checkpoint_timed(step)

                loss = self.train_step(step)
                train_losses.append(loss)

                # everything that follows now is just eval, diagnostics, prints, logging, etc.
                if step % 100 == 0:
                    train_time_sec = self.train_timer.get_time()
                    avg_loss = sum(train_losses) / len(train_losses)
                    
                    # Gather training loss across all processes for accurate logging
                    if self.ddp_world_size > 1:
                        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
                        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                        avg_loss = avg_loss_tensor.item()
                    
                    log_msg = f'step:{step+1}/{self.args.num_steps} train_time:{train_time_sec:.0f} sec step_avg:{1000*train_time_sec/timed_steps:.2f}ms loss:{avg_loss:.4f} mask_rate:{self.current_mask_rate:.4f}'
                    if hasattr(self, 'last_clip_value') and self.last_clip_value is not None:
                        log_msg += f' clip_value:{self.last_clip_value:.4f}'
                    self.print0(log_msg)
                    train_losses = []

                    # Log training progress to wandb
                    if self.master_process and self.wandb_initialized:
                        log_dict = {
                            "time_sec": train_time_sec,
                            "step_avg_ms": 1000*train_time_sec/timed_steps if timed_steps > 0 else 0,
                            "step": step,
                            "loss": avg_loss,
                            "mask_rate": self.current_mask_rate
                        }
                        if hasattr(self, 'last_clip_value') and self.last_clip_value is not None:
                            log_dict["clip_value"] = self.last_clip_value
                        self.log_wandb(log_dict, prefix='train')

            # Stop the timer and get final training time
            self.train_timer.pause()
            final_training_time_sec = self.train_timer.get_time()
            
            self.print0(f'peak memory consumption training: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB')
            self.print0(f'Train Time: {final_training_time_sec:.0f}s | Step Avg: {final_training_time_sec/timed_steps:.2f}s')
            self.print0(f'Total train time (min): {final_training_time_sec / 60:.2f}')
            self.print0(f'Total train time (hours): {final_training_time_sec / 3600:.2f}')
            # Save final checkpoint locally
            self._save_checkpoint_timed(self.args.num_steps)
            # Push final weights to HF Hub
            if self.master_process and self.args.hf_model_name:
                self.print0(f"Pushing final weights to {self.args.hf_model_name}...")
                model_ref = self.model.module if self.ddp_world_size > 1 else self.model
                model_ref.push_weights_to_hub(self.args.hf_model_name)
                self.print0("Final weights pushed to hub.")

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            set_seed(self.args.seed)

            test_loss, test_perplexity, test_tokens, test_metrics = self._run_eval_loader_timed(
                self.test_loader, prefix='Test'
            )

            self.print0(f"peak memory consumption testing: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB")
        
            # Final wandb logging
            if self.master_process and self.wandb_initialized:
                log_dict = {
                    "test_loss": test_loss,
                    "test_perplexity": test_perplexity,
                    "test_tokens": test_tokens.item() if hasattr(test_tokens, "item") else test_tokens,
                    "test_accuracy": test_metrics['accuracy'],
                    "test_precision": test_metrics['precision'],
                    "test_recall": test_metrics['recall'],
                    "test_f1": test_metrics['f1'],
                    "test_mcc": test_metrics['mcc'],
                    "final_train_time_sec": final_training_time_sec,
                    "final_step_avg_sec": final_training_time_sec/(timed_steps-1) if timed_steps > 1 else 0,
                    "peak_memory_training_gb": torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024,
                }
                self.log_wandb(log_dict, prefix='test')

            # Log final summary
            log_dict = {
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_perplexity": test_perplexity,
                "train_time_sec": final_training_time_sec,
                "step_avg_sec": final_training_time_sec/(timed_steps-1) if timed_steps > 1 else 0,
                "peak_memory_training_gb": torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024,
            }
            self.log_wandb(log_dict, prefix='final')
            
        except KeyboardInterrupt:
            self.print0("\nTraining interrupted by user!")
        except Exception as e:
            self.print0(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources
            if self.master_process and self.wandb_initialized:
                wandb.finish()
        
            # clean up nice
            if self.ddp_world_size > 1:
                dist.destroy_process_group()


if __name__ == '__main__':
    args = arg_parser()

    if args.bugfix:
        args.hidden_size = 128
        args.num_attention_heads = 2
        args.num_hidden_layers = 2
        args.expansion_ratio = 2.0
        args.soft_logit_cap = 16.0
        args.tie_embeddings = False
        args.unet = True
        args.batch_size = 2048
        args.grad_accum = 1
        args.num_steps = 10
        args.cooldown_steps = 2
        args.max_length = 512
        args.auto_grad_clip = True
        args.grad_clip = 0.0  # Disable regular grad clip for bugfix testing

    # Validate mode arguments
    if args.mlm and args.masked_diffusion:
        raise ValueError("Only one of --mlm or --masked_diffusion can be true.")
    # Validate gradient clipping arguments
    if args.auto_grad_clip and args.grad_clip > 0:
        raise ValueError("Cannot use both --auto_grad_clip and --grad_clip at the same time. Choose one.")
    
    model_config = PLMConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        num_unet_layers=args.num_unet_layers,
        num_extra_layers=args.num_extra_layers,
        max_sequence_length=args.max_length,
        vocab_size=args.vocab_size,
        expansion_ratio=args.expansion_ratio,
        soft_logit_cap=args.soft_logit_cap,
        tie_embeddings=args.tie_embeddings,
        unet=args.unet,
        patch_unet=args.patch_unet,
        mlm=args.mlm or args.masked_diffusion,
        masked_diffusion=args.masked_diffusion,
        token_dropout=args.token_dropout,
        compile_flex_attention=args.compile_flex_attention,
    )

    # Initialize wandb before clearing tokens for security
    wandb_initialized = False
    if args.wandb_token and os.environ['WANDB_AVAILABLE'] == 'true':
        wandb.login(key=args.wandb_token)
        wandb_initialized = True
    
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)
        # Clear tokens for security
        args.hf_token = None
    
    # Clear wandb token for security but keep track that we logged in
    if args.wandb_token:
        args.wandb_token = None
    
    trainer = Trainer(args, model_config)
    trainer.wandb_initialized = wandb_initialized
    trainer.train()
