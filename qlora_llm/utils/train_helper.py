# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional, Union
import math
import torch
from torch import nn
import torch.distributed as dist
import bitsandbytes as bnb

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def create_trace_profiler(tb_trace_dir: str) -> torch.profiler.profile:
    torch_profiler = torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=False,
    )

    return torch_profiler


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    eps: float,
    weight_decay: float,
    betas: Tuple[float],
    paged_adamw: bool = False,
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    optim_params = []
    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
            optim_params.append(params)

    kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
        'eps': eps,
        'betas': betas,
    }
    if paged_adamw:
        optimizer = bnb.optim.PagedAdamW(optim_params, **kwargs)
    else:
        optimizer = torch.optim.AdamW(optim_params, **kwargs)

    return optimizer


def compute_num_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    num_trainable_params = 0
    num_frozen_params = 0

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad
        is_quantized = hasattr(params, 'quant_state')

        # quantized layer is not trainable
        if not is_trainable and is_quantized:
            num_params = math.prod(params.quant_state.shape)
        else:
            num_params = params.numel()

        num_trainable_params += num_params if is_trainable else 0
        num_frozen_params += num_params if not is_trainable else 0

    return num_trainable_params, num_frozen_params


def get_grad_norm_local(model: torch.nn.Module) -> torch.Tensor:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


# GPU memory usage
gigabyte_size = 1073741824
megabyte_size = 1048576


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def get_gpu_ram_usage_in_gb() -> float:
    updated_reserved = torch.cuda.memory_reserved()
    updated_reserved = format_to_gb(updated_reserved)
    return updated_reserved
