# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Run SFT, starting with Meta's pretrained model."""
import os
import itertools
import functools
from typing import Tuple, Mapping, Text, Any
import tqdm
import random
import time
import numpy as np
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from qlora_llm.model_lora import Transformer, LoraModelArgs
from qlora_llm.configs.finetune_lora import config as cfg
from qlora_llm.utils.schedule import CosineDecayWithWarmupLRScheduler
from qlora_llm.utils.custom_dataset import FineTuneDataset

from qlora_llm.tokenizer import Tokenizer

from qlora_llm.utils.logging import create_logger
from qlora_llm.lora import (
    lora_state_dict,
    mark_only_lora_as_trainable,
)
from qlora_llm.utils.train_helper import (
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_local,
    get_gpu_ram_usage_in_gb,
)


logger = create_logger()


def clear_gpu_cache():
    torch.cuda.empty_cache()


def create_lora_checkpoint(model, save_file):
    full_path = os.path.join(cfg.ckpt_dir, save_file)
    ckpt_state = lora_state_dict(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    torch.save(ckpt_state, full_path)
    logger.info(f'LoRA model checkpoint saved at {full_path!r}')

    # save LoRA configuration params so we can re-create the same model after training, which is required to merge weights
    meta_file = os.path.join(cfg.ckpt_dir, 'lora_params.json')
    if not os.path.exists(meta_file):
        meta = {
            'lora_r': cfg.lora_r,
            'lora_scaling': cfg.lora_scaling,
            'lora_dropout': cfg.lora_dropout,
            'lora_attn_query': cfg.lora_attn_query,
            'lora_attn_key': cfg.lora_attn_key,
            'lora_attn_value': cfg.lora_attn_value,
            'lora_attn_proj': cfg.lora_attn_proj,
            'lora_attn_mlp': cfg.lora_attn_mlp,
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        logger.info(f'LoRA model params saved at {meta_file!r}')


def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]

    B, T, *_ = logits.shape

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    assert not torch.any(torch.isnan(loss))

    loss = loss.view(B, T)

    assert loss.shape == mask.shape

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, cfg.prompt_loss_weight).masked_fill(mask == 1, cfg.completion_loss_weight)
    loss *= weights

    loss = torch.mean(loss)

    return loss


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert targets.shape == mask.shape  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)


def run_single_train_step(
    model: Transformer,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    return_stats: bool = False,
) -> Mapping[Text, Any]:
    """A single training iteration consists of N micro batch * M gradient accumulation steps.

    ```
    optimizer.zero_grad()
    for step in range(gradient_accum_steps):
        data, target = next(iter(train_loader))
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """

    if return_stats:
        metrics = torch.zeros(5).to('cuda')
        t_start = time.perf_counter()

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, y, loss_mask in itertools.islice(train_loader, cfg.gradient_accum_steps):
        x, y, loss_mask = (
            x.to('cuda', non_blocking=True),
            y.to('cuda', non_blocking=True),
            loss_mask.to('cuda', non_blocking=True),
        )

        output = model(x)

        loss = compute_finetune_loss(output, y, loss_mask)

        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps
        scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(output, y, loss_mask)
            metrics[0] += loss.item()  # sum up batch loss
            metrics[1] += np.exp(loss.item())  # sum up perplexity
            metrics[2] += 1  # increase number of micro batches
            metrics[3] += num_acc  # sum up number of accurate prediction tokens
            metrics[4] += num_samples  # sum up number of tokens

    if return_stats:
        grad_norm = get_grad_norm_local(model)

    if cfg.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    optimizer.step()
    scheduler.step()

    if return_stats:
        train_loss = metrics[0] / metrics[2]
        train_perplexity = metrics[1] / metrics[2]
        train_accuracy = 100 * metrics[3] / metrics[4]
        t_iter = time.perf_counter() - t_start

        return {
            'loss': train_loss.item(),
            'accuracy': train_accuracy.item(),
            'perplexity': train_perplexity.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
            'gpu_ram_gb': get_gpu_ram_usage_in_gb(),
            'time(s)_per_iter': round(t_iter, 4),
        }
    else:
        return None


@torch.no_grad()
def run_validation_steps(model: Transformer, val_loader: DataLoader) -> Mapping[Text, Any]:
    """Run M validation iterations"""
    model.eval()  # set model in validation mode

    metrics = torch.zeros(5).to('cuda')

    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour='green', desc='validation iterations')

    for x, y, loss_mask in itertools.islice(val_loader, cfg.val_iters):
        x, y, loss_mask = (
            x.to('cuda', non_blocking=True),
            y.to('cuda', non_blocking=True),
            loss_mask.to('cuda', non_blocking=True),
        )

        output = model(x)

        loss = compute_finetune_loss(output, y, loss_mask)
        num_acc, num_samples = compute_metrics(output, y, loss_mask)
        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += np.exp(loss.item())  # sum up perplexity
        metrics[2] += 1  # increase number of micro batches
        metrics[3] += num_acc  # sum up number of accurate prediction tokens
        metrics[4] += num_samples  # sum up number of tokens

        if inner_pbar is not None:
            inner_pbar.update(1)

    val_loss = metrics[0] / metrics[2]
    val_perplexity = metrics[1] / metrics[2]
    val_accuracy = 100 * metrics[3] / metrics[4]

    inner_pbar.close()

    model.train()  # set model in training mode after validation runs

    return {'loss': val_loss.item(), 'accuracy': val_accuracy.item(), 'perplexity': val_perplexity.item()}


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    and compute the loss mask for the batch.
    """

    batch_size = len(batch)

    max_batch_seq_len = max([len(item[0]) + len(item[1]) for item in batch])
    assert max_batch_seq_len <= max_seq_len

    if full_pad:
        max_batch_seq_len = max_seq_len

    # concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

    # loss mask where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    loss_mask = torch.full((batch_size, max_batch_seq_len), 0, dtype=torch.long)

    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)
        seq_len = prompt_len + completion_len
        seq = torch.concat((prompt, completion), dim=0).type(torch.long)

        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_sequences[i, :seq_len] = seq
        loss_mask[i, :prompt_len] = -1  # prompt tokens
        loss_mask[i, prompt_len : prompt_len + completion_len] = 1  # completion tokens

    x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_len - 1]
    y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_len - 1]

    # shift to right to align with y
    loss_mask = loss_mask[:, 1:]

    return x, y, loss_mask


def main():
    assert cfg.num_epochs >= 1
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 1
    assert cfg.val_iters >= 1

    batch_size = int(cfg.micro_batch_size * cfg.gradient_accum_steps)

    assert batch_size >= 1

    if not torch.version.cuda or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script requires Pytorch with CUDA and bfloat16 data type.')

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint {cfg.pretrain_ckpt_file!r}, aborting...')

    # --------------- Load datasets ---------------

    tokenizer = Tokenizer(cfg.tokenizer_file)

    logger.info('Loading datasets...')

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': cfg.dataloader_workers,
        'pin_memory': True,
        'shuffle': False,
    }

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    train_kwargs = {'batch_size': cfg.micro_batch_size, 'sampler': None}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    num_train_iters = int((len(train_dataset) / batch_size) * cfg.num_epochs)

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = FineTuneDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_kwargs = {'batch_size': cfg.val_batch_size, 'sampler': None}
        val_kwargs.update(cuda_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    clear_gpu_cache()

    torch.cuda.set_device('cuda:0')

    compute_dtype = torch.bfloat16
    torch.set_default_dtype(compute_dtype)

    logger.info('Initializing model...')

    model_args = LoraModelArgs.from_model_type(
        model_type=cfg.model_type,
        # LoRA configurations
        lora_r=cfg.lora_r,
        lora_scaling=cfg.lora_scaling,
        lora_dropout=cfg.lora_dropout,
        # LoRA trainable layers
        lora_attn_query=cfg.lora_attn_query,
        lora_attn_key=cfg.lora_attn_key,
        lora_attn_value=cfg.lora_attn_value,
        lora_attn_proj=cfg.lora_attn_proj,
        lora_attn_mlp=cfg.lora_attn_mlp,
        # Quantization configurations
        quant_4bit=cfg.quant_4bit,
        quant_lora_4bit=cfg.quant_lora_4bit,
        quant_4bit_double=cfg.quant_4bit_double,
        quant_4bit_type=cfg.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights not contained in checkpoint state
    if os.path.exists(cfg.pretrain_ckpt_file):
        logger.info(f'Loading pretrained checkpoint {cfg.pretrain_ckpt_file!r}...')
        model_state = torch.load(cfg.pretrain_ckpt_file)
        model.load_state_dict(model_state, strict=False)
        del model_state

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'token_embeddings' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype != compute_dtype:
                    module = module.to(compute_dtype)

    # This is where the weights quantization happens
    # when we move the model to cuda, the bnb.nn.Params4bit.cuda() method is called,
    # and the weights is quantized using bnb.functional.quantize_4bit
    model = model.to('cuda')

    logger.info('Initializing optimizer...')

    num_trainable, num_frozen = compute_num_trainable_params(model)
    logger.info(f'Number of trainable parameters: {num_trainable:,}')
    logger.info(f'Number of frozen parameters: {num_frozen:,}')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.adamw_eps,
        betas=cfg.adamw_betas,
        paged_adamw=True,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=int(cfg.warmup_ratio * num_train_iters),
        max_decay_steps=num_train_iters,
    )

    # --------------- Start Training ---------------

    logger.info(
        f'Starting to run {cfg.num_epochs} training epochs, total of {num_train_iters} iterations, with batch size {batch_size}'
    )

    torch_profiler = None
    tb_writer = None
    best_val_accuracy = 0.0
    inner_pbar = tqdm.tqdm(range(num_train_iters), colour='blue', desc='Training iterations')

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    if cfg.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    model.train()
    for i in range(1, num_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            return_stats=i == 1 or i % cfg.log_interval == 0 or i == num_train_iters,
        )

        inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None:
            logger.info(
                f'Training iteration {i}: train loss: {train_stats["loss"]:.4f}, '
                f'train accuracy: {train_stats["accuracy"]:.2f}%, train perplexity: {train_stats["perplexity"]:.2f}, learning rate: {train_stats["learning_rate"]:.6f}'
            )

            if tb_writer is not None:
                for k, v in train_stats.items():
                    tb_writer.add_scalar(f'train/{k}', v, i)

        # regular checkpointing
        if cfg.ckpt_interval > 0 and (i % cfg.ckpt_interval == 0 or i == num_train_iters):
            create_lora_checkpoint(model, f'lora_{cfg.model_type}-iter-{i}.pth')

        if cfg.val_iters > 0 and (i % cfg.val_interval == 0 or i == num_train_iters):
            val_stats = run_validation_steps(model=model, val_loader=val_loader)

            logger.info(
                f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, '
                f'validation accuracy: {val_stats["accuracy"]:.2f}%, validation perplexity: {val_stats["perplexity"]:.2f}'
            )

            if tb_writer is not None:
                for k, v in val_stats.items():
                    tb_writer.add_scalar(f'val/{k}', v, i)

            # save best model
            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                # save model state
                create_lora_checkpoint(model, f'lora_{cfg.model_type}-best.pth')

    # show some training stats.
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
