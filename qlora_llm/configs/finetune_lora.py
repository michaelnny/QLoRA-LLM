# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
from dataclasses import dataclass


@dataclass
class config:
    """Supervised fine-tuning using LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    max_seq_len: int = 512  # use smaller sequence length to save GPU RAM

    pretrain_ckpt_file: str = './meta_checkpoints/llama-2-7b/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = './meta_checkpoints/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        './datasets/dolly/train.pkl',
        './datasets/alpaca/train.pkl',
        './datasets/squad/train.pkl',
        './datasets/commonsense_dialogues/train.pkl',
    )
    val_datasources: Tuple[str] = (
        './datasets/dolly/validation.pkl',
        './datasets/alpaca/validation.pkl',
        './datasets/squad/validation.pkl',
        './datasets/commonsense_dialogues/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 2
    # accumulate gradients so the actual batch size is = micro_batch_size x gradient_accum_steps
    micro_batch_size: int = 2
    gradient_accum_steps: int = 30
    val_interval: int = 100
    val_batch_size: int = 30
    val_iters: int = 20
    log_interval: int = 10  # log training metrics (loss, accuracy)
    ckpt_interval: int = 200  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 64
    lora_scaling: float = 1.0  # we don't use alpha here, instead directly set the scaling
    lora_dropout: float = 0.0

    # LoRA trainable layers
    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = True  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = True  # train Attention projection layer
    lora_attn_mlp: bool = True  # train Attention MLP block

    train_bias: str = 'none'  # none, lora_only, all
    train_head: bool = True  # do not apply LoRA or quantize to the lm_head layer

    # Quantization
    quant_4bit: bool = True  # quantize frozen linear layer
    quant_lora_4bit: bool = True  # quantize LoRA linear layer
    quant_4bit_double: bool = True  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # learning rate
    init_lr: float = 2e-6  # initial learning rate
    max_lr: float = 2e-5  # max learning rate after warm up
    min_lr: float = 2e-6  # min learning rate after decay
    warmup_ratio: float = 0.05

    # prompt is lesser important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    weight_decay: float = 0.0
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-5
    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    # others
    seed: int = 127
    log_dir: str = './logs/finetune_lora-4bit'  # save logs and traces
    ckpt_dir: str = './checkpoints/finetune_lora-4bit'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
