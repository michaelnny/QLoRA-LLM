# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file"""

import os
import shutil
import json
import torch
import torch.nn as nn


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from qlora_llm.model_lora import Transformer, LoraModelArgs
from qlora_llm.configs.finetune_lora import config as cfg


def get_clean_state_dict(model: nn.Module):
    """Clean up lora weights and return cleaned state dict."""
    model_dict = model.state_dict()
    key_to_delete = [k for k in model_dict if 'lora_' in k]
    for del_key in key_to_delete:
        del model_dict[del_key]
    return model_dict


def convert_model_to_dtype(model: torch.nn.Module, dtype) -> None:
    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'token_embeddings' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype != dtype:
                    module = module.to(dtype)
        else:
            module = module.to(dtype)


def merge_lora_checkpoint(
    model_type: str, lora_ckpt_path: str, base_ckpt_dir: str, save_path: str, dtype=torch.bfloat16
) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        model_type: The llama-2 model type, supports 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        base_ckpt_dir: The base checkpoint (like pre-trained or fine-tuned) used for training with lora.
        save_path: target path to save the merged stat_dict.

    """

    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f'LoRA checkpoint file {lora_ckpt_path!r} does not exist, aborting...')
    if not os.path.exists(base_ckpt_dir):
        raise ValueError(f'Pretrained checkpoint dir {base_ckpt_dir!r} does not exist, aborting...')

    if os.path.exists(save_path):
        print(f'The checkpoint file {save_path!r} already exists, aborting...')
        return

    # try to get lora_params.json file based on the lora_ckpt_path
    lora_dir = os.path.dirname(lora_ckpt_path)

    # Create the path to the JSON file based on the directory
    lora_params_path = os.path.join(lora_dir, 'lora_params.json')
    if not os.path.exists(lora_params_path):
        print(f'Can not find LoRA params file {lora_params_path!r}, aborting...')
        return

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        # Create the output directory if necessary
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    print('Loading model checkpoints ...')

    # try to find and load pre-trained and lora checkpoints
    checkpoints = sorted(Path(base_ckpt_dir).glob('*.pth'))
    assert len(checkpoints) == 1, f'no checkpoint files found in {base_ckpt_dir!r}'
    pretrained_ckpt_file = checkpoints[0]

    pretrained_checkpoint = torch.load(pretrained_ckpt_file)
    lora_checkpoint = torch.load(lora_ckpt_path)

    with open(lora_params_path, 'r') as f:
        lora_params = json.load(f)

    model_args = LoraModelArgs.from_model_type(
        model_type=model_type,
        # LoRA configurations
        lora_r=lora_params['lora_r'],
        lora_scaling=lora_params['lora_scaling'],
        # LoRA trainable layers
        lora_attn_query=lora_params['lora_attn_query'],
        lora_attn_key=lora_params['lora_attn_key'],
        lora_attn_value=lora_params['lora_attn_value'],
        lora_attn_proj=lora_params['lora_attn_proj'],
        lora_attn_mlp=lora_params['lora_attn_mlp'],
        # No quantization during merge weights
        quant_4bit=False,
        quant_lora_4bit=False,
    )

    model = Transformer(model_args)

    # 1. Load the pretrained weights
    model.load_state_dict(pretrained_checkpoint, strict=False)

    # 2. Load the fine-tuned lora weights
    model.load_state_dict(lora_checkpoint, strict=False)

    # 3. Merge LoRA weights, this is handled inside the LoraLinear.train() method
    model.eval()

    # 4. optional, convert to bfloat16
    convert_model_to_dtype(model, dtype)

    # 5. Remove LoRA parameters from the model state
    state_dict = get_clean_state_dict(model)

    print(f'Saving merged model weights to {save_path!r} ...')
    torch.save(state_dict, save_path)

    print(f'Copying params.json to {output_dir!r}...')
    shutil.copy(os.path.join(base_ckpt_dir, 'params.json'), output_dir)


if __name__ == '__main__':
    # fine-tuned model
    merge_lora_checkpoint(
        model_type='7B',
        lora_ckpt_path='./checkpoints/finetune_lora-4bit/lora_7B-iter-400.pth',
        base_ckpt_dir='./meta_checkpoints/llama-2-7b/',
        save_path='./merged_checkpoints/7b-finetune/iter-400-merged.pth',
    )
