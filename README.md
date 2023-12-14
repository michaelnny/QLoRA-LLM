# QLoRA-LLM

Uses QLoRA for fine-tuning a language model (LLM) with basic tools such as PyTorch and Bitsandbytes, without any Hugging Face tools.

# Motivation and Objective

We commonly use LoRA to fine-tune a large language model (LLM), and can further reduce GPU memory requirements by approximately 30% with QLoRA. Many projects recommend using the PEFT library from Hugging Face for QLoRA fine-tuning.

However, we aim to explore the possibility of QLoRA fine-tuning without relying on Hugging Face's tools. This has potential advantages:

- We might have a custom model not based on Hugging Face's transformers, making PEFT unsuitable.
- Implementing our own method provides greater control over the model and training pipeline. For example, we implement custom dropout layers in the model, similar to GPT-3, instead of using dropout from the standard LoRA layer.
- Developing our approach is an excellent opportunity to gain a deeper understanding of the process.

Given these considerations, we decided to implement QLoRA fine-tuning for LLM using basic tools like PyTorch and Bitsandbytes, decoupled from Hugging Face transformers and PEFT. We choose to use the LLaMA-2 2 7B model in this experiment, however this approach can be easily adapted for other LLM model.

# Disclaimer

**Project Purpose:** This project is dedicated to research and education, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** Rigorous testing has been conducted in specific scenarios, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

**Optimization:** For simplicity, we only focus on QLoRA and neglect other technics. For example, we can also extend the project by using gradient checkpointing to further reduce GPU RAM usage.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.1.1
- Tensorboard 2.13.0
- Bitsandbytes 0.41.3

# Code Structure

- `qlora_llm` directory contains main source code for the project.
  - `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
  - `utils` directory contains helper modules like custom datasets, logging, tokenization, LoRA module etc.
  - `lora.py` contains the LoRA layers.
  - `model.py` contains the LLaMA-2 model class.
  - `model_lora.py` contains the LLaMA-2 model class with linear layers been replaced by either LoRALinear or LoRALinear4bit.
  - `finetune_lora.py` run supervised fine-tuning starting from Meta's pre-trained model, using LoRA parameter efficient fine-tuning method (only supports single GPU).
  - `chat_completion.py` for chat completion, code adapted from the original LLaMA2 project.
- `scripts` directory contains all source code for convert the model weights and build datasets for different phases.
  - `build_finetune_datasets.py` build fine-tuning datasets (save the dataset to .jsonl files).
  - `convert_meta_checkpoint.py` convert Meta pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `convert_lora_checkpoint.py` convert fine-tunned LoRA weights to a full state_dict checkpoint.
- `datasets` directory contains the processed, tokenized datasets, which are ready to use.
- `logs` directory contains training logs for the different phases.

# QLoRA Fine-Tuning

## Overview

To fine-tune using QLoRA, we generally follow these simplified steps:

1. Replace frozen linear layers with quantized linear layers (Linear4bit from Bitsandbytes).
2. Replace trainable LoRA linear layers with quantized LoRA linear layers (custom LoRALinear4bit, extending Linear4bit from Bitsandbytes).
3. Load pre-trained model weights.
4. Set only LoRA parameters as trainable.
5. Train the model, saving checkpoints for LoRA parameters only.
6. After training, merge LoRA weights into pre-trained weights.

For simplicity, when working with a single custom model, it's easier to build the model with replaced linear layers upfront.

## QLoRA Options

The training settings are in `qlora_llm\configs\finetune_lora.py`. This file lets us choose which layers to train and the quantization methods.

### LoRA parameters

We use a slightly modified LoRALayer class, where we set the scaling directly instead of using an alpha parameter, we found this more consistent and easy to maintain. Since in most case, using a scaling of 1 makes more sense.

```
lora_r: int = 32
lora_scaling: float = 1.0  # we don't use alpha here, instead directly set the scaling
lora_dropout: float = 0.05
```

### Trainable layers

For example, we can specify which layers in the model should be trainable using options like the ones below.

```
lora_attn_query: bool = True  # train Attention query layer
lora_attn_key: bool = False  # train Attention key layer
lora_attn_value: bool = True  # train Attention value layer
lora_attn_proj: bool = False  # train Attention projection layer
lora_attn_mlp: bool = False  # train Attention MLP block
```

One thing to mention is that we don't apply LoRA or quantization to the lm_head layer. But we're not sure if this helps improve the performance or not.

### Quantization layers

We have various quantization options. For instance, we can quantize only the frozen linear layers or both the frozen linear layers and trainable LoRA layers.

When quantizing a LoRA layer, only the pre-trained weights are quantized, while the LoRA parameters remain unchanged.

It's important to mention that our current support is limited to 4-bit quantization, and we utilize Bitsandbytes.

```
quant_4bit: bool = True  # quantize frozen linear layer
quant_lora_4bit: bool = True  # quantize LoRA linear layer
quant_4bit_double: bool = True  # double quantize
quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'
```

### How does it work

**bitsandbytes.nn.Linear4bit**: This is an extended class over `torch.nn.Linear`, where it replaces the weights with `bitsandbytes.nn.Params4bit`

```
class Linear4bit(nn.Linear):

  def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', device=None):
      super().__init__(input_features, output_features, bias, device)
      self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type)

  ...

```

The weights quantization really happens when we move the model to cuda, as we can see from the `bitsandbytes.nn.Params4bit.cuda()` method. Which will use `bitsandbytes.functional.quantize_4bit()` to quantize the original pre-trained weights.

In theory, if we have the weights and quant_state, we can use `bitsandbytes.functional.dequantize_4bit()` to reverse or undo the quantization to get the original weights.

```
class Params4bit(torch.nn.Parameter):

  ...

  def cuda(self, device):
    if self.quant_state is not None:
        return self
    w = self.data.contiguous().half().cuda(device)
    w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
    self.data = w_4bit
    self.quant_state = quant_state

    return self
```

## Preparation

To begin, download the pre-trained weights from Meta by following the instructions at https://github.com/facebookresearch/llama.

After obtaining the weights, use the provided script to convert them to our own format. This step is crucial because we've adjusted the model naming convention. Make sure to update the file path in the script accordingly.

```
python3 scripts/convert_meta_checkpoint.py
```

## Start training

Once preparations are complete, use the command to execute the training script.

```
python3 qlora_llm/finetune_lora.py
```

Additionally, we can monitor the progress using TensorBoard.

```
tensorboard --logdir=./logs
```

![QLoRA LLaMA 7b Tensorboard](/images/qlora_llama_7b.png)

## Merge LoRA weights

After completing the training, combine the LoRA weights with the pre-trained weights to obtain the fully fine-tuned weights.

Which can be summarized into the following steps:

1. Construct a model with LoRA layers, matching the configuration used in fine-tuning but without quantized layers.
2. Load the pre-trained weights.
3. Load the LoRA weights
4. Set the model to evaluation mode (model.eval()) to merge the weights. This triggers the LoRALinear.train() method, and making the merging process.
5. Remove any LoRA parameters from the state dict
6. Save the merged checkpoint

You can use the following script to do the conversion, remember to update the file path in the script accordingly.

```
python3 scripts/convert_lora_checkpoint.py
```

After the merge, we can start run the chat_completion.py script to test the fine-tuned model. Make sure always adapt the checkpoint file path before running the script.

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license.

# Acknowledgments

This project is greatly influenced by the following projects:

- [Llama 2] (https://github.com/facebookresearch/llama)
- [Lit-LLaMA-2] (https://github.com/Lightning-AI/lit-llama)
- [LoRA] (https://github.com/microsoft/LoRA)
- [InstructLLaMA] (https://github.com/michaelnny/InstructLLaMA)
