# scratch_gpt

Custom PyTorch GPT model implementation following *Build a Large Language Model (from Scratch)* by Sebastian Raschka.

## Purpose

Provides an alternative `ILLMProvider` implementation that runs a trained miniature GPT model whose architecture is built from scratch using custom PyTorch blocks.

## Responsibilities

- `gpt_model.py`: Defines `LayerNorm`, `GELU`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `GPTModel`, and `load_gpt2_weights`.
- `scratch_gpt_adapter.py`: Implements `ILLMProvider` using the custom `GPTModel` loaded from `data/mini_gpt.pt`.

## Dependencies

- `torch`
- `transformers` (for GPT-2 tokenizer and pre-trained weight access)
