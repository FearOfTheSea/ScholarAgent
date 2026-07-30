# scratch_gpt

Custom PyTorch GPT model implementation following *Build a Large Language
Model (from Scratch)* by Sebastian Raschka.

## Purpose

Provides an alternative `ILLMProvider` implementation that runs ScholarGPT, a
localized GPT-2 124M model whose architecture is built from scratch using custom
PyTorch blocks.

## Responsibilities

- `gpt_model.py`: Defines the typed model configuration, custom blocks,
  complete `GPTModel`, GPT-2 124M dimensions, GPT-2 weight transfer, and
  key-value cached inference.
- `checkpoint.py`: Saves self-describing versioned checkpoints while retaining
  compatibility with the original miniature checkpoint.
- `structured_generation.py`: Enforces the planner, quiz, flashcard, and
  document-map schemas using deterministic grounded decoding.
- `scratch_gpt_adapter.py`: Implements `ILLMProvider` using
  `data/scholar_gpt.pt`, extractive grounding for long retrieved passages, and
  the tuned model for assessment, verification, and free-form completions.

## Dependencies

- `torch`
- `transformers` (for GPT-2 tokenizer and pre-trained weight access)

Train and score the model with `scripts/train_scholar_gpt.py` and
`scripts/evaluate_scholar_gpt.py`. Both reuse the project `.venv`; the training
script works offline after GPT-2 has been cached.
