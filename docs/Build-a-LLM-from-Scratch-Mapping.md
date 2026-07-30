# Build-a-LLM-from-Scratch Concept Mapping

This document maps concepts from Sebastian Raschka's *Build a Large Language
Model (from Scratch)* to the educational custom-GPT implementation.

## Chapter Mapping

| Book chapter | Covered concepts | Implementation |
| --- | --- | --- |
| Chapter 2: Text Data | GPT-2 tokenization and sliding context windows | `scratch_gpt_adapter.py`, `scripts/train_mini_gpt.py` |
| Chapter 3: Attention | Scaled dot-product attention, causal masking, and multiple heads | `CausalSelfAttention` in `gpt_model.py` |
| Chapter 4: GPT Architecture | Layer normalization, GELU, feed-forward layers, residual paths, and the complete model | `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, and `GPTModel` |
| Chapter 5: Pre-training | Cross-entropy, AdamW, text generation, and GPT-2 weight transfer | `scripts/train_mini_gpt.py` and `load_gpt2_weights` |
| Chapter 6: Fine-Tuning | Instruction formatting, supervised updates, and checkpoint saving | `scripts/fine_tune_gpt2.py` |

## Key Implementations

### Causal Self-Attention

The upper-triangular mask is a registered model buffer, so it follows the model
between devices without becoming a trainable parameter:

```python
mask = torch.triu(
    torch.ones(config["context_length"], config["context_length"]),
    diagonal=1,
)
self.register_buffer("mask", mask)
```

The mask is applied before softmax, preventing each position from attending to
future tokens.

### Transformer Residual Paths

The transformer block keeps the pre-normalized attention and feed-forward
residuals explicit:

```python
attention_output = self.attn(self.ln1(x))
x = x + attention_output
feed_forward_output = self.ff(self.ln2(x))
return x + feed_forward_output
```

### Miniature and GPT-2 Configurations

Two deliberately distinct configurations prevent incompatible weights from
being mixed:

- `MINI_GPT_CONFIG` is a 128-dimensional, four-layer teaching model used by the
  runtime adapter and `data/mini_gpt.pt`.
- `GPT2_124M_CONFIG` has the 768-dimensional, twelve-layer GPT-2 architecture
  required by Hugging Face's `gpt2` weights and the fine-tuning demonstration.

### GPT-2 Weight Transfer

`load_gpt2_weights` copies embeddings, layer normalization, feed-forward,
attention, and output-head tensors. Hugging Face Conv1D weights are transposed
to match the custom model's `nn.Linear` layout:

```python
input_projection.weight.data.copy_(weights[f"{prefix}mlp.c_fc.weight"].t())
```

The mapping is explicit so learners can see how equivalent transformer
architectures use different parameter layouts.
