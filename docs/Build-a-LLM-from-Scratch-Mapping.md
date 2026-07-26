# Build-a-LLM-from-Scratch Concept Mapping

This document provides a detailed mapping between the chapters of the book *Build a Large Language Model (from Scratch)* by Sebastian Raschka and the concrete implementations in this repository.

---

## 🗺️ Chapter Mapping Table

| Book Chapter | Covered Concepts | Target Code / Files |
| --- | --- | --- |
| **Chapter 2: Working with Text Data** | Tokenization, sliding context windows, vocabulary mappings, PyTorch datasets, and data loaders. | - [scratch_gpt_adapter.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/scratch_gpt_adapter.py) (uses the GPT-2 BPE tokenizer)<br>- [train_mini_gpt.py](file:///C:/Users/Admin/Desktop/ScholarAgent/scripts/train_mini_gpt.py) (implements custom sliding window sequence batching) |
| **Chapter 3: Attention Mechanisms** | Self-attention, scaled dot-product attention, causal masking, and Multi-Head Attention. | - `CausalSelfAttention` in [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py) |
| **Chapter 4: GPT Architecture** | Custom `LayerNorm`, custom `GELU` activation, feed-forward sub-layers, residual shortcut connections, and the full `GPTModel` class. | - `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, and `GPTModel` in [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py) |
| **Chapter 5: Pre-training & Weights** | Loss function (cross-entropy), training loop, text generation utility, and loading OpenAI GPT-2 weights. | - `load_gpt2_weights` in [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py)<br>- [train_mini_gpt.py](file:///C:/Users/Admin/Desktop/ScholarAgent/scripts/train_mini_gpt.py) (custom training loop, cross-entropy calculation, and AdamW optimizer implementation) |
| **Chapter 6: Instruction Fine-Tuning** | Prompt formatting templates, fine-tuning pre-trained models on question-answer instruction datasets. | - [fine_tune_gpt2.py](file:///C:/Users/Admin/Desktop/ScholarAgent/scripts/fine_tune_gpt2.py) (implements instruction formatting, custom loss gradient updates, and checkpoint saving) |

---

## 🛠️ Detailed Conceptual Implementations

### 1. Causal Self-Attention (Chapter 3)
In [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py#L48-L92), the causal masking is registered as a buffer and applied to dot-product attention scores:
```python
# Upper-triangular mask to prevent looking at future tokens
self.register_buffer(
    "mask",
    torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1)
)
...
# Masking step
mask = self.mask[:num_tokens, :num_tokens].bool()
attn_scores.masked_fill_(mask, -float("inf"))
```

### 2. Transformer Block & Residuals (Chapter 4)
In [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py#L95-L109), the shortcut paths around the attention and feed-forward blocks are explicitly implemented with LayerNorm pre-activation:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln1(x)) # Attention residual
    x = x + self.ff(self.ln2(x))   # FeedForward residual
    return x
```

### 3. Loading OpenAI Weights (Chapter 5)
In [gpt_model.py](file:///C:/Users/Admin/Desktop/ScholarAgent/src/scholar_agent/infrastructure/adapters/scratch_gpt/gpt_model.py#L137-L182), the weights from `gpt2` (124M parameters) are mapped parameter-by-parameter, including transposing the Conv1D projections of the Hugging Face structure to standard `nn.Linear` layers:
```python
# Transposition of Conv1D weights to match PyTorch Linear layout
block.ff.layers[0].weight.data.copy_(d[f"{hf_prefix}mlp.c_fc.weight"].t())
```

### 4. Custom Training & Fine-Tuning Loops (Chapters 5 & 6)
- The [train_mini_gpt.py](file:///C:/Users/Admin/Desktop/ScholarAgent/scripts/train_mini_gpt.py) script contains a complete PyTorch training loop demonstrating optimization, loss monitoring, and text generation.
- The [fine_tune_gpt2.py](file:///C:/Users/Admin/Desktop/ScholarAgent/scripts/fine_tune_gpt2.py) script demonstrates instruction fine-tuning on custom academic QA instructions.
