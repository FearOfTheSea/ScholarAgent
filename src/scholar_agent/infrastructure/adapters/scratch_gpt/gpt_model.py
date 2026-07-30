"""A small, educational GPT implementation built from explicit PyTorch blocks."""

import math
from collections.abc import Mapping
from typing import Protocol, TypedDict, cast

import torch
import torch.nn as nn

type AttentionCache = tuple[torch.Tensor, torch.Tensor]
type ModelCache = tuple[AttentionCache, ...]


class GPTConfig(TypedDict):
    """Strongly typed dimensions shared by every custom GPT block."""

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    bias: bool


GPT2_124M_CONFIG: GPTConfig = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "bias": True,
}


class HuggingFaceGPT2(Protocol):
    """Minimum pretrained-model surface required for weight transfer."""

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return named GPT-2 tensors."""


class LayerNorm(nn.Module):
    """Custom layer normalization."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last tensor dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.shift


class GELU(nn.Module):
    """Custom GELU activation approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tanh GELU approximation used by GPT."""
        result = (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
        return result


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand, activate, and project token representations."""
        return cast(torch.Tensor, self.layers(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    mask: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config["emb_dim"] % config["n_heads"] != 0:
            raise ValueError("emb_dim must be divisible by n_heads.")
        self.num_heads = config["n_heads"]
        self.emb_dim = config["emb_dim"]
        self.head_dim = config["emb_dim"] // config["n_heads"]

        self.W_query = nn.Linear(
            config["emb_dim"], config["emb_dim"], bias=config["bias"]
        )
        self.W_key = nn.Linear(
            config["emb_dim"], config["emb_dim"], bias=config["bias"]
        )
        self.W_value = nn.Linear(
            config["emb_dim"], config["emb_dim"], bias=config["bias"]
        )
        self.out_proj = nn.Linear(
            config["emb_dim"], config["emb_dim"], bias=config["bias"]
        )

        mask = torch.triu(
            torch.ones(config["context_length"], config["context_length"]),
            diagonal=1,
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot-product attention without future-token access."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past: AttentionCache | None = None,
    ) -> tuple[torch.Tensor, AttentionCache]:
        """Apply attention and return reusable key/value tensors."""
        batch_size, token_count, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        queries = queries.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        past_length = 0
        if past is not None:
            past_keys, past_values = past
            past_length = past_keys.size(2)
            keys = torch.cat((past_keys, keys), dim=2)
            values = torch.cat((past_values, values), dim=2)

        attention_scores = queries @ keys.transpose(-2, -1)
        total_length = past_length + token_count
        causal_mask = self.mask[
            past_length : past_length + token_count,
            :total_length,
        ].bool()
        attention_scores.masked_fill_(causal_mask, -float("inf"))
        attention_weights = torch.softmax(
            attention_scores / math.sqrt(self.head_dim),
            dim=-1,
        )

        context = (attention_weights @ values).transpose(1, 2).contiguous()
        context = context.view(batch_size, token_count, self.emb_dim)
        output = cast(torch.Tensor, self.out_proj(context))
        return output, (keys, values)


class TransformerBlock(nn.Module):
    """Pre-normalized transformer block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config["emb_dim"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config["emb_dim"])
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual paths."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past: AttentionCache | None = None,
    ) -> tuple[torch.Tensor, AttentionCache]:
        """Apply the block and return its reusable attention state."""
        attention_output, cache = self.attn.forward_with_cache(self.ln1(x), past)
        x = x + attention_output
        feed_forward_output = self.ff(self.ln2(x))
        return cast(torch.Tensor, x + feed_forward_output), cache


class GPTModel(nn.Module):
    """GPT language model composed from the educational blocks above."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.trf_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config["n_layers"])
        )
        self.final_ln = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
        )
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every input position."""
        logits, _ = self.forward_with_cache(in_idx)
        return logits

    def forward_with_cache(
        self,
        in_idx: torch.Tensor,
        past: ModelCache | None = None,
    ) -> tuple[torch.Tensor, ModelCache]:
        """Return logits and key/value state for efficient generation."""
        _, sequence_length = in_idx.shape
        past_length = past[0][0].size(2) if past else 0
        token_embeddings = self.tok_emb(in_idx)
        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=in_idx.device,
        )
        position_embeddings = self.pos_emb(positions)
        x = token_embeddings + position_embeddings
        caches: list[AttentionCache] = []
        for index, module in enumerate(self.trf_blocks):
            block = cast(TransformerBlock, module)
            block_past = past[index] if past is not None else None
            x, cache = block.forward_with_cache(x, block_past)
            caches.append(cache)
        x = self.final_ln(x)
        logits = cast(torch.Tensor, self.out_head(x))
        return logits, tuple(caches)


def load_gpt2_weights(gpt: GPTModel, pretrained: HuggingFaceGPT2) -> None:
    """Copy Hugging Face GPT-2 weights into the custom model."""
    weights = pretrained.state_dict()

    gpt.tok_emb.weight.data.copy_(weights["transformer.wte.weight"])
    gpt.pos_emb.weight.data.copy_(weights["transformer.wpe.weight"])

    for index, module in enumerate(gpt.trf_blocks):
        block = cast(TransformerBlock, module)
        prefix = f"transformer.h.{index}."

        block.ln1.scale.data.copy_(weights[f"{prefix}ln_1.weight"])
        block.ln1.shift.data.copy_(weights[f"{prefix}ln_1.bias"])
        block.ln2.scale.data.copy_(weights[f"{prefix}ln_2.weight"])
        block.ln2.shift.data.copy_(weights[f"{prefix}ln_2.bias"])

        input_projection = cast(nn.Linear, block.ff.layers[0])
        output_projection = cast(nn.Linear, block.ff.layers[2])
        input_projection.weight.data.copy_(weights[f"{prefix}mlp.c_fc.weight"].t())
        input_projection.bias.data.copy_(weights[f"{prefix}mlp.c_fc.bias"])
        output_projection.weight.data.copy_(weights[f"{prefix}mlp.c_proj.weight"].t())
        output_projection.bias.data.copy_(weights[f"{prefix}mlp.c_proj.bias"])

        query_key_value_weights = weights[f"{prefix}attn.c_attn.weight"].t()
        query_key_value_bias = weights[f"{prefix}attn.c_attn.bias"]
        embedding_dimension = gpt.pos_emb.weight.shape[1]
        query_weight, key_weight, value_weight = torch.split(
            query_key_value_weights,
            embedding_dimension,
            dim=0,
        )
        query_bias, key_bias, value_bias = torch.split(
            query_key_value_bias,
            embedding_dimension,
            dim=0,
        )

        _copy_linear_projection(
            block.attn.W_query,
            query_weight,
            query_bias,
        )
        _copy_linear_projection(block.attn.W_key, key_weight, key_bias)
        _copy_linear_projection(
            block.attn.W_value,
            value_weight,
            value_bias,
        )
        _copy_linear_projection(
            block.attn.out_proj,
            weights[f"{prefix}attn.c_proj.weight"].t(),
            weights[f"{prefix}attn.c_proj.bias"],
        )

    gpt.final_ln.scale.data.copy_(weights["transformer.ln_f.weight"])
    gpt.final_ln.shift.data.copy_(weights["transformer.ln_f.bias"])
    gpt.out_head.weight.data.copy_(weights["lm_head.weight"])


def _copy_linear_projection(
    projection: nn.Linear,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    """Copy tensors into a projection that is expected to include bias."""
    if projection.bias is None:
        raise ValueError("GPT-2 weight transfer requires biased projections.")
    projection.weight.data.copy_(weight)
    projection.bias.data.copy_(bias)
