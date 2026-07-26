"""Custom PyTorch GPT model implementation based on the book 'Build a Large Language Model (from Scratch)'."""

import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Custom Layer Normalization module."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Custom GELU activation function approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CausalSelfAttention(nn.Module):
    """Multi-head Causal Self-Attention mechanism."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.num_heads = cfg["n_heads"]
        self.emb_dim = cfg["emb_dim"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        
        self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"])
        self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"])
        self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"])
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Multi-head split
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product causal attention
        attn_scores = queries @ keys.transpose(-2, -1)
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, -float("inf"))

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.emb_dim)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    """A standard Transformer encoder block with residual connections."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """The full GPT model built from custom blocks."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_ln = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        b, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.out_head(x)
        return logits


def load_gpt2_weights(gpt: GPTModel, gpt_hf) -> None:
    """Copies weights from Hugging Face's pretrained GPT-2 into our custom model structure."""
    d = gpt_hf.state_dict()
    
    gpt.tok_emb.weight.data.copy_(d["transformer.wte.weight"])
    gpt.pos_emb.weight.data.copy_(d["transformer.wpe.weight"])
    
    for i in range(len(gpt.trf_blocks)):
        block = gpt.trf_blocks[i]
        hf_prefix = f"transformer.h.{i}."
        
        block.ln1.scale.data.copy_(d[f"{hf_prefix}ln_1.weight"])
        block.ln1.shift.data.copy_(d[f"{hf_prefix}ln_1.bias"])
        block.ln2.scale.data.copy_(d[f"{hf_prefix}ln_2.weight"])
        block.ln2.shift.data.copy_(d[f"{hf_prefix}ln_2.bias"])
        
        # MLPs require transposition of weights due to Conv1D/Linear layers mismatch
        block.ff.layers[0].weight.data.copy_(d[f"{hf_prefix}mlp.c_fc.weight"].t())
        block.ff.layers[0].bias.data.copy_(d[f"{hf_prefix}mlp.c_fc.bias"])
        block.ff.layers[2].weight.data.copy_(d[f"{hf_prefix}mlp.c_proj.weight"].t())
        block.ff.layers[2].bias.data.copy_(d[f"{hf_prefix}mlp.c_proj.bias"])
        
        # Attention projection merging split
        qkv_weights = d[f"{hf_prefix}attn.c_attn.weight"].t()
        qkv_bias = d[f"{hf_prefix}attn.c_attn.bias"]
        
        emb_dim = gpt.pos_emb.weight.shape[1]
        W_q, W_k, W_v = torch.split(qkv_weights, emb_dim, dim=0)
        b_q, b_k, b_v = torch.split(qkv_bias, emb_dim, dim=0)
        
        block.attn.W_query.weight.data.copy_(W_q)
        block.attn.W_query.bias.data.copy_(b_q)
        block.attn.W_key.weight.data.copy_(W_k)
        block.attn.W_key.bias.data.copy_(b_k)
        block.attn.W_value.weight.data.copy_(W_v)
        block.attn.W_value.bias.data.copy_(b_v)
        
        block.attn.out_proj.weight.data.copy_(d[f"{hf_prefix}attn.c_proj.weight"].t())
        block.attn.out_proj.bias.data.copy_(d[f"{hf_prefix}attn.c_proj.bias"])
        
    gpt.final_ln.scale.data.copy_(d["transformer.ln_f.weight"])
    gpt.final_ln.shift.data.copy_(d["transformer.ln_f.bias"])
    gpt.out_head.weight.data.copy_(d["lm_head.weight"])
