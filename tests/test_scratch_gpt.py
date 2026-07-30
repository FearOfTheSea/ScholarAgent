"""Unit tests for the custom scratch GPT model."""

from pathlib import Path

import pytest
import torch

from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import (
    GPT2_124M_CONFIG,
    GPTModel,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import (
    MINI_GPT_CONFIG,
    ScratchGPTAdapter,
)


def test_custom_gpt_model_forward_pass() -> None:
    """The custom model returns logits for every batch and token position."""
    # Create a scaled-down config for fast CPU testing
    test_config = MINI_GPT_CONFIG.copy()
    test_config["n_layers"] = 2
    test_config["n_heads"] = 2
    test_config["context_length"] = 64

    model = GPTModel(test_config)
    model.eval()

    # Batch of 2 sequences, each of length 8 tokens
    dummy_input = torch.randint(0, test_config["vocab_size"], (2, 8))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 8, test_config["vocab_size"])


def test_cached_generation_matches_full_forward_pass() -> None:
    """Key/value caching preserves next-token logits."""
    test_config = MINI_GPT_CONFIG.copy()
    test_config["n_layers"] = 2
    model = GPTModel(test_config)
    model.eval()
    tokens = torch.randint(0, test_config["vocab_size"], (1, 8))

    with torch.no_grad():
        full_logits = model(tokens)
        cached_parts = []
        cache = None
        for index in range(tokens.size(1)):
            logits, cache = model.forward_with_cache(
                tokens[:, index : index + 1],
                cache,
            )
            cached_parts.append(logits)

    cached_logits = torch.cat(cached_parts, dim=1)
    assert torch.allclose(full_logits, cached_logits, atol=2e-4, rtol=1e-5)


def test_mini_and_pretrained_configs_have_distinct_dimensions() -> None:
    """Mini checkpoints cannot be confused with pretrained GPT-2 weights."""
    assert MINI_GPT_CONFIG["emb_dim"] == 128
    assert MINI_GPT_CONFIG["context_length"] == 16
    assert GPT2_124M_CONFIG["emb_dim"] == 768
    assert GPT2_124M_CONFIG["context_length"] == 1024


def test_scratch_adapter_reports_checkpoint_presence(tmp_path: Path) -> None:
    """Readiness reflects the local checkpoint instead of an automatic download."""
    checkpoint = tmp_path / "mini_gpt.pt"
    adapter = ScratchGPTAdapter(16, 4, checkpoint)

    assert not adapter.has_model()

    checkpoint.touch()

    assert adapter.has_model()


@pytest.mark.parametrize(("context_length", "maximum_tokens"), [(0, 1), (1, 0)])
def test_scratch_adapter_rejects_invalid_generation_budgets(
    tmp_path: Path,
    context_length: int,
    maximum_tokens: int,
) -> None:
    """Invalid generation budgets fail before any model loading."""
    with pytest.raises(ValueError):
        ScratchGPTAdapter(
            context_length,
            maximum_tokens,
            tmp_path / "mini_gpt.pt",
        )
