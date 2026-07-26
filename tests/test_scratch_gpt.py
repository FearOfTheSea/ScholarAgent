"""Unit tests for the custom scratch GPT model."""

import torch

from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import GPTModel
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import GPT_CONFIG_124M


def test_custom_gpt_model_forward_pass() -> None:
    """Verifies that the custom model executes a forward pass and returns correct logit shapes."""
    # Create a scaled-down config for fast CPU testing
    test_config = GPT_CONFIG_124M.copy()
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
