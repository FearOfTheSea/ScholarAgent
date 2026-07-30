"""Versioned checkpoint helpers for custom GPT models."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import GPTConfig

CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated checkpoint data required to construct a custom GPT."""

    config: GPTConfig
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, object]


def save_checkpoint(
    path: Path,
    config: GPTConfig,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> None:
    """Persist model tensors with their architecture and training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "config": dict(config),
            "state_dict": dict(state_dict),
            "metadata": dict(metadata),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    legacy_config: GPTConfig,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load a versioned checkpoint or a legacy state-dict-only checkpoint."""
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("Scratch-GPT checkpoint must contain a mapping.")
    if "state_dict" not in raw:
        return LoadedCheckpoint(
            config=legacy_config,
            state_dict=_state_dict(raw),
            metadata={"legacy": True},
        )
    version = raw.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported Scratch-GPT checkpoint version: {version}.")
    config = _config(raw.get("config"))
    metadata_value = raw.get("metadata", {})
    if not isinstance(metadata_value, dict):
        raise ValueError("Checkpoint metadata must be a mapping.")
    metadata = {str(key): value for key, value in metadata_value.items()}
    return LoadedCheckpoint(
        config=config,
        state_dict=_state_dict(raw.get("state_dict")),
        metadata=metadata,
    )


def _state_dict(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Checkpoint state_dict must be a non-empty mapping.")
    if not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise ValueError("Checkpoint state_dict contains invalid entries.")
    return cast(dict[str, torch.Tensor], value)


def _config(value: object) -> GPTConfig:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint config must be a mapping.")
    integer_fields = (
        "vocab_size",
        "context_length",
        "emb_dim",
        "n_heads",
        "n_layers",
    )
    parsed: dict[str, int | bool] = {}
    for field in integer_fields:
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"Checkpoint config field '{field}' must be positive.")
        parsed[field] = item
    bias = value.get("bias")
    if not isinstance(bias, bool):
        raise ValueError("Checkpoint config field 'bias' must be boolean.")
    parsed["bias"] = bias
    if int(parsed["emb_dim"]) % int(parsed["n_heads"]) != 0:
        raise ValueError("Checkpoint emb_dim must be divisible by n_heads.")
    return cast(GPTConfig, parsed)
