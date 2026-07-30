"""LLM adapter for the educational custom PyTorch GPT model."""

import logging
import re
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.infrastructure.adapters.scratch_gpt.checkpoint import (
    load_checkpoint,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import (
    GPT2_124M_CONFIG,
    GPTConfig,
    GPTModel,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.structured_generation import (
    structured_response,
)

logger = logging.getLogger(__name__)

MINI_GPT_CONFIG: GPTConfig = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "bias": True,
}
# Backward-compatible import for earlier teaching material.
GPT_CONFIG_124M = GPT2_124M_CONFIG


class ScratchGPTAdapter(ILLMProvider):
    """Uses a custom PyTorch GPTModel loaded with pre-trained GPT-2 124M weights."""

    def __init__(
        self, context_length: int, maximum_tokens: int, checkpoint_path: Path
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive.")
        if maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive.")
        self._context_length = context_length
        self._maximum_tokens = maximum_tokens
        self._checkpoint_path = checkpoint_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: GPT2Tokenizer | None = None
        self._model: GPTModel | None = None
        self._model_context_length = MINI_GPT_CONFIG["context_length"]

    def _ensure_loaded(self) -> None:
        """Lazily load the trained miniature model and tokenizer."""
        if self._model is not None:
            return

        logger.info("Loading MiniGPT checkpoint from %s", self._checkpoint_path)
        try:
            if not self._checkpoint_path.exists():
                raise FileNotFoundError(
                    f"MiniGPT checkpoint not found: {self._checkpoint_path}. "
                    "Run scripts/train_mini_gpt.py first."
                )
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            checkpoint = load_checkpoint(
                self._checkpoint_path,
                legacy_config=MINI_GPT_CONFIG,
                device=self._device,
            )
            model = GPTModel(checkpoint.config)
            model.load_state_dict(checkpoint.state_dict)
            model.to(self._device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._model_context_length = checkpoint.config["context_length"]
            logger.info("Custom GPT-2 model loaded successfully.")
        except Exception as error:
            logger.exception("Failed to load custom GPT-2 model.")
            raise RuntimeError("The MiniGPT model is unavailable.") from error

    def generate(self, prompt: str) -> str:
        """Generate a response using the custom PyTorch GPT model."""
        structured = structured_response(prompt)
        if structured is not None:
            return structured
        grounded = _extract_grounded_response(prompt)
        if grounded is not None:
            return grounded
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        inference_prompt = f"{prompt.rstrip()}\n\nResponse:"
        input_ids = self._tokenizer.encode(inference_prompt)
        max_context = min(self._context_length, self._model_context_length)
        generation_limit = _generation_budget(max_context, self._maximum_tokens)
        maximum_prompt_tokens = max(1, max_context - generation_limit)
        input_ids = _fit_prompt_to_context(input_ids, maximum_prompt_tokens)
        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        prompt_len = len(input_ids)

        generated = input_tensor
        expects_json = "JSON" in prompt or "json" in prompt
        with torch.no_grad():
            logits, cache = self._model.forward_with_cache(input_tensor)
        for _ in range(generation_limit):
            # Predict next token greedily
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for End of Sequence
            if next_token.item() == self._tokenizer.eos_token_id:
                break

            generated = torch.cat((generated, next_token), dim=-1)
            if expects_json:
                completion = self._tokenizer.decode(generated[0, prompt_len:].tolist())
                if _contains_complete_json(completion):
                    break
            with torch.no_grad():
                logits, cache = self._model.forward_with_cache(next_token, cache)

        # Decode completion only (excluding the input prompt)
        completion_ids = generated[0, prompt_len:].tolist()
        completion_text = self._tokenizer.decode(completion_ids)
        return completion_text.strip()

    def is_available(self) -> bool:
        """Return True if custom model can be loaded/used."""
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    def has_model(self) -> bool:
        """Return whether the trained local checkpoint exists."""
        return self._checkpoint_path.is_file()


def _extract_grounded_response(prompt: str) -> str | None:
    """Return a small extractive answer for prompts containing retrieved text.

    A tiny language model cannot reliably follow a long RAG prompt. Selecting
    sentences from the supplied evidence is a deliberately simple safety net:
    it keeps MiniGPT in the application while ensuring output comes from the
    document being studied rather than from its training corpus.
    """
    if prompt.startswith("Write a concise study summary") and "Source text:" in prompt:
        source = prompt.split("Source text:", 1)[1]
        question = "summary"
    elif "LEARNER QUESTION:" in prompt and "SOURCES:" in prompt:
        question, source = prompt.split("SOURCES:", 1)
        question = question.split("LEARNER QUESTION:", 1)[1]
    elif "Question:" in prompt and "Sources:" in prompt:
        question, source = prompt.split("Sources:", 1)
        question = question.split("Question:", 1)[1]
    else:
        return None

    sentences = tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source)
        if len(sentence.strip()) >= 35
    )
    if not sentences:
        return None
    keywords = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{4,}", question)
        if word.lower() not in {"what", "which", "where", "when", "that", "from"}
    }
    scored = sorted(
        enumerate(sentences),
        key=lambda item: (
            sum(word in item[1].lower() for word in keywords),
            -item[0],
        ),
        reverse=True,
    )
    limit = 5 if question == "summary" else 3
    selected = sorted(scored[:limit], key=lambda item: item[0])
    return " ".join(sentence for _, sentence in selected)


def _fit_prompt_to_context(token_ids: list[int], limit: int) -> list[int]:
    """Preserve both instructions and recent evidence when a prompt is too long."""
    if len(token_ids) <= limit:
        return token_ids
    head_length = max(1, limit // 2)
    tail_length = limit - head_length
    if tail_length == 0:
        return token_ids[:head_length]
    return token_ids[:head_length] + token_ids[-tail_length:]


def _generation_budget(context_length: int, maximum_tokens: int) -> int:
    """Reserve at least half of the model context for the source prompt."""
    return min(maximum_tokens, max(1, context_length // 2))


def _contains_complete_json(value: str) -> bool:
    """Return whether text contains one balanced top-level JSON value."""
    starts = [index for index in (value.find("{"), value.find("[")) if index >= 0]
    if not starts:
        return False
    start = min(starts)
    opening = value[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for character in value[start:]:
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return True
    return False
