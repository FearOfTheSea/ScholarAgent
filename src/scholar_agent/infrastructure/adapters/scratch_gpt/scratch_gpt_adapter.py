"""Alternative LLM provider adapter using a custom PyTorch GPT model built from scratch."""

import logging
from pathlib import Path
import re

import torch
from transformers import GPT2Tokenizer

from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)

MINI_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "bias": True,
}
# Backward-compatible name used by the model-focused tests and teaching notes.
GPT_CONFIG_124M = MINI_GPT_CONFIG


class ScratchGPTAdapter(ILLMProvider):
    """Uses a custom PyTorch GPTModel loaded with pre-trained GPT-2 124M weights."""

    def __init__(
        self, context_length: int, maximum_tokens: int, checkpoint_path: Path
    ) -> None:
        self._context_length = context_length
        self._maximum_tokens = maximum_tokens
        self._checkpoint_path = checkpoint_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer: GPT2Tokenizer | None = None
        self._model: GPTModel | None = None

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
            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._model = GPTModel(MINI_GPT_CONFIG)
            state_dict = torch.load(
                self._checkpoint_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            
            self._model.to(self._device)
            self._model.eval()
            logger.info("Custom GPT-2 model loaded successfully.")
        except Exception as error:
            logger.exception("Failed to load custom GPT-2 model.")
            raise RuntimeError("The MiniGPT model is unavailable.") from error

    def generate(self, prompt: str) -> str:
        """Generate a response using the custom PyTorch GPT model."""
        grounded = _extract_grounded_response(prompt)
        if grounded is not None:
            return grounded
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        input_ids = self._tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        prompt_len = len(input_ids)

        # Truncate context if it exceeds model capability (1024 tokens)
        max_context = min(self._context_length, MINI_GPT_CONFIG["context_length"])

        generated = input_tensor
        for _ in range(self._maximum_tokens):
            # Slice context to fit context window
            idx_cond = generated[:, -max_context:]
            with torch.no_grad():
                logits = self._model(idx_cond)
            
            # Predict next token greedily
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for End of Sequence
            if next_token.item() == self._tokenizer.eos_token_id:
                break
                
            generated = torch.cat((generated, next_token), dim=-1)

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
        """Always return True since model downloads automatically if needed."""
        return True


def _extract_grounded_response(prompt: str) -> str | None:
    """Return a small extractive answer for prompts containing retrieved text.

    A tiny language model cannot reliably follow a long RAG prompt. Selecting
    sentences from the supplied evidence is a deliberately simple safety net:
    it keeps MiniGPT in the application while ensuring output comes from the
    document being studied rather than from its training corpus.
    """
    if "Source text:" in prompt:
        source = prompt.split("Source text:", 1)[1]
        question = "summary"
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
