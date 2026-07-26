"""CPU sentence-transformer embedding implementation."""

from collections.abc import Sequence
from typing import Protocol, cast

from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider


class _EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: str | list[str],
        normalize_embeddings: bool,
    ) -> object:
        """Return normalized embedding values."""


class SentenceTransformerEmbedding(IEmbeddingProvider):
    """Lazily loads the configured local Sentence Transformers model."""

    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model: _EmbeddingModel | None = None

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed one text value using a locally cached model."""
        return self.embed_many((text,))[0]

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embed multiple text values in one local model call."""
        if not texts:
            return ()
        model = self._get_model()
        encoded_values = cast(
            Sequence[Sequence[float]],
            model.encode(list(texts), normalize_embeddings=True),
        )
        return tuple(
            tuple(float(component) for component in embedding)
            for embedding in encoded_values
        )

    def _get_model(self) -> _EmbeddingModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = cast(
                _EmbeddingModel,
                SentenceTransformer(self._model_name, device=self._device),
            )
        return self._model
