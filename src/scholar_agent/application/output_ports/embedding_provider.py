"""Embedding-provider port."""

from abc import ABC, abstractmethod


class IEmbeddingProvider(ABC):
    """Converts text into an embedding vector."""

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]:
        """Return an embedding for text."""

    @abstractmethod
    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Return embeddings for multiple texts."""
