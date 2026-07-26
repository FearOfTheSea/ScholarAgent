"""Memory-store port."""

from abc import ABC, abstractmethod


class IMemoryStore(ABC):
    """Stores small string values by key."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the value for a key when it exists."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a value by key."""
