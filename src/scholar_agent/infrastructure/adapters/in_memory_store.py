"""Simple in-memory implementation of the memory-store port."""

from scholar_agent.application.output_ports.memory_store import IMemoryStore


class InMemoryStore(IMemoryStore):
    """Stores values only for the lifetime of the process."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return a stored value when available."""
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a value in process memory."""
        self._values[key] = value
