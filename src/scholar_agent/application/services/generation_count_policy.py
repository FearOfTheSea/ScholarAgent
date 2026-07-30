"""Application policy for bounded quiz and flashcard generation."""

from dataclasses import dataclass

from scholar_agent.application.validators.common import require_positive


@dataclass(frozen=True, slots=True)
class GenerationCount:
    """A requested count normalized to an internal generation limit."""

    requested: int
    effective: int
    maximum: int

    @property
    def was_limited(self) -> bool:
        """Return whether the requested count exceeded the internal limit."""
        return self.requested > self.effective


class GenerationCountPolicy:
    """Owns the internal defaults and limits for generated study material."""

    QUIZ_DEFAULT = 5
    QUIZ_MAXIMUM = 10
    FLASHCARDS_DEFAULT = 10
    FLASHCARDS_MAXIMUM = 20

    def quiz(self, requested: int) -> GenerationCount:
        """Normalize a quiz count."""
        return self._normalize(requested, self.QUIZ_MAXIMUM, "question_count")

    def flashcards(self, requested: int) -> GenerationCount:
        """Normalize a flashcard count."""
        return self._normalize(requested, self.FLASHCARDS_MAXIMUM, "card_count")

    @staticmethod
    def _normalize(requested: int, maximum: int, field_name: str) -> GenerationCount:
        validated = require_positive(requested, field_name)
        return GenerationCount(
            requested=validated,
            effective=min(validated, maximum),
            maximum=maximum,
        )


def generation_limit_notice(
    item_name: str,
    count: GenerationCount,
) -> str | None:
    """Build the learner-facing notice for a capped request."""
    if not count.was_limited:
        return None
    return (
        f"You requested {count.requested} {item_name}; the current limit is "
        f"{count.maximum}, so {count.effective} were generated."
    )
