"""Framework-free application validators."""

from scholar_agent.application.validators.common import (
    require_non_blank,
    require_positive,
)

__all__ = ["require_non_blank", "require_positive"]
