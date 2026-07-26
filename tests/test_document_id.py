"""Tests for domain value objects."""

import pytest

from scholar_agent.domain.exceptions import DomainValidationError
from scholar_agent.domain.value_objects import DocumentId


def test_document_id_rejects_blank_values() -> None:
    """A document identifier is a required domain value."""
    with pytest.raises(DomainValidationError):
        DocumentId("   ")
