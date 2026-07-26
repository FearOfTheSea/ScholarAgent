"""Health endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import HealthResponse, ReadinessResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health() -> HealthResponse:
    """Return the service health without invoking application capabilities."""
    return HealthResponse(status="ok", service="scholar-agent", version="0.1.0")


@router.get("/ready", response_model=ReadinessResponse)
def get_readiness(
    container: Annotated[Container, Depends(get_container)],
) -> ReadinessResponse:
    """Check local Ollama and the configured model without an inference call."""
    result = container.check_runtime_readiness_use_case().execute()
    is_ready = result.ollama_available and result.model_available
    return ReadinessResponse(
        status="ready" if is_ready else "unavailable",
        ollama_available=result.ollama_available,
        model_available=result.model_available,
    )
