"""FastAPI dependencies for the composition root."""

from typing import cast

from fastapi import Request

from scholar_agent.infrastructure.di.container import Container


def get_container(request: Request) -> Container:
    """Return the configured container attached by the application factory."""
    return cast(Container, request.app.state.container)
