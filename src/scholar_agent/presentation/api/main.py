"""FastAPI application factory and server entry point."""

import uvicorn
from fastapi import FastAPI

from scholar_agent.config.settings import Settings
from scholar_agent.infrastructure.di.container import build_container
from scholar_agent.presentation.api.documents import router as documents_router
from scholar_agent.presentation.api.health import router as health_router
from scholar_agent.presentation.api.study import router as study_router
from scholar_agent.presentation.api.tutor import router as tutor_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the HTTP application and attach its dependency container."""
    application_settings = settings or Settings()
    app = FastAPI(
        title="ScholarAgent",
        version="0.1.0",
        debug=application_settings.debug,
    )
    app.state.container = build_container(application_settings)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(study_router)
    app.include_router(tutor_router)
    return app


app = create_app()


def run() -> None:
    """Run the API with Uvicorn's development defaults."""
    uvicorn.run(app, host="127.0.0.1", port=8000)
