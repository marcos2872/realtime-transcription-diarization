"""FastAPI application factory and CLI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query, WebSocket

from transcript import __version__
from transcript.adapters.api.routes import create_router
from transcript.adapters.api.websocket_server import handle_stream
from transcript.adapters.config.settings import load_settings
from transcript.di import Application, build_application

logger = logging.getLogger(__name__)


def create_app(application: Application | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured = application or build_application(load_settings())
        app.state.application = configured
        await configured.startup()
        logger.info(
            "transcript-service %s ready (asr=%s, diarization=%s, device=%s)",
            __version__,
            configured.settings.asr_provider,
            configured.settings.diarization_provider,
            configured.info.device,
        )
        try:
            yield
        finally:
            await configured.shutdown()

    app = FastAPI(title="transcript-service", version=__version__, lifespan=lifespan)
    app.include_router(create_router())

    @app.websocket("/ws/streams/{stream_id}")
    async def stream(
        websocket: WebSocket, stream_id: str, partials: bool | None = Query(default=None)
    ) -> None:
        await handle_stream(websocket, stream_id, partials)

    return app


app = create_app()


def run() -> None:
    """CLI entry point (``transcript-server``): load settings and serve."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
