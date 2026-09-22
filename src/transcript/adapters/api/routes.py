"""REST endpoints (identity + health). The streaming API lives on /ws/streams/..."""

from __future__ import annotations

from fastapi import APIRouter, Request


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", tags=["meta"])
    async def identity(request: Request) -> dict:
        info = request.app.state.application.info
        return {
            "name": info.name,
            "version": info.version,
            "language": info.language,
            "sample_rate": info.sample_rate,
            "max_streams": info.max_streams,
            "asr": {
                "provider": info.asr_provider,
                "model": info.asr_model,
                "chunk_ms": info.chunk_ms,
            },
            "diarization": {
                "provider": info.diarization_provider,
                "pipeline": info.diarization_pipeline,
            },
        }

    @router.get("/health", tags=["meta"])
    async def health(request: Request) -> dict:
        application = request.app.state.application
        return {
            "status": "ok",
            "active_streams": len(application.registry),
            "max_streams": application.settings.max_streams,
            "language": application.settings.language,
            "asr_provider": application.settings.asr_provider,
            "diarization_provider": application.settings.diarization_provider,
            "device": application.info.device,
        }

    return router
