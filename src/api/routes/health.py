"""Rota de health check."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_dispatcher, get_sessions, get_settings
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Retorna GPUs configuradas, se o Whisper está carregado, "
    "endpoint de refine e sessões ativas.",
    responses={200: {"description": "Servidor respondendo."}},
    operation_id="getHealth",
)
async def health(
    dispatcher=Depends(get_dispatcher),
    sessions=Depends(get_sessions),
    settings=Depends(get_settings),
) -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="ok",
        gpus=settings.whisper_gpus_list,
        whisperLoaded=len(dispatcher._transcribers) > 0,
        refineEndpoint=settings.refine_base_url,
        activeSessions=sessions.active_count,
    )
