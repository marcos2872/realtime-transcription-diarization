"""Rota de refinamento via LLM (``POST /refine``)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_refiner, get_settings
from src.api.mappers import to_domain, to_wire
from src.api.schemas import RefineRequest, RefinedResult
from src.application.use_cases.refine_transcript import refine_transcript

router = APIRouter(tags=["refine"])


@router.post(
    "/refine",
    response_model=RefinedResult,
    summary="Refinar transcrição com LLM",
    description="Corrige ortografia, pontuação e gramática preservando "
    "conteúdo e timestamps. Falha no LLM retorna o original (fails open).",
    responses={
        200: {"description": "Refinamento concluído (ou original em fallback)."},
        422: {"description": "Body inválido."},
    },
    operation_id="refineTranscript",
)
async def refine(
    req: RefineRequest,
    refiner=Depends(get_refiner),
    settings=Depends(get_settings),
) -> RefinedResult:
    """Refina uma transcrição usando LLM (OpenAI-compatible)."""
    domain = to_domain(req.transcription)
    refined, model = await refine_transcript(
        transcript=domain,
        model=req.model,
        prompt=req.prompt,
        default_model=settings.refine_model,
        refiner=refiner,
    )
    return RefinedResult(refined=to_wire(refined), model=model, tokensUsed=0)
