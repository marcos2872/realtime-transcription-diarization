"""Composition root — monta o app FastAPI (Swagger em ``/docs``).

Toda lógica vive em ``api/`` (rotas finas), ``application/``
(use cases) e ``infrastructure/`` (adapters). Aqui só: settings,
lifespan (dispatcher start/stop), CORS, routers e ``app.state``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, refine, stream, transcribe
from src.api.ws import speech
from src.config import settings
from src.infrastructure.dispatching.queue import Dispatcher
from src.infrastructure.refine.llama_refiner import Refiner
from src.infrastructure.sessions.memory_store import SessionManager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "health", "description": "Status do servidor e GPUs."},
    {"name": "transcribe", "description": "Transcrição em lote de arquivos WAV."},
    {"name": "streaming", "description": "Sessões de streaming (chunks, SSE)."},
    {"name": "refine", "description": "Refinamento de transcrição via LLM."},
    {"name": "websocket", "description": "Streaming estilo Azure STT (fora do OpenAPI — ver `docs/usage.md`)."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa dispatcher (Whisper nas GPUs) e guarda deps em ``app.state``."""
    logger.info("Iniciando servidor STT API...")
    app.state.settings = settings
    app.state.dispatcher = Dispatcher()
    app.state.sessions = SessionManager()
    app.state.refiner = Refiner()
    await app.state.dispatcher.start()
    logger.info(
        "Servidor pronto: %d GPU(s) | %s",
        len(app.state.dispatcher._transcribers),
        settings.refine_base_url,
    )
    yield
    logger.info("Parando servidor...")
    await app.state.dispatcher.stop()


app = FastAPI(
    title="STT API",
    summary="Transcrição remota com Whisper large-v3 multi-GPU",
    description=(
        "Servidor de transcrição: **Whisper large-v3** em múltiplas GPUs "
        "(round-robin), **streaming** por chunks/SSE/WebSocket estilo Azure, "
        "**diarização** opcional (pyannote) e **refinamento** via LLM "
        "(llama.cpp Qwen 2.5 7B).\n\n"
        "Áudio sempre em **WAV 16kHz 16-bit mono**. "
        "Diarização sem `HF_TOKEN` faz fallback silencioso (nunca 500). "
        "Detalhes e exemplos: `docs/usage.md`."
    ),
    version="0.1.0",
    contact={"name": "STT API"},
    license_info={"name": "MIT"},
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS liberado para o front de teste em web/ (browser → API direto).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(stream.router)
app.include_router(refine.router)
app.include_router(speech.router)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
