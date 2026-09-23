# STT API — Transcrição remota com Whisper

Servidor FastAPI para transcrição de áudio com **Whisper `large-v3`**
em GPU dedicada, **streaming** (SSE + WebSocket estilo Azure),
**diarização** opcional (pyannote 4) e **refinamento** via LLM
(Qwen3-8B no llama.cpp, em GPU dedicada).

## Quickstart

```bash
docker compose up --build -d        # produção (recomendado)
curl http://localhost:4321/health   # health (local: :8000)
python3 -m pytest tests/ -q
cd web && npm install && npm run dev   # front de teste: http://localhost:3000
```

Áudio em WAV 16kHz 16-bit mono. Referência interativa: `/docs` (Swagger).

## Docs

- [`docs/usage.md`](docs/usage.md) — subir, configurar, endpoints, troubleshooting
- [`docs/architecture.md`](docs/architecture.md) — camadas, fluxos, GPUs, decisões
- [`docs/code.md`](docs/code.md) — mapa de módulos, convenções, como estender
- [`docs/api.md`](docs/api.md) — Swagger (`/docs`, `/redoc`, `/openapi.json`)

## Desenvolvimento

```bash
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
python3 -m pytest tests/ -q
cd web && npm install && npm run dev   # front de teste: http://localhost:3000
```
