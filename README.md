# transcript-service

Serviço de transcrição realtime com diarização de falantes — **Nemotron 3.5 ASR (streaming)** + **pyannote community-1** — para RTX 4000 20GB, com suporte a **4 streams em paralelo**.

> Documentação completa (contrato de API, arquitetura e uso): veja [`docs/api-contract.md`](docs/api-contract.md).

## Rápido (desenvolvimento, sem GPU)

```bash
uv sync
uv run pytest                 # 77 testes unitários + integração (providers fake)
TRANSCRIPT_ASR_PROVIDER=fake TRANSCRIPT_DIARIZATION_PROVIDER=fake uv run transcript-server
```

## Produção (Docker + GPU)

```bash
cp .env.example .env   # e preencha TRANSCRIPT_HF_TOKEN=hf_...
docker compose up --build
# API: http://localhost:8000 · front de teste: http://localhost:3000
```

- Imagem multi-stage sobre `nvidia/cuda` com `uv sync --frozen` (build reproduzível via `uv.lock`).
- Pesos dos modelos (~3 GB) persistem no volume `hf-cache` (`HF_HOME=/cache/huggingface`).
- `HEALTHCHECK` em `GET /health`; `restart: unless-stopped`.
- Sem Docker na máquina de dev? Instale os extras e rode nativo:
  `uv sync --extra asr --extra diarization && uv run transcript-server`
  (requer CUDA + ~5 GB de VRAM para os 4 streams).

## Camadas (Clean Architecture)

```
src/transcript/
├── domain/        # regras de negócio puras — sem NeMo, pyannote, FastAPI
│   ├── value_objects/   Timestamp, Speaker, SpeakerTurn, Word
│   ├── entities/        AudioStream (aggregate), Utterance
│   ├── services/        assign_speaker
│   └── ports/           Transcriber, Diarizer (interfaces)
├── application/   # casos de uso: OpenStream, ProcessChunk, CloseStream
├── adapters/      # detalhes concretos: nemotron, pyannote, fake, WebSocket/FastAPI, settings
├── di.py          # composition root (único ponto que conhece adapters)
└── main.py        # FastAPI app + lifespan
```

Setas da dependência apontam para dentro: `adapters → application → domain`.

## Testes

- `tests/unit/domain` — regras puras (sem I/O)
- `tests/unit/application` — casos de uso com ports fake
- `tests/unit/adapters` — helpers puros dos adapters
- `tests/integration` — WebSocket + REST de ponta a ponta com providers fake (sem GPU)
- `tests/integration/adapters` — smoke tests reais, marcados `@pytest.mark.gpu`

## Front de teste (`web/`)

Next.js + Tailwind (pt-BR) que sobe junto no compose. Todo o áudio é
processado no browser e enviado como PCM16 mono 16 kHz pelo WebSocket:

- **Ao vivo**: microfone → AudioWorklet → partial ao vivo + frases com falante
- **Arquivo .mp3**: decode local → chunks de ~1 s → `end` → frases finais

```bash
cd web && npm install && npm run dev   # http://localhost:3000 (API em :8000)
```
