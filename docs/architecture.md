# Arquitetura — STT API

> Para uso prático (curl, scripts, `.env`): ver `usage.md`.
> Para mapa de módulos e convenções: ver `code.md`.

## Visão geral

Servidor FastAPI de transcrição remota: **faster-whisper `large-v3`**
em múltiplas GPUs, **streaming** por chunks/SSE/WebSocket estilo Azure,
**diarização** opcional (pyannote) e **refinamento** via LLM
(llama.cpp + Qwen 2.5 7B).

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Cliente     │────▶│  stt-api         │────▶│ llama-refine│
│  (Electron)  │◀────│  (FastAPI)       │◀────│  (llama.cpp)│
└─────────────┘     │                  │     └─────────────┘
                    │  ┌─ GPU 0 ─┐      │
                    │  │ Whisper  │      │
                    │  └─────────┘      │
                    │  ┌─ GPU 1 ─┐      │
                    │  │ Whisper  │      │
                    │  └─────────┘      │
                    └──────────────────┘
```

## Camadas (Clean Architecture)

Dependências apontam para dentro. O núcleo (`domain`, `application`)
não importa FastAPI, torch, pyannote ou OpenAI.

```
src/
  main.py              # composition root: app, lifespan, CORS, routers
  config.py            # Settings (pydantic-settings, lê .env)
  domain/              # regras puras, zero deps externas
    entities/transcript.py
    services/speaker_rules.py
    value_objects/audio_format.py, timestamp.py
  application/         # orquestração
    ports.py           # TranscriberPort, DiarizerPort, RefinerPort,
                       # SessionStore, AudioTempFiles
    use_cases/         # transcribe_audio, finalize_stream, refine_transcript
  infrastructure/      # adapters (detalhes trocáveis)
    dispatching/queue.py      # Dispatcher: fila + 1 worker por GPU
    asr/faster_whisper.py     # Transcriber por GPU (executor dedicado)
    diarization/pyannote.py   # pipeline lazy + PyannoteDiarizer
    refine/llama_refiner.py   # cliente OpenAI-compatible, fails open
    sessions/memory_store.py  # Session + SessionManager em memória
    audio/wav.py              # WAV canônico + WavTempFiles
  api/                 # transporte (fino)
    routes/            # health, transcribe, stream, refine
    ws/                # protocolo Azure + handler /speech/stream
    sse/events.py      # gerador de parciais
    schemas.py, mappers.py, deps.py
```

Shims legados (`src/dispatcher.py`, `src/session.py`, `src/schemas.py`,
etc.) re-exportam os novos caminhos — imports antigos continuam
funcionando, mas código novo deve usar os caminhos canônicos.

## Decisões (ADRs curtas)

1. **faster-whisper `large-v3` residente por GPU.** Carregar o modelo
   uma vez por GPU elimina latência de cold start. Cada `Transcriber`
   tem `ThreadPoolExecutor` dedicado de 1 thread: `load` + `transcribe`
   na mesma thread preservam o contexto CUDA (trocar de thread causa
   `CUDA failed with error invalid argument`).
2. **Dispatcher com fila única + 1 worker por GPU.** Round-robin via
   fila, não por índice explícito. GPUs ausentes são filtradas via
   `torch.cuda.device_count()`; sem CUDA, fallback para `["cpu"]`.
3. **pyannote `speaker-diarization-3.1` lazy e opcional.** Só carrega
   com `diarize=true` + `HF_TOKEN`. Falha = fallback para locutor
   genérico, nunca 500. Inclui monkey-patch documentado
   (`use_auth_token` → `token`) pela incompatibilidade
   pyannote 3.4.0 × `huggingface_hub>=0.20` — não remover.
4. **Refine via llama.cpp (OpenAI-compatible).** Batches de 50
   segmentos por chamada, `max_tokens=4096`. Fails open: erro no LLM
   retorna o original.
5. **Wire camelCase estável.** `sessionId`, `tStart`/`tEnd`,
   `durationSec`. Mudança de wire = BREAKING CHANGE.

## Fluxos

### Lote (`POST /transcribe`)

`audio bytes → tmp → dispatcher.dispatch → [diarize] →
build_transcript → wire`. Duração estimada por
`(file_size - 44) / 32000`, sem parsear header.

### Streaming (`POST /stream/{id}` + `/audio` + `/events`)

1. `start` cria `Session` (`mic`/`system`, `diarize`, `language`).
2. Chunks base64 PCM Int16LE 16kHz acumulam em buffers por canal.
3. SSE emite parciais a cada ~3s, só bytes novos desde o último
   flush (`<8000` bytes / ~0,5s é ignorado).
4. `stop` transcreve cada canal, aplica `apply_stop_rules`
   (`mic` → `Eu`; `system` sem diarize → `Sistema`), remove a sessão.

### WebSocket (`/speech/stream`, estilo Azure)

`config` → `turn.start` → `speech.hypothesis` (interim, sem locutor,
a cada `interimIntervalMs`) → `end` → transcrição final →
`speech.phrase` (uma por segmento, `Speaker` int só com diarização)
→ `turn.end` → fecha (1000). `Offset`/`Duration` em ticks de 100ns.

## GPUs

`stt-api` usa `CUDA_VISIBLE_DEVICES=0,1`; `llama-refine` fixa
`CUDA_VISIBLE_DEVICES=0` com `--main-gpu 0 --parallel 8
--ctx-size 32768`. Sem CUDA, Whisper cai para CPU (bem mais lento).
