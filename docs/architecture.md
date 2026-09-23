# Arquitetura — STT API

> Para uso prático (curl, scripts, `.env`): ver `usage.md`.
> Para mapa de módulos e convenções: ver `code.md`.

## Visão geral

Servidor FastAPI de transcrição remota: **faster-whisper `large-v3`**
em GPU dedicada, **streaming** por chunks/SSE/WebSocket estilo Azure,
**diarização** opcional (pyannote 4) e **refinamento** via LLM
(llama.cpp + Qwen3-8B).

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Cliente     │────▶│  stt-api         │────▶│ llama-refine│
│  (Electron)  │◀────│  (FastAPI)       │◀────│  (llama.cpp)│
└─────────────┘     │  (GPU 1: Whisper  │     │ (GPU 0: só  │
                    │   + pyannote)     │     │  Qwen3-8B)  │
                    └──────────────────┘     └─────────────┘
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
3. **pyannote `speaker-diarization-3.1` sob `pyannote.audio` v4,
   lazy + pré-carga no lifespan.** Só ativa com `diarize=true` +
   `HF_TOKEN`. Falha = fallback para locutor genérico, nunca 500.
   v4 lê áudio via torchcodec (exige ffmpeg no container) e devolve
   `DiarizeOutput` (ler `.speaker_diarization`). Aceita
   `num/min/max_speakers` — `min_speakers` evita fundir vozes
   similares/minoritárias. Parciais SSE têm locutor best-effort
   (estável entre flushes via sobreposição); o `stop` re-roda do
   zero e é autoritativo.
4. **Refine via llama.cpp (OpenAI-compatible), Qwen3-8B Q4_K_M.**
   Batches de 50 segmentos por chamada, `max_tokens=4096`, thinking
   desligado (`enable_thinking: False`). Fails open: erro no LLM
   retorna o original.
5. **Wire camelCase estável.** `sessionId`, `tStart`/`tEnd`,
   `durationSec`. Mudança de wire = BREAKING CHANGE.
6. **GPUs separadas (2x RTX 4000 Ada).** GPU 0 só `llama-refine`,
   GPU 1 só `stt-api` (Whisper + pyannote) — sem briga por VRAM.
   Troca-se vazão concorrente do Whisper (era round-robin em 2 GPUs)
   por um refine maior e estável.
7. **Runtime CUDA 12 para o ctranslate2.** torch 2.9+ traz cu13, mas
   o engine do faster-whisper exige `libcublas.so.12` → deps diretas
   `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (coexistem com cu13).

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

`stt-api` em `CUDA_VISIBLE_DEVICES=1` com `WHISPER_GPUS=cuda:0`
(GPU física 1: Whisper + pyannote); `llama-refine` dedicado em
`CUDA_VISIBLE_DEVICES=0` com `--main-gpu 0 --parallel 8
--ctx-size 32768`. Sem CUDA, Whisper cai para CPU (bem mais lento).
