# AGENTS.md — stt-api

FastAPI remote transcription server: faster-whisper `large-v3` (GPU dedicada) + optional pyannote 4 diarization + LLM refine via llama.cpp (Qwen3-8B Q4_K_M).

## Commands (use `uv`, not `pip`/`python`)

```bash
uv sync                                        # install (Python >=3.11,<3.13; Docker uses 3.12-slim)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload   # local dev (needs CUDA GPU)
docker compose up --build -d                   # production (recommended)
docker compose logs -f                         # logs
curl http://localhost:4321/health              # health (compose maps 4321->8000; local is :8000)
python3 -m pytest tests/ -q                    # domain + use-case tests (no GPU needed)
./deploy.sh [user]                             # rsync to 192.168.3.81 (default user `administrador`)
```

`tests/` runs without GPU/network (fakes). E2E manual: `/health` + front `web/` (4 abas: batch, SSE, WebSocket, refine).

Docs: `README.md` (apresentação) + `docs/usage.md` (uso), `docs/architecture.md` (arquitetura), `docs/code.md` (código), `docs/api.md` (Swagger em `/docs`).

## Architecture (`src/`)

Clean Architecture — dependências apontam para dentro (`api → application → domain`; `infrastructure` implementa ports). Use cases nunca importam `infrastructure`.

- `main.py` — composition root `src.main:app` (enxuto, ~110 linhas): lifespan, CORS, routers, `app.state` (dispatcher/sessions/refiner). Lifespan pré-carrega o pyannote (`preload`) quando há `HF_TOKEN`. Swagger metadata aqui (`/docs`, `/redoc`, `/openapi.json`).
- `config.py` — `pydantic-settings`, reads `.env` (see `.env.example`). `WHISPER_GPUS` is comma string, parsed via `whisper_gpus_list` (no compose, `cuda:0` = GPU física 1). `REFINE_BASE_URL` differs: `localhost:8080` local vs `http://llama-refine:8080/v1` in compose. `REFINE_MODEL` default `Qwen3-8B`.
- `domain/` — regras puras, zero deps externas: `entities/transcript.py` (`SegmentData`, `TranscriptData`, `build_transcript`), `services/speaker_rules.py` (`Eu`/`Sistema`/`Pessoa N`/`Locutor`, `apply_stop_rules`, `overlap`/`match_raw_label`/`resolve_stable_labels`), `value_objects/audio_format.py` (16kHz/mono, `estimate_duration_sec`), `value_objects/timestamp.py`.
- `application/ports.py` — Protocols: `TranscriberPort`, `DiarizerPort` (`diarize` com `num/min/max_speakers`), `RefinerPort`, `SessionStore`, `AudioTempFiles`.
- `application/use_cases/` — `transcribe_audio.py`, `finalize_stream.py`, `refine_transcript.py` (1 arquivo por objetivo, ports injetados).
- `infrastructure/dispatching/queue.py` — single `asyncio.Queue`, one worker task per GPU (round-robin via queue, not explicit index). Filters missing GPUs via `torch.cuda.device_count()`, falls back to `["cpu"]`.
- `infrastructure/asr/faster_whisper.py` — one `Transcriber` per GPU, each with **dedicated single-thread `ThreadPoolExecutor`**. Keep `load` + `transcribe` on that same executor — CUDA context breaks if moved across threads. Compute: `int8_float16` on GPU, `int8` on CPU. `download_root="/app/models"`.
- `infrastructure/sessions/memory_store.py` — `Session`/`SessionManager`; channels `mic`/`system`; hints `min/max_speakers` por sessão. Audio is base64 PCM Int16LE 16kHz mono, assembled into WAV in-memory. Parciais: rótulos `Pessoa N` estáveis entre flushes via sobreposição com a timeline rotulada (`_resolve` em `speaker_rules`); `stop` sempre re-roda do zero, cache parcial é só fallback (`cached_diarization()`).
- `infrastructure/audio/wav.py` — canônico WAV (build/finalize/header) + `save_upload_to_temp`/`save_bytes_to_temp` + `WavTempFiles` (implements `AudioTempFiles`).
- `infrastructure/diarization/pyannote.py` — lazy global pipeline (`preload()` no lifespan), `pyannote/speaker-diarization-3.1` sob `pyannote.audio` v4 (I/O via torchcodec, auth com `token=`). v4 devolve `DiarizeOutput` — ler `.speaker_diarization`. Aceita `num/min/max_speakers`. Labels mapped to `Pessoa N`. `PyannoteDiarizer` implements `DiarizerPort`.
- `infrastructure/refine/llama_refiner.py` — OpenAI-compatible client to llama.cpp. Batches **50 segments/call**, `max_tokens=4096`, thinking desligado (`enable_thinking: False`, Qwen3). Fails open (returns original on LLM error).
- `api/routes/` — `health.py`, `transcribe.py`, `stream.py`, `refine.py`: routers finos (validar → use case → `to_wire`), cada um com `summary`/`description`/`responses` pro Swagger.
- `api/ws/` — `azure_protocol.py` (StreamConfig, ticks, phrase builders, SpeakerMapper) + `speech.py` (handler `/speech/stream`).
- `api/sse/events.py` — `partial_event_generator`: partial flush every ~3s; only new bytes since last flush (`<8000` bytes / ~0.5s skipped).
- `api/schemas.py` — camelCase fields (`sessionId`, `tStart/tEnd`, `durationSec`) + `Field()` docs pro Swagger. `StreamAction`/`AudioChunk` require `sessionId` in body **and** path `{session_id}` — send both. `StreamAction` aceita `minSpeakers`/`maxSpeakers` opcionais; `/transcribe` idem via form.
- `api/mappers.py` — `to_wire`/`to_domain` (único lugar de conversão). `api/deps.py` — DI via `app.state`.

## Gotchas

- Audio must be WAV 16kHz 16-bit mono; duration estimated as `(file_size-44)/32000`. Stereo must be downmixed client-side.
- Streaming speaker rules in `domain/services/speaker_rules.py:apply_stop_rules`: `mic` → `Eu`; `system` without diarize → `Sistema`. Diarization failure falls back to generic, never 500s. Parciais SSE têm locutor best-effort (só o `stop` é autoritativo).
- torch>=2.9 + `pyannote.audio` v4 (era pin `<2.9` na v3 pelo `AudioMetaData` removido; upstream wontfix). ctranslate2 (faster-whisper) ainda exige runtime cu12 → deps diretas `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (coexistem com cu13 do torch). Também pinado `pyannote.audio<4`→`<5` (v5 futura pode mudar a API).
- v4 lê áudio via torchcodec → `Dockerfile` instala ffmpeg do sistema (sem `libavutil`, `libtorchcodec_core*.so` não carrega).
- Diarization needs `HF_TOKEN` **plus** accepting terms on both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. Missing token = silent fallback, not error. Com 2+ vozes similares, considere `minSpeakers=2` (front tem o campo) — o clustering pode fundir a voz minoritária.
- `models/` (Whisper + `Qwen3-8B-Q4_K_M.gguf` ~5GB) is local-only, excluded from `deploy.sh` rsync (also excludes `.venv`, `.env`, `.git`). `entrypoint-refine.sh` baixa a lista `MODEL_FILES` se ausente/corrompida (<10MB).
- Corporate DNS blocking UDP 53 can break Docker build/model download — use local gateway DNS in compose if needed (see `docs/usage.md`).
- GPU layout (2x RTX 4000 Ada 20GB): `stt-api` com `CUDA_VISIBLE_DEVICES=1` + `WHISPER_GPUS=cuda:0` (só GPU física 1: Whisper + pyannote); `llama-refine` dedicado em `CUDA_VISIBLE_DEVICES=0` com `--main-gpu 0 --parallel 8 --ctx-size 32768`.
