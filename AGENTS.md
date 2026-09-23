# AGENTS.md — stt-api

FastAPI remote transcription server: faster-whisper `large-v3` (multi-GPU) + optional pyannote diarization + LLM refine via llama.cpp (Qwen 2.5 7B GGUF).

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

- `main.py` — composition root `src.main:app` (enxuto, ~100 linhas): lifespan, CORS, routers, `app.state` (dispatcher/sessions/refiner). Swagger metadata aqui (`/docs`, `/redoc`, `/openapi.json`).
- `config.py` — `pydantic-settings`, reads `.env` (see `.env.example`). `WHISPER_GPUS` is comma string, parsed via `whisper_gpus_list`. `REFINE_BASE_URL` differs: `localhost:8080` local vs `http://llama-refine:8080/v1` in compose.
- `domain/` — regras puras, zero deps externas: `entities/transcript.py` (`SegmentData`, `TranscriptData`, `build_transcript`), `services/speaker_rules.py` (`Eu`/`Sistema`/`Pessoa N`/`Locutor`, `apply_stop_rules`), `value_objects/audio_format.py` (16kHz/mono, `estimate_duration_sec`), `value_objects/timestamp.py`.
- `application/ports.py` — Protocols: `TranscriberPort`, `DiarizerPort`, `RefinerPort`, `SessionStore`, `AudioTempFiles`.
- `application/use_cases/` — `transcribe_audio.py`, `finalize_stream.py`, `refine_transcript.py` (1 arquivo por objetivo, ports injetados).
- `infrastructure/dispatching/queue.py` — single `asyncio.Queue`, one worker task per GPU (round-robin via queue, not explicit index). Filters missing GPUs via `torch.cuda.device_count()`, falls back to `["cpu"]`.
- `infrastructure/asr/faster_whisper.py` — one `Transcriber` per GPU, each with **dedicated single-thread `ThreadPoolExecutor`**. Keep `load` + `transcribe` on that same executor — CUDA context breaks if moved across threads. Compute: `int8_float16` on GPU, `int8` on CPU. `download_root="/app/models"`.
- `infrastructure/sessions/memory_store.py` — `Session`/`SessionManager`; channels `mic`/`system`. Audio is base64 PCM Int16LE 16kHz mono, assembled into WAV in-memory. `cached_diarization()` exposes the partial diarization cache to the `stop` use case.
- `infrastructure/audio/wav.py` — canônico WAV (build/finalize/header) + `save_upload_to_temp`/`save_bytes_to_temp` + `WavTempFiles` (implements `AudioTempFiles`).
- `infrastructure/diarization/pyannote.py` — lazy global pipeline, `pyannote/speaker-diarization-3.1`. Labels mapped to `Pessoa N`. `PyannoteDiarizer` implements `DiarizerPort`.
- `infrastructure/refine/llama_refiner.py` — OpenAI-compatible client to llama.cpp. Batches **50 segments/call**, `max_tokens=4096`. Fails open (returns original on LLM error).
- `api/routes/` — `health.py`, `transcribe.py`, `stream.py`, `refine.py`: routers finos (validar → use case → `to_wire`), cada um com `summary`/`description`/`responses` pro Swagger.
- `api/ws/` — `azure_protocol.py` (StreamConfig, ticks, phrase builders, SpeakerMapper) + `speech.py` (handler `/speech/stream`).
- `api/sse/events.py` — `partial_event_generator`: partial flush every ~3s; only new bytes since last flush (`<8000` bytes / ~0.5s skipped).
- `api/schemas.py` — camelCase fields (`sessionId`, `tStart/tEnd`, `durationSec`) + `Field()` docs pro Swagger. `StreamAction`/`AudioChunk` require `sessionId` in body **and** path `{session_id}` — send both.
- `api/mappers.py` — `to_wire`/`to_domain` (único lugar de conversão). `api/deps.py` — DI via `app.state`.

## Gotchas

- Audio must be WAV 16kHz 16-bit mono; duration estimated as `(file_size-44)/32000`. Stereo must be downmixed client-side.
- Streaming speaker rules in `domain/services/speaker_rules.py:apply_stop_rules`: `mic` → `Eu`; `system` without diarize → `Sistema`. Diarization failure falls back to generic, never 500s.
- `infrastructure/diarization/pyannote.py` monkey-patches `huggingface_hub.hf_hub_download` (`use_auth_token` → `token`) — pyannote 3.4.0 vs `huggingface_hub>=0.20` incompatibility. Do not remove.
- Diarization needs `HF_TOKEN` **plus** accepting terms on both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. Missing token = silent fallback, not error.
- `models/` (Whisper + `qwen2.5-7b-instruct-q4_k_m` 2×~2GB shards) is local-only, excluded from `deploy.sh` rsync (also excludes `.venv`, `.env`, `.git`). `entrypoint-refine.sh` re-downloads GGUF shards if missing or <10MB.
- Corporate DNS blocking UDP 53 can break Docker build/model download — use local gateway DNS in compose if needed (see `docs/usage.md`).
- GPU layout: `stt-api` uses `CUDA_VISIBLE_DEVICES=0,1`; `llama-refine` pins `CUDA_VISIBLE_DEVICES=0` with `--main-gpu 0 --parallel 8 --ctx-size 32768`.
