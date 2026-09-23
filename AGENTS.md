# AGENTS.md — stt-api

FastAPI remote transcription server: faster-whisper `large-v3` (multi-GPU) + optional pyannote diarization + LLM refine via llama.cpp (Qwen 2.5 7B GGUF).

## Commands (use `uv`, not `pip`/`python`)

```bash
uv sync                                        # install (Python >=3.11,<3.13; Docker uses 3.12-slim)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload   # local dev (needs CUDA GPU)
docker compose up --build -d                   # production (recommended)
docker compose logs -f                         # logs
curl http://localhost:4321/health              # health (compose maps 4321->8000; local is :8000)
uv run test_api.py --audio ~/audio.wav --diarize        # batch test
uv run test_streaming.py --audio ~/audio.wav --diarize  # streaming test
./deploy.sh [user]                             # rsync to 192.168.3.81 (default user `administrador`)
```

No pytest/unit suite. Verify with `/health` + `test_api.py` / `test_streaming.py`. Test scripts have hardcoded `SERVER_URL`/`AUDIO_PATH` defaults — always override with `--url`/`--audio`.

## Architecture (`src/`)

- `main.py` — entrypoint `src.main:app`; lifespan starts `dispatcher`. All routes here.
- `config.py` — `pydantic-settings`, reads `.env` (see `.env.example`). `WHISPER_GPUS` is comma string, parsed via `whisper_gpus_list`. `REFINE_BASE_URL` differs: `localhost:8080` local vs `http://llama-refine:8080/v1` in compose.
- `dispatcher.py` — single `asyncio.Queue`, one worker task per GPU (round-robin via queue, not explicit index). Filters missing GPUs via `torch.cuda.device_count()`, falls back to `["cpu"]`.
- `transcriber.py` — one `Transcriber` per GPU, each with **dedicated single-thread `ThreadPoolExecutor`**. Keep `load` + `transcribe` on that same executor — CUDA context breaks if moved across threads. Compute: `int8_float16` on GPU, `int8` on CPU. `download_root="/app/models"`.
- `session.py` — streaming sessions; channels `mic`/`system`. Audio is base64 PCM Int16LE 16kHz mono, assembled into WAV in-memory. Partial flush every ~3s in SSE loop; only new bytes since last flush (`<8000` bytes / ~0.5s skipped).
- `diarizer.py` — lazy global pipeline, `pyannote/speaker-diarization-3.1`. Labels mapped to `Pessoa N`.
- `refiner.py` — OpenAI-compatible client to llama.cpp. Batches **50 segments/call**, `max_tokens=4096`. Fails open (returns original on LLM error).
- `schemas.py` — camelCase fields (`sessionId`, `tStart/tEnd`, `durationSec`). `StreamAction`/`AudioChunk` require `sessionId` in body **and** path `{session_id}` — send both.

## Gotchas

- Audio must be WAV 16kHz 16-bit mono; duration estimated as `(file_size-44)/32000`. Stereo must be downmixed client-side (see `test_streaming.py`).
- Streaming speaker rules in `main.py:stream_action(stop)`: `mic` → `Eu`; `system` without diarize → `Sistema`. Diarization failure falls back to generic, never 500s.
- `diarizer.py` monkey-patches `huggingface_hub.hf_hub_download` (`use_auth_token` → `token`) — pyannote 3.4.0 vs `huggingface_hub>=0.20` incompatibility. Do not remove.
- Diarization needs `HF_TOKEN` **plus** accepting terms on both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. Missing token = silent fallback, not error.
- `models/` (Whisper + `qwen2.5-7b-instruct-q4_k_m` 2×~2GB shards) is local-only, excluded from `deploy.sh` rsync (also excludes `.venv`, `.env`, `.git`). `entrypoint-refine.sh` re-downloads GGUF shards if missing or <10MB.
- Corporate DNS blocking UDP 53 can break Docker build/model download — use local gateway DNS in compose if needed (see README §3).
- GPU layout: `stt-api` uses `CUDA_VISIBLE_DEVICES=0,1`; `llama-refine` pins `CUDA_VISIBLE_DEVICES=0` with `--main-gpu 0 --parallel 8 --ctx-size 32768`.
