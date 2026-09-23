# Uso — STT API

> Referência interativa com exemplos: `/docs` (Swagger) e `/redoc`.
> Arquitetura: `architecture.md`. Código: `code.md`.

Áudio sempre em **WAV 16kHz 16-bit mono**. Estéreo precisa de
downmix no cliente.

## Subir

```bash
# Desenvolvimento local (precisa de GPU com CUDA)
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Produção com Docker (recomendado, requer nvidia-docker)
docker compose up --build -d
docker compose logs -f

# Health (compose mapeia 4321→8000; local é :8000)
curl http://localhost:4321/health
```

## Configuração (`.env`)

Copie `.env.example` para `.env`. Principais variáveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Modelo Whisper |
| `WHISPER_COMPUTE` | `int8_float16` | Tipo de computação (`int8` em CPU) |
| `WHISPER_GPUS` | `cuda:0` | GPUs do Whisper (no compose = só a física 1) |
| `REFINE_BASE_URL` | `http://llama-refine:8080/v1` | URL do llama.cpp |
| `REFINE_API_KEY` | — | Opcional |
| `REFINE_MODEL` | `Qwen3-8B` | Modelo de refinamento |
| `HF_TOKEN` | — | HuggingFace (obrigatório p/ diarização) |
| `MAX_FILE_SIZE_MB` | `500` | Limite de upload |
| `SESSION_TIMEOUT_MIN` | `60` | Expiração de sessão |
| `LOG_LEVEL` | `info` | Nível de log |

## Lote (`POST /transcribe`)

```bash
curl -X POST http://localhost:4321/transcribe \
  -F "audio=@~/audio.wav" -F "language=pt" -F "diarize=true"
```

Campos (multipart): `audio` (WAV), `language` (`pt`), `sessionId`
(opcional), `diarize` (`false`), `minSpeakers`/`maxSpeakers`
(opcionais). Resposta: `sessionId`, `segments`
(`speaker`, `text`, `tStart`, `tEnd`), `participants`,
`durationSec`, `language`.

## Streaming (`POST /stream/{id}`, `/audio`, `/events`)

Teste manual pela aba SSE do front `web/` ou via curl:

1. `POST /stream/{id}` body `{"sessionId":"{id}","action":"start",
   "language":"pt","channels":["mic","system"],"diarize":true,
   "minSpeakers":2}` (este último evita fundir vozes similares).
   (`sessionId` vai no body **e** no path.)
2. `POST /stream/{id}/audio` com
   `{"sessionId":"{id}","channel":"system","seq":1,"data":"base64..."}`.
3. `GET /stream/{id}/events` (SSE): `partial` a cada ~3s;
   `heartbeat/closed` ao final. Opcional — `stop` sozinho já basta.
4. `POST /stream/{id}` body `{"sessionId":"{id}","action":"stop"}`
   retorna a transcrição (`mic` → `Eu`; `system` sem diarize →
   `Sistema`).

## WebSocket estilo Azure (`/speech/stream`)

Para apps que falam com o Azure sem SDK. Mesagens:

| Direção | Mensagem |
|---|---|
| C→S texto | `{"type":"config","language":"pt","locale":"pt-BR","diarize":true,"maxSpeakers":4,"interimIntervalMs":1000,"words":true}` (primeira, obrigatória) |
| C→S binário | PCM bruto (ou texto `{"type":"audio","data":"base64..."}`) |
| C→S texto | `{"type":"end"}` |
| S→C texto | `{"path":"turn.start"}` |
| S→C texto | `{"path":"speech.hypothesis","Text":"...","Offset":...,"Duration":...}` (interim, sem locutor) |
| S→C texto | `{"path":"speech.phrase","RecognitionStatus":"Success","DisplayText":"...","Offset":...,"Duration":...,"Speaker":1,"Locale":"pt-BR","Words":[...]}` (`Speaker` só com diarização; `NoMatch` se sem fala) |
| S→C texto | `{"path":"turn.end"}` (depois fecha 1000; erros: `{"path":"error","code":"...","message":"..."}`) |

`Offset`/`Duration` em ticks de 100ns (unidade do Azure). Teste
manual pela aba WebSocket do front `web/`.

## Refine (`POST /refine`)

```json
{"transcription": { "...TranscriptionResult..." }, "model": "Qwen3-8B"}
```

Qwen3-8B Q4_K_M em GPU dedicada, thinking desligado.
Refinamento em batches de 50 segmentos (`max_tokens=4096`). Falha
no LLM retorna o original.

## Front de teste (`web/`)

```bash
cd web && npm install && npm run dev   # http://localhost:3000
```

4 abas (batch, SSE, WebSocket Azure, refine), captura do mic em PCM
16kHz mono, toggle de diarização e log de rede. CORS liberado.

## Problemas conhecidos

1. **Diarização — modelos gated.** Aceitar termos em
   `pyannote/speaker-diarization-3.1` **e**
   `pyannote/segmentation-3.0`, criar token em
   `huggingface.co/settings/tokens` e definir `HF_TOKEN`. Sem acesso,
   fallback para `Locutor` (log + continua, sem 500).
2. **pyannote v4.** Auth com `token=` (o monkey-patch da v3 foi
   removido); I/O via torchcodec exige ffmpeg no container
   (`Dockerfile` instala); `pipeline()` devolve `DiarizeOutput`
   (ler `.speaker_diarization`). Com vozes similares, use
   `minSpeakers` — o clustering pode fundir a voz minoritária.
3. **DNS corporativo.** Se UDP 53 externo for bloqueado, o build e o
   download de modelos falham — usar DNS do gateway local no
   `docker-compose.yml`.
4. **Parciais SSE.** Cada flush transcreve só o áudio novo; segmentos
   podem atrasar/duplicar entre parciais e final.
5. **CPU fallback.** Sem CUDA o Whisper roda em CPU (bem mais lento).

6. **`torchaudio` sem `AudioMetaData`.** Erro
   `module 'torchaudio' has no attribute 'AudioMetaData'` significa
   imagem com torchaudio 2.9+ (pyannote 3.x é incompatível —
   upstream marcou wontfix). O `pyproject.toml` pina
   `torch/torchaudio<2.9`: rebuild com o `uv.lock` atual
   (`docker compose up --build -d`) resolve.
