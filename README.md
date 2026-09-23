# STT API — Servidor de Transcrição Remoto

Servidor FastAPI para transcrição de áudio via **Whisper large-v3** com suporte a **múltiplas GPUs**, **streaming**, **diarização opcional** (pyannote) e **refinamento via LLM** (Qwen 2.5 7B Instruct).

## Arquitetura

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

- **2 GPUs**: GPU 0 roda Whisper + llama.cpp (compartilhada), GPU 1 roda Whisper dedicada
- **Round-robin**: requisições de transcrição distribuídas entre as GPUs via `asyncio.Queue`
- **Refinamento**: llama.cpp server com Qwen 2.5 7B Instruct GGUF (Q4_K_M), ~69 t/s

## Rotas da API

### `GET /health`

Health check. Retorna status do servidor, GPUs disponíveis, se Whisper está carregado e endpoint de refine.

```json
{
  "status": "ok",
  "gpus": ["cuda:0", "cuda:1"],
  "whisperLoaded": true,
  "refineEndpoint": "http://llama-refine:8080/v1",
  "activeSessions": 0
}
```

### `POST /transcribe`

Transcrição por lote — envia um arquivo WAV e recebe a transcrição completa.

**Parâmetros** (multipart/form-data):
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `audio` | File | — | Arquivo WAV (16kHz, 16-bit, mono) |
| `language` | string | `"pt"` | Código do idioma (ISO 639-1) |
| `sessionId` | string | `null` | ID opcional para rastreamento |
| `diarize` | bool | `false` | Se `true`, executa diarização (requer HF_TOKEN) |

**Resposta:**
```json
{
  "sessionId": "abc123def456",
  "segments": [
    {"speaker": "Locutor", "text": "...", "tStart": 6.5, "tEnd": 9.8}
  ],
  "participants": ["Locutor"],
  "durationSec": 83.6,
  "language": "pt"
}
```

### `POST /stream/{session_id}`

Gerencia sessões de streaming.

**Body** (`StreamAction`):
```json
{"action": "start", "language": "pt", "channels": ["mic", "system"]}
// ou
{"action": "stop"}
```

- `action: "start"` — cria uma nova sessão de streaming
- `action: "stop"` — finaliza a sessão, transcreve todo o áudio acumulado e retorna o resultado

### `POST /stream/{session_id}/audio`

Envia um chunk de áudio para uma sessão ativa.

**Body** (`AudioChunk`):
```json
{
  "channel": "system",
  "seq": 1,
  "data": "base64_encoded_pcm_int16le_16khz"
}
```

O áudio deve ser PCM Int16LE 16kHz mono, codificado em base64. O header de 8 bytes (channel_id + seq uint32 LE) é opcional.

### `GET /stream/{session_id}/events`

SSE (Server-Sent Events) para eventos em tempo real durante o streaming.

> ⚠️ **Placeholder**: atualmente só envia heartbeats (`{"event": "heartbeat", "data": "ping"}`) a cada 2s. A transcrição parcial em tempo real será implementada em versão futura.

### `WS /speech/stream` — streaming com formato estilo Azure STT

WebSocket para qualquer app que chame a API do Azure diretamente (sem SDK).
Protocolo simples sobre PCM Int16LE 16kHz mono:

| Direção | Mensagem |
|---|---|
| C→S texto | `{"type":"config","language":"pt","locale":"pt-BR","diarize":true,"maxSpeakers":4,"interimIntervalMs":1000,"words":true}` (primeira mensagem, obrigatória) |
| C→S binário | PCM bruto (ou texto `{"type":"audio","data":"base64..."}`) |
| C→S texto | `{"type":"end"}` — finaliza e transcreve o áudio acumulado |
| S→C texto | `{"path":"turn.start"}` |
| S→C texto | `{"path":"speech.hypothesis","Text":"...","Offset":17000000,"Duration":5000000}` — interim mutável, sem locutor (como no Azure) |
| S→C texto | `{"path":"speech.phrase","RecognitionStatus":"Success","DisplayText":"...","Offset":...,"Duration":...,"Speaker":1,"Locale":"pt-BR","Words":[...]}` — uma por segmento; `Speaker` (int) só com diarização; `NoMatch` se não houver fala |
| S→C texto | `{"path":"turn.end"}` — em seguida a conexão fecha (1000); erros usam `{"path":"error","code":"...","message":"..."}` |

`Offset`/`Duration` em ticks de 100ns (mesma unidade do Azure). Teste com:
`uv run test_ws.py --audio ~/audio.wav --url http://localhost:4321 --diarize`

### `POST /refine`

Refina uma transcrição usando LLM (Qwen 2.5 7B Instruct).

**Body** (`RefineRequest`):
```json
{
  "transcription": { /* TranscriptionResult */ },
  "model": "Qwen2.5-7B-Instruct",
  "prompt": "Instruções opcionais para o refinamento"
}
```

**Resposta:**
```json
{
  "refined": { /* TranscriptionResult refinado */ },
  "model": "Qwen2.5-7B-Instruct",
  "tokensUsed": 0
}
```

## Como usar

### Desenvolvimento local

```bash
# Instalar dependências
uv sync

# Rodar servidor (precisa de GPU com CUDA)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose (produção — recomendado)

```bash
# Subir com Docker (requer GPU NVIDIA + nvidia-docker)
docker compose up --build -d

# Ver logs
docker compose logs -f
```

### Testar a API

```bash
# Health check
curl http://localhost:4321/health

# Transcrição simples
uv run test_api.py --audio ~/audio.wav

# Transcrição + diarização
uv run test_api.py --audio ~/audio.wav --diarize

# Streaming
uv run test_streaming.py --audio ~/audio.wav
```

### Front de teste (web/)

```bash
cd web && npm install && npm run dev   # http://localhost:3000
```

Página única com 4 abas (batch, streaming SSE, WebSocket formato Azure,
refine), captura do mic em PCM 16kHz mono, toggle de diarização e log de
rede. O servidor libera CORS para o browser (`CORSMiddleware` em `src/main.py`).

## Peculiaridades e problemas conhecidos

### 1. 🚫 Diarização — modelos gated (pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0)

A diarização depende de **dois modelos gated** no HuggingFace. Ambos precisam ter os termos aceitos:

1. Criar conta em https://huggingface.co
2. Acessar **https://huggingface.co/pyannote/speaker-diarization-3.1** e clicar em "Agree and access repository"
3. Acessar **https://huggingface.co/pyannote/segmentation-3.0** e clicar em "Agree and access repository"
4. Criar um token em https://huggingface.co/settings/tokens
5. Definir `HF_TOKEN=<seu_token>` no `.env`

O `speaker-diarization-3.1` baixa primeiro (～1.2 GB); em seguida o `segmentation-3.0` (～200 MB) é baixado automaticamente.

**Se o token não tiver acesso a qualquer um deles**, o servidor loga o erro e continua sem diarização (fallback para `Locutor` genérico).

### 2. 🐍 Incompatibilidade pyannote + huggingface_hub

`pyannote.audio` 3.4.0 usa o parâmetro `use_auth_token` (deprecado) em chamadas internas para `hf_hub_download()`. O `huggingface_hub >= 0.20` removeu esse parâmetro em favor de `token`.

**Solução**: o `src/diarizer.py` aplica um monkey-patch em `huggingface_hub.hf_hub_download` que converte `use_auth_token` → `token` automaticamente.

### 3. 🔌 DNS em redes corporativas

Em redes que bloqueiam UDP 53 para resolvers externos (8.8.8.8), o build do Docker e o download de modelos podem falhar.

**Solução**: usar o DNS local da rede no `docker-compose.yml`:

```yaml
dns:
  - 192.168.3.1   # gateway local
  - 8.8.8.8       # fallback
```

### 4. 🎧 Streaming — transcrição parcial em tempo real

O endpoint `/stream/{session_id}/events` envia resultados **parciais** via SSE a cada ~3s
enquanto o áudio está sendo acumulado. Os segmentos aparecem no cliente conforme são
detectados pelo Whisper.

**Limitação**: cada flush parcial transcreve apenas o áudio novo desde o último flush.
Segmentos podem aparecer com atraso ou duplicados entre parciais e resultado final.

Clientes que não querem SSE podem ignorar o endpoint e obter o resultado completo
via `action: "stop"` — o comportamento anterior é preservado.

### 5. 🗣️ Refinamento em lote

O refiner divide a transcrição em batches de **50 segmentos** para evitar estouro de contexto do LLM (4096 tokens por slot no llama.cpp com `--ctx-size 32768` e 8 slots paralelos).

### 6. 🖥️ GPU fallback

Se CUDA não estiver disponível, o Whisper faz fallback automático para CPU. O desempenho será significativamente pior. Para forçar GPU, certifique-se de que as bibliotecas CUDA estão instaladas (`cublas64_12.dll`, `cuDNN`, etc.).

## Configuração (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Modelo Whisper |
| `WHISPER_COMPUTE` | `int8_float16` | Tipo de computação |
| `WHISPER_GPUS` | `cuda:0,cuda:1` | GPUs disponíveis |
| `REFINE_BASE_URL` | `http://llama-refine:8080/v1` | URL do llama.cpp |
| `REFINE_API_KEY` | — | API key (opcional) |
| `REFINE_MODEL` | `Qwen2.5-7B-Instruct` | Modelo de refinamento |
| `HF_TOKEN` | — | Token HuggingFace (obrigatório para diarização) |
| `MAX_FILE_SIZE_MB` | `500` | Tamanho máximo de áudio |
| `SESSION_TIMEOUT_MIN` | `60` | Timeout de sessão |
| `HOST` | `0.0.0.0` | Host do servidor |
| `PORT` | `8000` | Porta do servidor |
| `LOG_LEVEL` | `info` | Nível de log |
