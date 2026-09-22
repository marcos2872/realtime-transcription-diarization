# Contrato de API — transcript-service `0.1.0`

Serviço de **transcrição realtime com diarização de falantes**: o cliente abre
um WebSocket por canal de áudio, envia PCM16 e recebe frases com
`SPEAKER_00`, `SPEAKER_01`, … em tempo real.

- Transporte de streaming: **WebSocket** — `ws://HOST:8000/ws/streams/{stream_id}`
- Metadados/diagnóstico: **REST** — `GET /`, `GET /health`
- OpenAPI (só REST): `GET /openapi.json`

---

## 1. Formato de áudio

| Propriedade   | Valor                                              |
|---------------|----------------------------------------------------|
| Codec         | PCM linear 16-bit **signed little-endian**, mono  |
| Sample rate   | `16000` Hz (negociado no evento `ready`)           |
| Envio         | frames binários do WebSocket, qualquer tamanho     |

Um byte ímpar ao final de um frame **não é descartado**: é carregado para o
próximo frame (`carry`). Envie continuamente; o servidor acumula a timeline.

> Enviar ~1 s por frame (32000 bytes) é um bom compromisso entre latência e
> overhead. Frames muito pequenos (< 0,2 s) aumentam o tráfego sem melhorar a
> latência, pois o modelo emite por chunks de `TRANSCRIPT_CHUNK_MS`.

---

## 2. Ciclo de vida de um stream

```
cliente                                        servidor
  │  GET /ws/streams/reuniao-1 ─────────────────▶ │  abre stream (ou erro fatal)
  │ ◀──────────── {"type":"ready", ...} ───────── │
  │  [áudio binário] ───────────────────────────▶ │  partials + utterances
  │  [áudio binário] ───────────────────────────▶ │
  │  {"type":"end"} ────────────────────────────▶ │  flush + diarização final
  │ ◀──── utterance(s) pendentes (se houver) ──── │
  │ ◀──────────── {"type":"closed", ...} ──────── │
```

Fechar o socket abruptamente também libera o slot (o flush é descartado).

---

## 3. Cliente → servidor

### 3.1 Áudio (mensagem binária)

Bytes PCM16 conforme §1.

### 3.2 Fim do stream (texto JSON)

```json
{ "type": "end" }
```

Descarrega o áudio restante do decodificador, roda a diarização final,
emite as utterances pendentes e responde `closed`.

### 3.3 Ping (texto JSON)

```json
{ "type": "ping" }
```

Resposta imediata: `{"type": "pong"}`. Útil como keepalive.

Qualquer outro texto (não-JSON, ou `type` desconhecido) gera um erro
**não-fatal** `invalid_message` e a conexão continua.

---

## 4. Servidor → cliente (eventos)

Todos os eventos são JSON com campo discriminante `type`.

### 4.1 `ready` — stream admitido

```json
{
  "type": "ready",
  "stream_id": "reuniao-1",
  "language": "pt-BR",
  "sample_rate": 16000,
  "max_streams": 4,
  "partials": true
}
```

`partials` reflete o padrão do servidor (`TRANSCRIPT_EMIT_PARTIALS`), podendo
ser sobrescrito por conexão via query string: `/ws/streams/x?partials=false`.

### 4.2 `partial` — texto provisório da frase em aberto

```json
{ "type": "partial", "stream_id": "reuniao-1", "text": "Olá mundo" }
```

Emitido quando o texto parcial **muda**. Provisório por natureza: a frase só
é definitiva no evento `utterance`.

### 4.3 `utterance` — frase final com falante

```json
{
  "type": "utterance",
  "stream_id": "reuniao-1",
  "utterance_id": 0,
  "start": 0.0,
  "end": 3.0,
  "speaker": "SPEAKER_00",
  "text": "Olá mundo."
}
```

- `utterance_id`: inteiro monotônico **por stream**, a partir de 0.
- `start`/`end`: segundos absolutos desde o início do stream.
- `speaker`: `SPEAKER_NN` com zero-pad de 2 dígitos, ou `UNKNOWN` quando a
  diarização não cobriu a frase (ex.: stream curtíssimo encerrado antes da
  primeira janela de diarização).

### 4.4 `error`

```json
{ "type": "error", "code": "stream_limit", "message": "stream limit reached (4); ...", "fatal": true }
```

`fatal: true` → o servidor fecha a conexão em seguida; `false` → segue operando.

| `code`               | `fatal` | Quando                                                      |
|----------------------|---------|-------------------------------------------------------------|
| `invalid_stream_id`  | true    | `stream_id` vazio (ex.: `/ws/streams/%20`)                  |
| `stream_exists`      | true    | já existe um stream aberto com esse id                      |
| `stream_limit`       | true    | `max_streams` (padrão 4) já ocupados                        |
| `invalid_message`    | false   | texto não-JSON ou `type` desconhecido                       |
| `invalid_audio`      | true    | chunk de áudio vazio (não deve ocorrer via WS normal)       |
| `transcription_failed` | true  | falha do modelo ASR (ex.: GPU OOM)                          |
| `diarization_failed` | false   | falha da diarização — a transcrição **continua sem falante** |

### 4.5 `closed` — fim normal

```json
{ "type": "closed", "stream_id": "reuniao-1" }
```

Segue-se o fechamento WS com código `1000`.

### 4.6 `pong`

```json
{ "type": "pong" }
```

---

## 5. Códigos de fechamento do WebSocket

| Código | Significado                                             |
|--------|---------------------------------------------------------|
| `1000` | fim normal (`end` processado)                           |
| `4009` | stream não pôde ser aberto (`stream_exists`, `stream_limit`, `invalid_stream_id`) |
| `1011` | erro interno (ex.: `transcription_failed`)              |

---

## 6. REST

### 6.1 `GET /` — identidade e configuração

```json
{
  "name": "transcript-service",
  "version": "0.1.0",
  "language": "pt-BR",
  "sample_rate": 16000,
  "max_streams": 4,
  "asr": {
    "provider": "fake",
    "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "chunk_ms": 320
  },
  "diarization": {
    "provider": "fake",
    "pipeline": "pyannote/speaker-diarization-community-1"
  }
}
```

### 6.2 `GET /health` — saúde e ocupação

```json
{
  "status": "ok",
  "active_streams": 0,
  "max_streams": 4,
  "language": "pt-BR",
  "asr_provider": "fake",
  "diarization_provider": "fake",
  "device": "cpu"
}
```

O `HEALTHCHECK` do Docker usa este endpoint. `active_streams == max_streams`
significa que novos streams serão recusados com `stream_limit`.

---

## 7. Limites e comportamento

- **Streams paralelos**: `TRANSCRIPT_MAX_STREAMS` (padrão **4**, teto 64).
- **Latência de emissão**: o texto sai por chunks de `TRANSCRIPT_CHUNK_MS`
  (80/160/320/560/1120 ms — maior = mais preciso, mais latente); os timestamps
  de palavra descontam `TRANSCRIPT_LOOKAHEAD_S` do instante de chegada.
- **Atribuição de falante**: a diarização roda sobre janelas de
  `TRANSCRIPT_DIARIZATION_WINDOW_S` a cada `TRANSCRIPT_DIARIZATION_HOP_S`,
  começando após `TRANSCRIPT_DIARIZATION_MIN_S` de áudio. Frases só são
  emitidas com falante após o *watermark* de diarização cobri-las — antes
  disso ficam retidas (o `partial` continua fluindo).
- **Retenção de áudio**: o servidor guarda só a janela de diarização + 1 hop
  por stream (memória limitada mesmo em streams de horas).

---

## 8. Exemplo de cliente (Python)

```python
import asyncio
import json

import websockets

SAMPLE_RATE = 16_000


async def transcribe(pcm_chunks):
    async with websockets.connect("ws://localhost:8000/ws/streams/demo") as ws:
        print(await ws.recv())  # ready
        for chunk in pcm_chunks:  # bytes PCM16 mono 16 kHz
            await ws.send(chunk)
            async with asyncio.timeout(5):
                # parcial OU utterance OU erro de diarização (não-fatal)
                print(await ws.recv())
        await ws.send(json.dumps({"type": "end"}))
        async for raw in ws:
            event = json.loads(raw)
            print(event)
            if event["type"] in ("closed", "error"):
                break


asyncio.run(transcribe([b"\x00\x00" * SAMPLE_RATE] * 12))
```

---

## 9. Configuração (variáveis `TRANSCRIPT_*`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `TRANSCRIPT_HOST` / `TRANSCRIPT_PORT` | `0.0.0.0` / `8000` | bind do servidor |
| `TRANSCRIPT_MAX_STREAMS` | `4` | streams simultâneos |
| `TRANSCRIPT_SAMPLE_RATE` | `16000` | Hz esperados no WS |
| `TRANSCRIPT_LANGUAGE` | `auto` | `pt-BR`, `en-US`, … ou `auto` (detecção) |
| `TRANSCRIPT_EMIT_PARTIALS` | `true` | parciais por padrão (`?partials=` sobrescreve) |
| `TRANSCRIPT_ASR_PROVIDER` | `nemotron` | `nemotron` \| `fake` (sem GPU) |
| `TRANSCRIPT_ASR_MODEL` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | checkpoint HF |
| `TRANSCRIPT_CHUNK_MS` | `320` | 80/160/320/560/1120 |
| `TRANSCRIPT_LOOKAHEAD_S` | `1.4` | âncora dos timestamps de palavra |
| `TRANSCRIPT_DIARIZATION_PROVIDER` | `pyannote` | `pyannote` \| `fake` |
| `TRANSCRIPT_DIARIZATION_PIPELINE` | `pyannote/speaker-diarization-community-1` | pipeline HF (gated) |
| `TRANSCRIPT_DIARIZATION_WINDOW_S` | `30` | janela analisada por run |
| `TRANSCRIPT_DIARIZATION_HOP_S` | `5` | cadência entre runs |
| `TRANSCRIPT_DIARIZATION_OVERLAP_S` | `7` | sobreposição (< janela; hop+overlap ≤ janela) |
| `TRANSCRIPT_DIARIZATION_MIN_S` | `10` | áudio mínimo p/ 1º run |
| `TRANSCRIPT_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `TRANSCRIPT_HF_TOKEN` | — | **obrigatório** p/ pipeline pyannote gated |
