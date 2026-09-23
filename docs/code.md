# Código — mapa, convenções e como estender

> Arquitetura e fluxos: ver `architecture.md`. Uso e troubleshooting:
> ver `usage.md`. Referência interativa: `/docs` (Swagger).

## Mapa de módulos

| Caminho | Responsabilidade | Pode importar |
|---|---|---|
| `src/main.py` | Composition root (<100 linhas): app, lifespan, CORS, routers, `app.state` | `api/*`, `config`, `infrastructure/*` |
| `src/config.py` | `Settings` (pydantic-settings, `.env`) | stdlib |
| `src/domain/` | Regras puras: entidades, value objects, `speaker_rules` | stdlib + `domain` |
| `src/application/ports.py` | Protocols (DIP) | `domain` |
| `src/application/use_cases/` | Orquestração (1 arquivo por objetivo) | `domain`, `ports` |
| `src/infrastructure/` | Adapters concretos | `domain`, `ports`, `config`, libs externas |
| `src/api/routes/` | Rotas finas: validar → use case → `to_wire` | `application`, `infrastructure`, `api/*` |
| `src/api/schemas.py` | Wire camelCase + `Field()` p/ Swagger | pydantic |
| `src/api/mappers.py` | Conversão wire ↔ domínio (único lugar) | `api.schemas`, `domain` |
| `src/api/deps.py` | DI via `app.state` (`Depends`) | fastapi |
| `src/api/ws/`, `src/api/sse/` | Protocolo Azure e gerador de parciais | conforme acima |

## Convenções (Clean Code + SOLID)

- **SRP:** 1 módulo = 1 motivo para mudar. Rotas não têm regra de
  negócio; regra de locutor vive em
  `domain/services/speaker_rules.py`, não nas rotas.
- **DIP:** use cases recebem ports injetados (`transcriber`,
  `diarizer`, `tmp_files`) — nunca importam `infrastructure`.
- **Nomes revelam intenção:** `finalize_stream`, `WavTempFiles`,
  `apply_stop_rules`. Evitar `dados`, `info`, `temp`, abreviações.
- **Funções pequenas, early returns.** Sem flags booleanas que mudam
  fluxo interno — `diarize` vira caminhos explícitos no use case.
- **Sem side effects ocultos:** IO (tmp files, rede, GPU) só em
  `infrastructure/` e rotas; domínio é puro e testável sem GPU.
- **Erros nunca engolidos:** diarização/refine logam e fazem
  fallback documentado (genérico/original) — nunca 500 silencioso,
  nunca `except` vazio.
- **Comentários explicam o porquê** (ex: afinidade CUDA por thread,
  `DiarizeOutput` da v4). O código já diz o quê.

## Como estender

- **Novo transcriber:** implemente `TranscriberPort.dispatch()` em
  `infrastructure/` e injete na rota (ou `app.state`). O domínio não
  muda.
- **Nova diarização:** implemente `DiarizerPort` (`diarize` +
  `assign_speakers`). Rótulos novos devem passar por
  `speaker_rules` para manter o vocabulário (`Eu`, `Sistema`,
  `Pessoa N`, `Locutor`).
- **Novo endpoint:** crie o use case em `application/use_cases/`,
  o router fino em `api/routes/` com `summary`/`description`/
  `responses` para o Swagger, e registre em `main.py`.
- **Mudança de wire:** é BREAKING CHANGE — atualize `schemas.py`,
  `mappers.py`, `docs/usage.md` e o front `web/` juntos.

## Gotchas (não quebrar)

1. `load` + `transcribe` no **mesmo executor dedicado** por GPU
   (contexto CUDA).
2. `download_root="/app/models"` no Whisper.
3. Flush parcial só de **bytes novos**; `<8000` bytes ignorado.
4. `token=` (não `use_auth_token`) em `diarization/pyannote.py`
   (pyannote v4).
5. `sessionId` exigido no body **e** no path (compatibilidade).
6. `models/` (Whisper + GGUF) é local, fora do deploy rsync;
   `entrypoint-refine.sh` baixa a lista `MODEL_FILES` se ausente/<10MB.
7. Parciais SSE têm locutor best-effort; só o `stop` é autoritativo.

## Testes

```bash
python3 -m pytest tests/ -q   # domínio + use cases (sem GPU/rede)
```

Domínio e use cases usam fakes (`FakeTranscriber`, `FakeTmp`,
`FakeDiarizer`) — sem CUDA, HF ou llama.cpp. Teste de protocolo
Azure pula se `pydantic` não estiver instalado. Verificação
ponta-a-ponta manual: `/health` + front `web/` (abas batch, SSE,
WebSocket, refine).
