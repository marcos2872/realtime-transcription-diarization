# API — referência

A referência canônica é o **Swagger**, gerado do código:

- Swagger UI: `http://<host>:<porta>/docs`
- ReDoc: `http://<host>:<porta>/redoc`
- Schema: `http://<host>:<porta>/openapi.json`

Tags: `health`, `transcribe`, `streaming`, `refine`, `websocket`.
O WebSocket `/speech/stream` não aparece no OpenAPI (limitação do
padrão) — o protocolo está documentado em `usage.md#websocket-estilo-azure`
e no docstring de `src/api/ws/speech.py`.
