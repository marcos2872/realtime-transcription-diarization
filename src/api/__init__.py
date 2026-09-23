"""Camada de API — schemas de wire, rotas finas, SSE e WebSocket.

Schemas usam camelCase no wire (compatibilidade preservada) e
``Field(description, examples)`` para alimentar o Swagger em
``/docs``. A conversão wire ↔ domínio vive em ``mappers.py``.
"""
