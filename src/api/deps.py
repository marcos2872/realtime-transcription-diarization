"""Dependências injetáveis da API (Dependency Inversion).

Rotas recebem ``Dispatcher``/``SessionManager``/``Refiner`` via
``Depends`` em vez de importar singletons globais — permite override
em testes com ``app.dependency_overrides``.
"""

from __future__ import annotations

from fastapi import Request


def get_dispatcher(request: Request):
    """Devolve o ``Dispatcher`` guardado em ``app.state``."""
    return request.app.state.dispatcher


def get_sessions(request: Request):
    """Devolve o ``SessionManager`` guardado em ``app.state``."""
    return request.app.state.sessions


def get_refiner(request: Request):
    """Devolve o ``Refiner`` guardado em ``app.state``."""
    return request.app.state.refiner


def get_settings(request: Request):
    """Devolve o ``Settings`` guardado em ``app.state``."""
    return request.app.state.settings
