"""Timestamps — arredondamento canônico em 2 casas."""


def round2(value: float | int | None) -> float:
    """Arredonda para 2 casas; ``None``/inválido vira ``0.0``."""
    try:
        return round(float(value or 0.0), 2)
    except (TypeError, ValueError):
        return 0.0
