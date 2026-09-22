"""Lazy public model exports keep data and launcher imports lightweight."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from speedrunning_plms.models import PLM, PLMConfig


__all__ = ["PLM", "PLMConfig"]


def __getattr__(name: str) -> type[PLM] | type[PLMConfig]:
    if name in {"PLM", "PLMConfig"}:
        from speedrunning_plms.models import PLM, PLMConfig

        return {"PLM": PLM, "PLMConfig": PLMConfig}[name]
    raise AttributeError(name)
