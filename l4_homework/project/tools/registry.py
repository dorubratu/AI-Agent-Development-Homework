from __future__ import annotations

import inspect
import typing
from typing import Any, Callable

from pydantic import BaseModel

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(func: Callable) -> Callable:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Resolve forward refs — needed when the tool module uses
    # `from __future__ import annotations` (PEP 563), which keeps the
    # annotation as a string until explicitly evaluated.
    hints = typing.get_type_hints(func)
    annotation = hints.get(params[0].name) if params else None

    if len(params) != 1 or not (
        isinstance(annotation, type) and issubclass(annotation, BaseModel)
    ):
        raise TypeError(
            f"{func.__name__}: param unic de tip BaseModel obligatoriu"
        )

    docstring = (func.__doc__ or "").strip()
    if not docstring:
        raise ValueError(
            f"{func.__name__}: docstring obligatoriu — devine description "
            f"vizibil pentru LLM."
        )
    if len(docstring) < 15:
        raise ValueError(
            f"{func.__name__}: docstring prea scurt ({len(docstring)} "
            f"caractere). LLM-ul are nevoie de min 15 ca să decidă."
        )

    TOOL_REGISTRY[func.__name__] = {
        "func": func,
        "params_model": annotation,
        "description": docstring,
    }
    return func
