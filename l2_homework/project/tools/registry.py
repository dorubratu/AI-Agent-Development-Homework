from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(func: Callable) -> Callable:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 1 or not (
        isinstance(params[0].annotation, type)
        and issubclass(params[0].annotation, BaseModel)
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
        "params_model": params[0].annotation,
        "description": docstring,
    }
    return func
