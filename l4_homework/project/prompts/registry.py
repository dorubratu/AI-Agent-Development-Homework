from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import yaml
from jinja2 import StrictUndefined, Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    prompt: str
    description: str = ""


class PromptRegistry:
    def __init__(self, folder: Union[str, Path]):
        self._folder = Path(folder)
        self._templates: dict[str, PromptTemplate] = self._load()

    def _load(self) -> dict[str, PromptTemplate]:
        templates: dict[str, PromptTemplate] = {}
        for path in self._folder.rglob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            tpl = PromptTemplate(**data)
            templates[tpl.name] = tpl
        return templates

    def reload(self) -> None:
        self._templates = self._load()

    def render(self, name: str, **variabile) -> str:
        if name not in self._templates:
            raise KeyError(f"Prompt '{name}' inexistent în registry.")
        template = self._templates[name]
        return Template(template.prompt, undefined=StrictUndefined).render(**variabile)

    def list_names(self) -> list[str]:
        return sorted(self._templates.keys())


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    folder = Path(__file__).resolve().parent
    return PromptRegistry(folder=folder)
