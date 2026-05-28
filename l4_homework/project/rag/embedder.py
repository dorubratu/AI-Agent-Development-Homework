from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


class Embedder:
    _model_cache: dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @property
    def model(self):
        if self.model_name not in Embedder._model_cache:
            from sentence_transformers import SentenceTransformer

            Embedder._model_cache[self.model_name] = SentenceTransformer(self.model_name)
        return Embedder._model_cache[self.model_name]

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        vectors = self.model.encode(
            items, convert_to_numpy=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
