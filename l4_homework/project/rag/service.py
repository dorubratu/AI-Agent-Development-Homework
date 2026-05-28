from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from repositories import ChunkRepository, SimilarityHit
from .embedder import Embedder


@dataclass(frozen=True)
class SearchHit:
    filename: str
    chunk_index: int
    content: str
    score: float
    doc_type: str

    @classmethod
    def from_similarity(cls, hit: SimilarityHit) -> "SearchHit":
        return cls(
            filename=hit.chunk.document.filename,
            chunk_index=hit.chunk.chunk_index,
            content=hit.chunk.content,
            score=hit.score,
            doc_type=hit.chunk.document.doc_type,
        )


class RAGService:
    def __init__(
        self,
        db: Session,
        embedder: Embedder | None = None,
        default_threshold: float = 0.35,
    ) -> None:
        self.repo = ChunkRepository(db)
        self.embedder = embedder or Embedder()
        self.default_threshold = default_threshold

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        query_emb = self.embedder.encode_one(query)
        hits = self.repo.similarity_search(query_emb, top_k=top_k)
        return [SearchHit.from_similarity(h) for h in hits]

    def search_with_threshold(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        threshold = self.default_threshold if min_score is None else min_score
        hits = self.search(query, top_k=top_k)
        return [h for h in hits if h.score >= threshold]

    def render_context(self, hits: list[SearchHit]) -> str:
        if not hits:
            return ""
        return "\n\n".join(
            f"[{h.filename} | chunk {h.chunk_index} | score {h.score:.2f}]\n{h.content}"
            for h in hits
        )
