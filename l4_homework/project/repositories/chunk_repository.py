from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from models import Document, DocumentChunk


@dataclass(frozen=True)
class SimilarityHit:
    chunk: DocumentChunk
    score: float


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_chunks_batch(
        self,
        document_id: int,
        contents: list[str],
        embeddings: list[list[float]],
    ) -> list[DocumentChunk]:
        if len(contents) != len(embeddings):
            raise ValueError(
                f"contents and embeddings size mismatch: "
                f"{len(contents)} vs {len(embeddings)}"
            )
        chunks = [
            DocumentChunk(
                document_id=document_id,
                content=content,
                chunk_index=index,
                embedding=embedding,
            )
            for index, (content, embedding) in enumerate(zip(contents, embeddings))
        ]
        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def get_document_chunks(self, document_id: int) -> list[DocumentChunk]:
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SimilarityHit]:
        # 1 - cosine_distance = cosine_similarity in [0, 1] for normalized vectors
        similarity = (
            1 - DocumentChunk.embedding.cosine_distance(query_embedding)
        ).label("score")

        stmt = (
            select(DocumentChunk, similarity)
            .options(joinedload(DocumentChunk.document))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = self.db.execute(stmt).all()
        return [SimilarityHit(chunk=row[0], score=float(row[1])) for row in rows]

    def search_with_threshold(
        self,
        query_embedding: list[float],
        min_score: float = 0.4,
        top_k: int = 5,
    ) -> list[SimilarityHit]:
        hits = self.similarity_search(query_embedding, top_k=top_k)
        return [h for h in hits if h.score >= min_score]
