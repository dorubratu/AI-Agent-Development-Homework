# ─────────────────────────────────────────────────────────
# Slide 2 · Schema PostgreSQL pentru semantic cache
# Tabel dedicat cu coloană Vector(384) + index HNSW (cosine).
# 384 = dimensiunea embedding-urilor all-MiniLM-L6-v2.
# ─────────────────────────────────────────────────────────
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    query_text: Mapped[str] = mapped_column(Text)              # query original (debugging)
    query_embedding: Mapped[list[float]] = mapped_column(Vector(384))  # vectorul
    response: Mapped[str] = mapped_column(Text)               # răspunsul cache-uit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    hit_count: Mapped[int] = mapped_column(Integer, default=0)  # de câte ori a fost servit

    # Index HNSW pentru cosine distance — operatorul <=>.
    # vector_cosine_ops TREBUIE să se potrivească cu distanța din query,
    # altfel Postgres ignoră indexul și face full scan.
    __table_args__ = (
        Index(
            "ix_semantic_cache_embedding_hnsw",
            "query_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"query_embedding": "vector_cosine_ops"},
        ),
    )

# Necesită extensia pgvector activată o singură dată în DB:
#   CREATE EXTENSION IF NOT EXISTS vector;
