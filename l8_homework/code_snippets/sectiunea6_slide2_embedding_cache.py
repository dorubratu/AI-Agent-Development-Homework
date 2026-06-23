# ─────────────────────────────────────────────────────────
# Slide 2 · Embedding Cache — get_or_create(text)
# Embedding-urile aceluiași text sunt DETERMINISTE → calculează
# o dată, refolosește mereu. Cheia = hash MD5 al textului.
# ─────────────────────────────────────────────────────────
import hashlib
from sqlalchemy import String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector

from llm import embed


class Base(DeclarativeBase):
    pass


class EmbeddingCacheEntry(Base):
    __tablename__ = "embedding_cache"
    # PK = hash-ul textului → lookup O(1), text identic = aceeași cheie
    text_hash: Mapped[str] = mapped_column(String(32), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))


class EmbeddingCache:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_or_create(self, text: str) -> list[float]:
        key = self._hash(text)                       # 1. MD5 hash
        with self.session_factory() as db:
            entry = db.get(EmbeddingCacheEntry, key)  # 2. check DB
            if entry is not None:
                return entry.embedding                # HIT  · ~5ms lookup

            vector = embed(text)[0]                  # 3. MISS → generate
            db.add(EmbeddingCacheEntry(text_hash=key, embedding=vector))
            db.commit()
            return vector

# 💡 Beneficiu maxim: chunk-urile RAG — embedding calculat o dată,
#    refolosit perpetuu la fiecare retrieval.
