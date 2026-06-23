# ─────────────────────────────────────────────────────────
# Slide 3 · SemanticCacheManager — get() + set()
# get(): embed → cosine search → filtru prag → top-1 sau None
# set(): embed → INSERT
# ─────────────────────────────────────────────────────────
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession, sessionmaker
from pgvector.sqlalchemy import Vector   # noqa: F401  (înregistrează tipul)

from sectiunea5_slide2_cache_schema import SemanticCache
from llm import embed                    # all-MiniLM-L6-v2 → 384 dim, normalizate


class SemanticCacheManager:
    def __init__(self, session_factory: sessionmaker, threshold: float = 0.92):
        self.session_factory = session_factory
        self.threshold = threshold       # similaritate minimă pt. HIT

    def get(self, query: str) -> str | None:
        """Caută cel mai apropiat query cache-uit. Returnează răspunsul sau None."""
        emb = embed(query)[0]
        with self.session_factory() as db:
            # cosine_distance = 1 - cosine_similarity, deci ORDER BY ASC = cel mai similar.
            distance = SemanticCache.query_embedding.cosine_distance(emb)
            row = db.execute(
                select(SemanticCache, distance.label("dist"))
                .order_by(distance)       # folosește indexul HNSW
                .limit(1)
            ).first()

            if row is None:
                return None

            cache, dist = row
            similarity = 1 - dist
            if similarity < self.threshold:
                return None               # cel mai bun candidat e prea diferit → MISS

            cache.hit_count += 1          # contorizăm hit-ul
            db.commit()
            return cache.response

    def set(self, query: str, response: str) -> None:
        """Salvează un query + răspuns nou în cache."""
        emb = embed(query)[0]
        with self.session_factory() as db:
            db.add(SemanticCache(query_text=query, query_embedding=emb, response=response))
            db.commit()
