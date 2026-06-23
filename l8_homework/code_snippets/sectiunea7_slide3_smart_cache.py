# ─────────────────────────────────────────────────────────
# Slide 3 · Cache Invalidation — TTL + topic-based + manual
# Extinde semantic cache-ul (Secțiunea 5) cu logică de expirare.
# "There are only two hard things in CS: cache invalidation
#  and naming things." — Phil Karlton
# ─────────────────────────────────────────────────────────
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, delete, func
from sqlalchemy.orm import sessionmaker

from sectiunea5_slide2_cache_schema import SemanticCache
from llm import embed


class SmartCache:
    def __init__(self, session_factory: sessionmaker, threshold: float = 0.92, ttl: int = 3600):
        self.session_factory = session_factory
        self.threshold = threshold
        self.ttl = ttl                          # time-to-live în secunde (1h)

    def get(self, query: str) -> str | None:
        emb = embed(query)[0]
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl)
        with self.session_factory() as db:
            distance = SemanticCache.query_embedding.cosine_distance(emb)
            row = db.execute(
                select(SemanticCache, distance.label("dist"))
                .where(SemanticCache.created_at >= cutoff)   # ⏱️ doar entries ne-expirate
                .order_by(distance)
                .limit(1)
            ).first()

            if row is None:
                return None
            cache, dist = row
            if (1 - dist) < self.threshold:
                return None
            cache.hit_count += 1
            db.commit()
            return cache.response

    # 🔄 Topic-based: șterge tot ce ține de un subiect (ex: doc sursă schimbat)
    def invalidate_by_topic(self, topic: str) -> int:
        with self.session_factory() as db:
            result = db.execute(
                delete(SemanticCache).where(SemanticCache.query_text.ilike(f"%{topic}%"))
            )
            db.commit()
            return result.rowcount

    # 👆 Manual: user cere explicit "refresh" → golește tot
    def clear(self) -> None:
        with self.session_factory() as db:
            db.query(SemanticCache).delete()
            db.commit()

    # 🧹 Curățenie: șterge proactiv entries expirate (rulat periodic / cron)
    def purge_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl)
        with self.session_factory() as db:
            result = db.execute(delete(SemanticCache).where(SemanticCache.created_at < cutoff))
            db.commit()
            return result.rowcount
