"""
Demo Secțiunea 7 · Cache Strategies & Trade-offs
================================================

Punem cap la cap tot ce am construit: semantic cache (Sec. 5) + TTL +
invalidare + metrici (Sec. 7). Demonstrăm ciclul operațional complet
al unui cache de producție.

    1. Câteva query-uri → hit-uri și miss-uri, contorizate
    2. invalidate_by_topic("VAT") → documentul sursă s-a schimbat
    3. Același query → acum MISS (a fost invalidat)
    4. metrics.report() → hit rate + economie estimată

Demonstrează "Regula celor 3R" în practică și răspunde la întrebarea
cheie din slide: cache + TTL = control pe stale vs cost vs latență.

Cerințe: Postgres + pgvector (ca la Secțiunea 5).
    export DATABASE_URL="postgresql://skillab:skillab_dev@localhost:5432/skillab"
Rulează:
    python demo_sectiunea7.py
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from sectiunea5_slide2_cache_schema import Base, SemanticCache
from sectiunea7_slide3_smart_cache import SmartCache
from sectiunea7_slide4_cache_metrics import CacheMetrics
from llm import chat


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://skillab:skillab_dev@localhost:5432/skillab",
)


def setup():
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:                       # demo curat
        db.query(SemanticCache).delete()
        db.commit()
    return factory


def answer(cache: SmartCache, metrics: CacheMetrics, query: str) -> str:
    cached = cache.get(query)
    if cached is not None:
        metrics.record_hit()
        return cached
    metrics.record_miss()
    response = chat(query)
    # set: refolosim logica de scriere din cache (embed + insert)
    from llm import embed
    with cache.session_factory() as db:
        db.add(SemanticCache(query_text=query, query_embedding=embed(query)[0], response=response))
        db.commit()
    return response


if __name__ == "__main__":
    factory = setup()
    cache = SmartCache(factory, threshold=0.70, ttl=3600)  # 0.70 pt all-MiniLM + română
    metrics = CacheMetrics(cost_per_call=0.01)

    # ── 1. Trafic normal: mix de query-uri, unele similare ──────────
    print("=" * 60)
    print("1 · Trafic normal")
    print("=" * 60)
    queries = [
        "Ce este VAT?",          # MISS
        "Ce înseamnă VAT?",      # HIT (parafrază)
        "Explică-mi TVA",        # HIT
        "Capitala Franței?",     # MISS (topic nou)
        "Ce e VAT-ul?",          # HIT
    ]
    for q in queries:
        before = metrics.hits
        answer(cache, metrics, q)
        tag = "HIT " if metrics.hits > before else "MISS"
        print(f"  [{tag}] {q}")

    # ── 2. Documentul sursă despre VAT s-a schimbat → invalidăm ─────
    print("\n" + "=" * 60)
    print("2 · invalidate_by_topic('VAT')  (doc sursă schimbat)")
    print("=" * 60)
    removed = cache.invalidate_by_topic("VAT")
    print(f"  🔄 {removed} entries șterse")

    # ── 3. Același query acum dă MISS (a fost invalidat) ────────────
    print("\n" + "=" * 60)
    print("3 · Re-query după invalidare")
    print("=" * 60)
    before = metrics.hits
    answer(cache, metrics, "Ce înseamnă VAT?")
    tag = "HIT " if metrics.hits > before else "MISS"
    print(f"  [{tag}] Ce înseamnă VAT?   ← acum recalculat (fresh)")

    # ── 4. Raport final ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4 · metrics.report()")
    print("=" * 60)
    import json
    print(json.dumps(metrics.report(), indent=2))

    hr = metrics.hit_rate * 100
    verdict = ("excelent ⭐" if hr > 60 else "bun" if hr > 40
               else "normal" if hr > 20 else "ineficient — ajustează threshold")
    print(f"\n  Hit rate {hr:.0f}% → {verdict}")
