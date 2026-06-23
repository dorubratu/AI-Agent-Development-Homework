# ─────────────────────────────────────────────────────────
# Slide 4 · Integrare în Agent
# Pattern: check cache → call LLM (doar la miss) → save în cache.
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from sectiunea5_slide3_cache_manager import SemanticCacheManager


class CachedDocumentAnalyst:
    def __init__(self, cache: SemanticCacheManager):
        self.cache = cache
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    def query(self, user_input: str) -> str:
        # 1. CHECK CACHE — întrebare similară deja răspunsă?
        cached = self.cache.get(user_input)
        if cached is not None:
            return cached                          # HIT · $0 · ~10ms

        # 2. CALL LLM — doar la MISS
        response = self.llm.invoke(user_input).content   # MISS · $$ · ~2s

        # 3. SAVE CACHE — pentru viitoare query-uri similare
        self.cache.set(user_input, response)
        return response


# Comportament tipic:
#   "Ce este VAT?"        → MISS, call LLM       (prima dată)
#   "Ce înseamnă VAT?"    → HIT, sim 94%         ($0)
#   "Explică-mi TVA"      → HIT, sim 91%         ($0)
#   "Capitala Franței?"   → MISS, topic diferit  (call LLM)
