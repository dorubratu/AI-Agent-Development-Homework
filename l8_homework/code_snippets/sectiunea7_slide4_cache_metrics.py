# ─────────────────────────────────────────────────────────
# Slide 4 · Monitorizare — CacheMetrics
# "Dacă nu măsori, nu știi dacă funcționează."
# Track hits / misses → hit_rate + economie estimată.
# ─────────────────────────────────────────────────────────
from dataclasses import dataclass


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    cost_per_call: float = 0.01     # cost mediu estimat per apel LLM evitat ($)

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def estimated_savings(self) -> float:
        # fiecare HIT = un apel LLM evitat = bani economisiți
        return self.hits * self.cost_per_call

    def report(self) -> dict:
        return {
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
            "hits": self.hits,
            "misses": self.misses,
            "estimated_savings": f"${self.estimated_savings:.2f}",
        }


# Folosire — wrap în jurul oricărui cache:
#   metrics = CacheMetrics()
#   if cache.get(q): metrics.record_hit()
#   else:            metrics.record_miss()
#   print(metrics.report())
#   → {"hit_rate": "42.5%", "hits": 85, "misses": 115, "estimated_savings": "$0.85"}
#
# Target hit rates:  <20% ineficient · 20-40% normal · 40-60% bun (FAQ) · >60% excelent
