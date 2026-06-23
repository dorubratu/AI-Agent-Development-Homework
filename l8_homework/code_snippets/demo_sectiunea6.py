"""
Demo Secțiunea 6 · Prompt Caching (Anthropic) + Embedding Caching
=================================================================

Două demonstrații, ambele cu economia MĂSURATĂ, nu afirmată.

PARTEA A · Prompt Caching (provider-side, Anthropic)
    Trimitem un document mare ca prefix static cu cache_control.
    Apelul 1 SCRIE cache-ul (cache_creation > 0).
    Apelul 2 CITEȘTE cache-ul (cache_read > 0) → input la 0.1x = ~90% mai ieftin.
    Afișăm tokenii reali din usage.

PARTEA B · Embedding Caching (local, determinist)
    Același text → același vector. Calculăm o dată, refolosim.
    Măsurăm timpul: prima dată (calcul) vs a doua (lookup).

⚠️ Prompt caching pe Sonnet 4.x cere prefix de MIN 2048 tokeni, altfel
   nu se activează (cache_creation rămâne 0, fără eroare). Documentul de
   mai jos e dimensionat să treacă pragul.

Rulează (Partea A cere ANTHROPIC_API_KEY):
    python demo_sectiunea6.py
"""

import time
from llm import chat


# ── PARTEA A · Prompt Caching ────────────────────────────────────────
def demo_prompt_caching():
    print("=" * 60)
    print("PARTEA A · PROMPT CACHING (Anthropic)")
    print("=" * 60)

    # Prefix static mare (>2048 tokeni). În realitate = un document/contract.
    # Repetăm un paragraf ca să depășim pragul fără să inventăm 5 pagini.
    document = (
        "Contract de prestări servicii între TechCorp SRL și furnizorul ACME. "
        "Termenii includ: livrare în 30 de zile, penalizare 0.1% pe zi de "
        "întârziere, TVA 19% aplicat la valoarea netă, plata în 15 zile de la "
        "factură. Clauze de confidențialitate și forță majoră aplicabile. "
    ) * 60   # ~ depășește 2048 tokeni

    system_blocks = [
        {"type": "text", "text": "Ești un analist de contracte."},
        {
            "type": "text",
            "text": document,
            "cache_control": {"type": "ephemeral"},   # ← marchează prefixul de cache-uit
        },
    ]

    def ask(question):
        text, usage = chat(
            question,
            system=system_blocks,
            return_usage=True,
        )
        return text, usage

    print("\n→ Apel 1 (cache MISS — scrie cache-ul):")
    _, u1 = ask("Care e penalizarea pentru întârziere?")
    print(f"   cache_creation={u1['cache_creation_input_tokens']}  "
          f"cache_read={u1['cache_read_input_tokens']}  "
          f"fresh_input={u1['input_tokens']}")

    print("\n→ Apel 2 (cache HIT — citește cache-ul):")
    _, u2 = ask("Cât e TVA-ul aplicat?")
    print(f"   cache_creation={u2['cache_creation_input_tokens']}  "
          f"cache_read={u2['cache_read_input_tokens']}  "
          f"fresh_input={u2['input_tokens']}")

    # Economia: tokenii citiți din cache costă 0.1x față de input normal.
    cached = u2["cache_read_input_tokens"]
    if cached:
        full_cost = cached            # dacă i-am plăti la preț întreg
        cached_cost = cached * 0.1    # plătiți efectiv la 0.1x
        saved = (1 - cached_cost / full_cost) * 100
        print(f"\n💰 {cached} tokeni serviți din cache la 0.1x → ~{saved:.0f}% "
              f"reducere pe partea cache-uită a inputului.")
    else:
        print("\n⚠️ cache_read=0 — prefixul a fost sub prag sau cache-ul a expirat (TTL 5 min).")


# ── PARTEA B · Embedding Caching ─────────────────────────────────────
def demo_embedding_caching():
    print("\n" + "=" * 60)
    print("PARTEA B · EMBEDDING CACHING (local, determinist)")
    print("=" * 60)

    from llm import embed
    cache: dict[str, list[float]] = {}

    def get_embedding(text):
        if text in cache:
            return cache[text], True       # HIT
        vec = embed(text)[0]
        cache[text] = vec
        return vec, False                  # MISS

    text = "Care sunt termenii de plată din contract?"

    t0 = time.perf_counter()
    _, hit1 = get_embedding(text)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, hit2 = get_embedding(text)          # exact același text
    t2 = time.perf_counter() - t0

    print(f"\n→ Calcul 1 (MISS): {t1*1000:.1f}ms  (hit={hit1})")
    print(f"→ Calcul 2 (HIT):  {t2*1000:.1f}ms  (hit={hit2})")
    if t2 > 0:
        print(f"\n⚡ Lookup-ul e ~{t1/t2:.0f}× mai rapid decât recalcularea.")


if __name__ == "__main__":
    demo_prompt_caching()
    demo_embedding_caching()
