"""
L8 Task 2 - Prompt Caching (Anthropic) — Demo & Măsurători
==========================================================

Demonstrează prompt caching cu cache_control: ephemeral și măsoară:
  • tokenii economisiți (cache_read vs input full)
  • reducerea de latență (request cu cache HIT vs MISS)
  • reducerea de cost (estimată din prețurile Anthropic)

Ideea: prefixul STATIC (system prompt + document fix) e identic la fiecare
request. Marcat cu cache_control: ephemeral, primul request scrie cache-ul
(~1.25x cost), iar următoarele citesc din cache (~0.1x cost) → până la 90%
reducere pe tokenii de input.

‼️ Min. tokeni pentru cache la Claude Haiku 4.5 = 4096. Sub prag, caching-ul NU
   se activează (fără eroare, doar cache_creation/read = 0). De aceea folosim un
   document mare ca prefix.

Rulează:
    cd l8_homework/agent
    PYTHONPATH="skillab-py/src:src" python3 src/hw_prompt_cache_demo.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent / ".env"))

from skillab import get_llm

# Prețuri Claude Haiku 4.5 ($/MTok). cache_read ~0.1x input, cache_write ~1.25x.
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00
PRICE_CACHE_READ_PER_MTOK = 0.10   # ~0.1x input
PRICE_CACHE_WRITE_PER_MTOK = 1.25  # ~1.25x input


def build_static_system() -> str:
    """Un system prompt + document fix, suficient de mare ca să depășească pragul de 4096 tokeni."""
    instructions = (
        "Ești un analist de documente pentru achiziții publice. Răspunzi la "
        "întrebări folosind exclusiv documentul de mai jos. Răspuns concis.\n\n"
    )
    # Document mare, identic la fiecare request → prefixul cacheabil.
    document = (
        "CONTRACT-CADRU DE FURNIZARE SERVICII SOFTWARE\n"
        "Clauze, definiții, obligații, termene de plată, penalități, SLA. "
    ) * 400
    return instructions + "DOCUMENT:\n" + document


def cost_usd(usage: dict) -> float:
    """Cost estimat al unui apel, ținând cont de tokenii de cache."""
    return (
        usage.get("input_tokens", 0) * PRICE_INPUT_PER_MTOK
        + usage.get("output_tokens", 0) * PRICE_OUTPUT_PER_MTOK
        + usage.get("cache_read_input_tokens", 0) * PRICE_CACHE_READ_PER_MTOK
        + usage.get("cache_creation_input_tokens", 0) * PRICE_CACHE_WRITE_PER_MTOK
    ) / 1_000_000


def run_one(llm, system: str, question: str, cache: bool) -> tuple[dict, float]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    t0 = time.perf_counter()
    _, usage = llm.generate_with_usage(messages, cache_system=cache)
    latency = time.perf_counter() - t0
    return usage, latency


def main():
    llm = get_llm(provider="anthropic")
    if not llm.supports_prompt_caching():
        print("Provider-ul curent nu suportă prompt caching. Setează LLM_PROVIDER=anthropic.")
        return

    print(f"Model: {llm.model}\n")
    system = build_static_system()

    questions = [
        "Care e subiectul contractului?",
        "Ce penalități sunt prevăzute?",
        "Care sunt termenele de plată?",
        "Ce înseamnă SLA în acest contract?",
    ]

    # ---------------------------------------------------------------
    # FĂRĂ CACHE: fiecare request plătește tot prefixul ca input full.
    # ---------------------------------------------------------------
    print("=" * 70)
    print("FĂRĂ PROMPT CACHE (fiecare request plătește prefixul integral)")
    print("=" * 70)
    no_cache_cost = 0.0
    no_cache_latency = 0.0
    no_cache_input = 0
    for q in questions:
        u, lat = run_one(llm, system, q, cache=False)
        no_cache_cost += cost_usd(u)
        no_cache_latency += lat
        no_cache_input += u["input_tokens"]
        print(f"  q={q[:30]:32} input={u['input_tokens']:6} "
              f"read={u['cache_read_input_tokens']:6} lat={lat:.2f}s")

    # ---------------------------------------------------------------
    # CU CACHE: primul request scrie cache-ul, restul citesc din el.
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CU PROMPT CACHE (cache_control: ephemeral pe prefixul static)")
    print("=" * 70)
    cache_cost = 0.0
    cache_latency = 0.0
    cache_read_total = 0
    for i, q in enumerate(questions):
        u, lat = run_one(llm, system, q, cache=True)
        cache_cost += cost_usd(u)
        cache_latency += lat
        cache_read_total += u["cache_read_input_tokens"]
        tag = "MISS (scrie cache)" if u["cache_creation_input_tokens"] else "HIT  (citește cache)"
        print(f"  q={q[:30]:32} input={u['input_tokens']:6} "
              f"create={u['cache_creation_input_tokens']:6} "
              f"read={u['cache_read_input_tokens']:6} lat={lat:.2f}s  {tag}")

    # ---------------------------------------------------------------
    # SUMAR
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMAR")
    print("=" * 70)
    print(f"  Tokeni input full (fără cache) : {no_cache_input}")
    print(f"  Tokeni serviți din cache       : {cache_read_total}")
    print(f"  Cost fără cache  : ${no_cache_cost:.6f}")
    print(f"  Cost cu cache    : ${cache_cost:.6f}")
    if no_cache_cost:
        print(f"  Reducere cost    : {(1 - cache_cost / no_cache_cost) * 100:.1f}%")
    print(f"  Latență totală fără cache : {no_cache_latency:.2f}s")
    print(f"  Latență totală cu cache   : {cache_latency:.2f}s")
    if no_cache_latency:
        print(f"  Reducere latență          : {(1 - cache_latency / no_cache_latency) * 100:.1f}%")


if __name__ == "__main__":
    main()
