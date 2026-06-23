"""
L8 Task 3 - Comparație: LLM vs scikit-learn Intent Classifier
=============================================================

Compară, pe același set de test, doi clasificatori de intenție:
  1. LLM (Claude Haiku) — clasificare prin prompt
  2. scikit-learn (TF-IDF + LogisticRegression) — model local antrenat

Metrici comparate:
  • latență (medie per query)
  • cost (estimat din tokeni × prețuri Anthropic; clasificatorul local = $0)
  • accuracy (pe setul de test cu etichete cunoscute)

Rulează:
    cd l8_homework/agent
    PYTHONPATH="skillab-py/src:src" python3 src/hw_intent_compare.py
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent / ".env"))

from skillab import get_llm
from intent_data import TEST_DATA, LABELS
import intent_classifier as ic

# Prețuri Claude Haiku 4.5 ($/MTok)
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00

LLM_SYSTEM = (
    "Ești un clasificator de intenție. Clasifici un query într-una din clasele: "
    "search, extract, summarize.\n"
    "- search    = utilizatorul caută / vrea să găsească documente sau informații\n"
    "- extract   = utilizatorul vrea să extragă câmpuri / valori structurate\n"
    "- summarize = utilizatorul vrea un rezumat / sinteză\n"
    "Răspunzi DOAR cu un singur cuvânt: search, extract sau summarize."
)


def llm_classify(llm, query: str) -> tuple[str, dict, float]:
    """Clasifică un query cu LLM. Întoarce (label, usage, latency)."""
    messages = [
        {"role": "system", "content": LLM_SYSTEM},
        {"role": "user", "content": f"Query: {query}\nClasă:"},
    ]
    t0 = time.perf_counter()
    text, usage = llm.generate_with_usage(messages)
    latency = time.perf_counter() - t0

    # Normalizează răspunsul la una din clase.
    label = "search"
    low = text.strip().lower()
    for lbl in LABELS:
        if lbl in low:
            label = lbl
            break
    return label, usage, latency


def llm_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) * PRICE_INPUT_PER_MTOK
        + usage.get("output_tokens", 0) * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000


def run_llm(llm) -> dict:
    correct = 0
    total_latency = 0.0
    total_cost = 0.0
    rows = []
    for q, expected in TEST_DATA:
        pred, usage, lat = llm_classify(llm, q)
        ok = pred == expected
        correct += ok
        total_latency += lat
        total_cost += llm_cost(usage)
        rows.append((q, expected, pred, ok, lat))
    n = len(TEST_DATA)
    return {
        "accuracy": correct / n,
        "avg_latency": total_latency / n,
        "total_cost": total_cost,
        "rows": rows,
    }


def run_sklearn(pipe) -> dict:
    correct = 0
    total_latency = 0.0
    rows = []
    for q, expected in TEST_DATA:
        t0 = time.perf_counter()
        pred = ic.predict(pipe, q)
        lat = time.perf_counter() - t0
        ok = pred == expected
        correct += ok
        total_latency += lat
        rows.append((q, expected, pred, ok, lat))
    n = len(TEST_DATA)
    return {
        "accuracy": correct / n,
        "avg_latency": total_latency / n,
        "total_cost": 0.0,  # rulează local
        "rows": rows,
    }


def main():
    n = len(TEST_DATA)
    print(f"Set de test: {n} queries\n")

    # scikit-learn (antrenează dacă nu există modelul salvat)
    pipe = ic.load()
    sk = run_sklearn(pipe)

    # LLM
    llm = get_llm(provider="anthropic")
    print(f"LLM: {llm.model} (rulez {n} clasificări...)\n")
    llm_res = run_llm(llm)

    # --- detaliu erori LLM (clasificatorul local e adesea perfect pe acest set) ---
    llm_errors = [(q, exp, pred) for q, exp, pred, ok, _ in llm_res["rows"] if not ok]
    sk_errors = [(q, exp, pred) for q, exp, pred, ok, _ in sk["rows"] if not ok]

    print("=" * 74)
    print(f"{'Metric':<22}{'LLM (Claude Haiku)':<26}{'scikit-learn (local)':<26}")
    print("=" * 74)
    print(f"{'Accuracy':<22}{llm_res['accuracy']*100:>6.1f}%{'':<19}{sk['accuracy']*100:>6.1f}%")
    print(f"{'Latență medie/query':<22}{llm_res['avg_latency']*1000:>8.1f} ms{'':<15}{sk['avg_latency']*1000:>8.3f} ms")
    print(f"{'Cost total':<22}${llm_res['total_cost']:<25.6f}${sk['total_cost']:.6f}  (local, $0)")
    speedup = (llm_res['avg_latency'] / sk['avg_latency']) if sk['avg_latency'] else float('inf')
    print("-" * 74)
    print(f"  → scikit-learn e ~{speedup:,.0f}× mai rapid și costă $0.")

    if llm_errors:
        print(f"\n  Erori LLM ({len(llm_errors)}):")
        for q, exp, pred in llm_errors:
            print(f"    {q[:40]:42} aștept={exp:10} prezis={pred}")
    if sk_errors:
        print(f"\n  Erori scikit-learn ({len(sk_errors)}):")
        for q, exp, pred in sk_errors:
            print(f"    {q[:40]:42} aștept={exp:10} prezis={pred}")

    print("\nConcluzie:")
    print("  • Pentru intenții fixe și bine definite (search/extract/summarize),")
    print("    un clasificator local TF-IDF + LogisticRegression e ideal: rapid,")
    print("    gratuit, predictibil. Folosește-l ca router de intenție înainte de LLM.")
    print("  • LLM-ul rămâne util pentru clase noi/ambigue sau zero-shot, dar")
    print("    adaugă latență (sute de ms–secunde) și cost per request.")


if __name__ == "__main__":
    main()
