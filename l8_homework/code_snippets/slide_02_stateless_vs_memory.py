"""
Slide 2-3 · Demo: "Agentul cu Alzheimer" — stateless vs cu memorie
==================================================================

Punchline-ul lecției, văzut live: același LLM, aceeași întrebare finală.
Diferența nu e în model — e în CE îi trimitem.

    FĂRĂ memorie: fiecare apel pornește de la zero. Agentul uită cine ești.
    CU memorie:   acumulăm istoricul într-o listă și-l retrimitem. Își amintește.

Mesajul cheie (slide 3): LLM-urile sunt stateless by design.
NOI gestionăm memoria, în cod — nu în model.

Rulează:  python slide_02_stateless_vs_memory.py
"""

from llm import chat


# Scenariul de pe slide: Andrei, care lucrează cu facturi.
TURNS = [
    "Mă numesc Andrei și lucrez la facturi.",
    "Extrage datele din PDF-ul ăsta: Furnizor ACME, sumă 1500 RON, dată 2025-01-10.",
    "Cum mă numesc și cu ce mă ocup?",   # <- întrebarea care demască uitarea
]


def run_without_memory():
    """
    Fiecare request = conversație nouă pentru LLM.
    Trimitem DOAR mesajul curent. Nimic din trecut.
    """
    print("=" * 60)
    print("FĂRĂ MEMORIE  (fiecare apel e independent)")
    print("=" * 60)

    for turn in TURNS:
        reply = chat(turn)               # doar mesajul curent, zero context
        print(f"\n🧑 User : {turn}")
        print(f"🤖 Agent: {reply}")

    print("\n⚠️  La ultima întrebare agentul nu are de unde să știe — a uitat tot.\n")


def run_with_memory():
    """
    NOI ținem memoria: o listă de mesaje care crește la fiecare tură.
    Asta e toată 'magia'. Memoria nu e în model, e aici, în 'history'.
    """
    print("=" * 60)
    print("CU MEMORIE  (acumulăm istoricul și-l retrimitem)")
    print("=" * 60)

    history = []                         # <- ASTA e memoria. O listă.

    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        reply = chat(history)            # trimitem TOATĂ conversația
        history.append({"role": "assistant", "content": reply})

        print(f"\n🧑 User : {turn}")
        print(f"🤖 Agent: {reply}")

    print("\n✓ Acum își amintește — pentru că i-am dat contextul înapoi.\n")


if __name__ == "__main__":
    run_without_memory()
    run_with_memory()
