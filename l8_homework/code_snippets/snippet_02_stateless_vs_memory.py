# ─────────────────────────────────────────────────────────
# SNIPPET · Slide 2-3 · Stateless vs Memory
# Versiunea de pus pe slide. Doar esența conceptului.
# ─────────────────────────────────────────────────────────

# ❌ FĂRĂ MEMORIE — fiecare apel e independent, LLM uită tot
reply = chat("Mă numesc Andrei.")
reply = chat("Cum mă numesc?")        # -> "Nu am această informație"


# ✅ CU MEMORIE — ținem istoricul într-o listă și-l retrimitem
history = []                          # ← memoria e o listă, atât

def turn(user_msg):
    history.append({"role": "user", "content": user_msg})
    reply = chat(history)             # trimitem TOATĂ conversația
    history.append({"role": "assistant", "content": reply})
    return reply

turn("Mă numesc Andrei.")
turn("Cum mă numesc?")               # -> "Te numești Andrei."

# Ideea: LLM-ul e stateless. Memoria o gestionăm NOI, în cod.
