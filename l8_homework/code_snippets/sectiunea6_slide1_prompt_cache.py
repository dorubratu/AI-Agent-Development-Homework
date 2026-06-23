# ─────────────────────────────────────────────────────────
# Slide 1 · Prompt Caching (Anthropic) — cache_control: ephemeral
# Provider-ul cachează prefixul STATIC (system + document). Request-urile
# următoare plătesc cache_read (0.1x) în loc de input full → până la 90% mai ieftin.
# ─────────────────────────────────────────────────────────
import anthropic

client = anthropic.Anthropic()

# Documentul mare e prefixul static, identic la fiecare request.
# ⚠️ Min. tokeni pentru cache: 1024 (Claude 3.x) / 2048 (Sonnet 4.x).
#    Sub prag → caching NU se activează (fără eroare, doar cache_creation=0).
DOCUMENT = "<text lung de document, ex: contract de 5000 tokeni>"

def ask(question: str):
    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=[
            {"type": "text", "text": "Ești un analist de documente."},
            {
                "type": "text",
                "text": DOCUMENT,
                "cache_control": {"type": "ephemeral"},   # ← marchează prefixul de cache-uit
            },
        ],
        messages=[{"role": "user", "content": question}],   # partea dinamică, necache-uită
    )
    u = resp.usage
    print(f"creation={u.cache_creation_input_tokens}  "  # >0 la primul apel (scrie cache, 1.25x)
          f"read={u.cache_read_input_tokens}  "          # >0 la următoarele (citește cache, 0.1x)
          f"fresh={u.input_tokens}")                      # doar tokenii noi (întrebarea)
    return resp.content[0].text

ask("Care sunt termenii principali?")   # MISS: creation>0, read=0
ask("Există clauze de penalizare?")     # HIT:  creation=0, read>0  → ~90% economie
