# Tema L8 — Cum rulez testele 

Am pornit de la agentul din Lecția 6 (Orchestrator + RAG, LangGraph) — copiat în
`l8_homework/agent/` — și l-am optimizat cu cele 3 cerințe din temă. Tot codul nou
e marcat în comentarii cu `L8`. Toate testele rulează **cu apeluri reale** la
Anthropic (`claude-haiku-4-5`) și pe baza de date PostgreSQL (135 chunks).

| # | Cerință | Ce am făcut | Rezultat măsurat |
|---|---------|-------------|------------------|
| 1 | **Conversation Memory** | Memorie persistată în PostgreSQL (tabelele `chat_sessions` + `chat_messages`), manager `PersistentMemory`, integrată ca noduri `load_memory` / `save_memory` în graful LangGraph. | Memoria supraviețuiește restart-ului (instanță nouă încarcă istoricul din DB). |
| 2 | **Prompt Caching** | `cache_control: ephemeral` pe prefixul static (system prompt + document) în provider-ul Anthropic; măsor tokeni/cost/latență. | Cache cald: **~87% reducere cost** pe input (64.200 tokeni serviți din cache). |
| 3 | **Intent Classifier** | TF-IDF + LogisticRegression (search/extract/summarize), antrenat pe 45 exemple; script de comparație cu LLM. | Accuracy **100%**, **~5.000× mai rapid** decât LLM, cost **$0**. |

---

## Pasul 0 — Setup (o singură dată)

```bash
cd l8_homework/agent

# 1. Pornește PostgreSQL (pgvector) pe portul 5433
docker compose up -d

# 2. Aplică migrările (creează și tabelele de memorie chat_sessions / chat_messages)
PYTHONPATH="skillab-py/src:src" python3 -m alembic upgrade head
```

Fișierul `.env` e deja configurat cu Anthropic (`claude-haiku-4-5`). Baza de date
are deja cele 135 de chunks seed-uite din L6.

> Toate comenzile de mai jos se rulează din directorul `l8_homework/agent`.
> Prefixul `PYTHONPATH="skillab-py/src:src"` e necesar ca să găsească biblioteca
> `skillab` și modulele din `src/`.

---

## Task 1 — Conversation Memory

Memoria e integrată direct în agent (nodurile `load_memory` / `save_memory` din
graful LangGraph). Cel mai simplu test arată că memoria persistă între request-uri
și supraviețuiește unui "restart" (o instanță nouă încarcă din DB):

```bash
PYTHONPATH="skillab-py/src:src" python3 - <<'PY'
import sys; sys.path.insert(0,'skillab-py/src'); sys.path.insert(0,'src')
import os; from dotenv import load_dotenv; load_dotenv(".env")
from skillab import get_llm
from state import OrchestratorState
from orchestrator import Orchestrator
from memory import PersistentMemory

llm = get_llm(provider="anthropic")
mem = PersistentMemory(window=10); sess = "demo-andrei"; mem.clear(sess)
app = Orchestrator(llm=llm, memory=mem).build_graph()

app.invoke(OrchestratorState(query="Ce contact are DataPro?", session_id=sess))
print("Dupa tura 1, mesaje in DB:", mem.history_count(sess))

# Instanta noua = simuleaza restart aplicatie
mem2 = PersistentMemory(window=10)
print("Istoric incarcat din DB:", [m["role"] for m in mem2.load_messages(sess)])
mem2.clear(sess)
PY
```

Ce vezi: după prima tură există 2 mesaje în DB; o instanță nouă de
`PersistentMemory` regăsește istoricul → memoria trăiește în PostgreSQL, nu în RAM.

---

## Task 2 — Prompt Caching (Anthropic)

```bash
PYTHONPATH="skillab-py/src:src" python3 src/hw_prompt_cache_demo.py
```

Ce vezi: același document mare (~16K tokeni) e interogat de 4 ori. **Fără cache**
fiecare request plătește tot prefixul ca input. **Cu cache** (`cache_control:
ephemeral`) prefixul e servit din cache (`cache_read`), iar costul de input scade
cu ~60–90% (în funcție de cât de "cald" e cache-ul).

> Notă: pragul minim de cache la Claude Haiku 4.5 e **4096 tokeni**. De aceea
> demo-ul folosește un document mare — pe query-uri mici, caching-ul nu se
> activează (fără eroare, doar `cache_creation/read = 0`).

---

## Task 3 — Intent Classifier (scikit-learn vs LLM)

```bash
# Antrenează + salvează modelul (models/intent_clf.joblib)
PYTHONPATH="skillab-py/src:src" python3 src/intent_classifier.py

# Compară LLM vs scikit-learn pe latență, cost și accuracy
PYTHONPATH="skillab-py/src:src" python3 src/hw_intent_compare.py
```

Ce vezi: clasificatorul local TF-IDF + LogisticRegression prezice intenția
(search/extract/summarize) cu accuracy 100% pe setul de test, de mii de ori mai
rapid decât LLM-ul și gratuit.

---

## Rezultatele complete

Output-ul complet al unei rulări e salvat în
[`RUN_OUTPUT.txt`](RUN_OUTPUT.txt). Detalii de implementare (ce fișier face ce)
sunt în [`README.md`](README.md).
