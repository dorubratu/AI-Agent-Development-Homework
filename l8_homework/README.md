# Lecția 8 — Temă: Memory, Caching & Intent Classifier

Optimizarea agentului din L6 (Orchestrator + RAG) cu memorie persistentă, prompt
caching și un router de intenție pe scikit-learn.

Proiectul L6 a fost copiat în [`agent/`](agent/) și modificat direct. Tot codul
nou e marcat în comentarii cu `L8`.

## Structură

```
l8_homework/
├── agent/                     # proiectul L6 copiat + modificările L8
│   ├── src/
│   │   ├── memory.py              # L8 Task 1 — PersistentMemory (manager)
│   │   ├── models.py              # + ChatSession, ChatMessage
│   │   ├── repositories.py        # + ChatMessageRepository
│   │   ├── state.py               # OrchestratorState + session_id, history
│   │   ├── orchestrator.py        # noduri load_memory / save_memory + prompt caching
│   │   ├── intent_data.py         # L8 Task 3 — date de antrenare (train/test)
│   │   ├── intent_classifier.py   # L8 Task 3 — TF-IDF + LogisticRegression
│   │   ├── hw_prompt_cache_demo.py# L8 Task 2 — demo & măsurători caching
│   │   └── hw_intent_compare.py   # L8 Task 3 — comparație LLM vs sklearn
│   ├── skillab-py/src/skillab/llm/
│   │   ├── base.py                # + generate_with_usage / supports_prompt_caching
│   │   └── providers/anthropic.py # + prompt caching (cache_control: ephemeral)
│   ├── alembic/versions/004_create_chat_memory.py   # tabele memorie
│   └── models/intent_clf.joblib   # modelul antrenat (generat)
└── code_snippets/             # snippet-urile din lecție (referință)
```

## Setup

```bash
cd agent
# 1. Pornește PostgreSQL (pgvector) pe portul 5433
docker compose up -d
# 2. Aplică migrările (creează și tabelele de memorie)
PYTHONPATH="skillab-py/src:src" python3 -m alembic upgrade head
```

`.env` folosește deja Anthropic (`claude-haiku-4-5`). DB-ul are 135 chunks seed-uite.

---

## Task 1 — Conversation Memory (PostgreSQL)

**Cerință:** adaugă ConversationMemory la agent, integrează în nodul LangGraph
(context persistent între request-uri), persistă în PostgreSQL.

**Implementare:**
- `models.py` — tabelele `chat_sessions` (1) → `chat_messages` (N), per `session_id`.
- `repositories.py` — `ChatMessageRepository` (`add`, `latest` cu window, `delete_session`).
- `memory.py` — `PersistentMemory`: `load_messages()` (ultimele N mesaje, cronologic)
  și `save_message()` / `save_turn()` (tranzacție atomică).
- `orchestrator.py` — două noduri noi în graful LangGraph:
  ```
  START → load_memory → call_rag → evaluate ──┬─→ answer → save_memory → END
  ```
  `load_memory` încarcă istoricul din DB; `node_answer` îl injectează ca mesaje
  user/assistant; `save_memory` persistă turul. Memoria supraviețuiește
  restart-urilor (trăiește în DB, nu în RAM).

Strategie: tip `ConversationBufferWindowMemory` — reinjectăm doar ultimele N
mesaje (`memory_window=10`).

Verificat: două ture în aceeași sesiune → 4 mesaje persistate, tura 2 are acces
la contextul turei 1.

---

## Task 2 — Prompt Caching (Anthropic)

**Cerință:** activează prompt caching (`cache_control: ephemeral`), pune system
prompt-ul + contextul fix în cache, măsoară tokenii economisiți și latența.

**Implementare:**
- `anthropic.py` — `generate_with_usage(messages, cache_system=True)` construiește
  `system` ca bloc cu `cache_control: {"type": "ephemeral"}` și întoarce usage cu
  `cache_creation_input_tokens` / `cache_read_input_tokens`.
- `orchestrator.py` — `node_answer` trimite system prompt-ul static + contextul
  documentelor marcat ca prefix cacheabil; statisticile se cumulează în
  `orch.cache_stats`.

> ⚠️ Min. cache la Claude Haiku 4.5 = **4096 tokeni**. Sub prag caching-ul nu se
> activează (fără eroare, doar `cache_creation/read = 0`). Pe query-uri mici din
> orchestrator pragul nu e atins; demo-ul de mai jos folosește un document mare
> ca să arate efectul real.

**Demo & măsurători:**
```bash
PYTHONPATH="skillab-py/src:src" python3 src/hw_prompt_cache_demo.py
```
Rezultat tipic (4 întrebări pe același document de ~16K tokeni):

| Metric            | Fără cache | Cu cache              |
|-------------------|-----------:|-----------------------:|
| Tokeni input full | 64.263     | 15 fresh + 48.150 din cache |
| Cost total        | $0.066     | $0.027 (**−59%**)      |

Primul request scrie cache-ul (MISS, ~1.25×), următoarele 3 citesc din cache
(HIT, ~0.1×). Pe mai multe request-uri, reducerea se apropie de 90% pe input.

---

## Task 3 — Intent Classifier (scikit-learn)

**Cerință:** antrenează TF-IDF + LogisticRegression pe `query + label`
(search/extract/summarize); compară într-un script latența, costul și accuracy
între LLM și clasificator.

**Implementare:**
- `intent_data.py` — 45 exemple train + 15 test (română, domeniul documente).
- `intent_classifier.py` — pipeline `TfidfVectorizer(1,2-grame) → LogisticRegression`,
  cu `train()` / `load()` / `predict()` / `evaluate()`.

```bash
# Antrenează + salvează modelul (models/intent_clf.joblib)
PYTHONPATH="skillab-py/src:src" python3 src/intent_classifier.py
# Compară LLM vs sklearn
PYTHONPATH="skillab-py/src:src" python3 src/hw_intent_compare.py
```

Rezultat tipic (15 queries de test):

| Metric              | LLM (Claude Haiku) | scikit-learn (local) |
|---------------------|-------------------:|---------------------:|
| Accuracy            | 100.0%             | 100.0%               |
| Latență medie/query | ~1480 ms           | ~0.4 ms              |
| Cost total          | ~$0.0025           | $0 (local)           |

→ Pe intenții fixe, clasificatorul local e **~3.800× mai rapid** și **gratuit**.
Concluzie: folosește-l ca router de intenție înainte de LLM; LLM-ul rămâne util
pentru clase noi/ambigue sau zero-shot.
```
