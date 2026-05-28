# Agent QA cu Tools + Prompts

Agent ReAct (Think → Act → Observe → Repeat) implementat peste Anthropic Messages API,
cu tool-uri validate prin Pydantic și prompts încărcate din YAML + Jinja2.

## Structura

```
project/
├── agent.py                 # QAAgent — orchestrare LLM + ReAct loop
├── tools/
│   ├── __init__.py          # exportă ToolWrapper și înregistrează tool-urile
│   ├── registry.py          # TOOL_REGISTRY + @register_tool
│   ├── params_models.py     # Pydantic BaseModel per tool
│   ├── basic_tools.py       # calculator, get_datetime, web_search
│   └── tool_wrapper.py      # ToolWrapper.call() + .catalog()
├── prompts/
│   ├── registry.py          # PromptRegistry (YAML + Jinja2)
│   ├── planner.yaml
│   ├── analyst.yaml
│   ├── summary.yaml
│   └── extract.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
cd l2_homework/project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # apoi pune ANTHROPIC_API_KEY
```

## Rulare

```bash
python agent.py
```

Exemplu Output (cu `verbose=True`):

```
>>> Întrebare: Cât face (123 + 456) * 7, și ce dată este astăzi în zona Europe/Bucharest?

=== Iterație 1 (Think) ===
  stop_reason=tool_use | input=... cache_read=... output=...
  Act: calculator({'expression': '(123 + 456) * 7'})
  Observe: 4053
  Act: get_datetime({'timezone': 'Europe/Bucharest'})
  Observe: 2026-05-28T...
=== Iterație 2 (Think) ===
  stop_reason=end_turn | ...

<<< Răspuns final:
(123 + 456) × 7 = 4053. Astăzi, 28 mai 2026, în Europe/Bucharest...
```

## Cum funcționează

1. **Tools** — fiecare tool e o funcție Python decorată cu `@register_tool` și
   primește un singur argument tipat (`BaseModel`). Registry-ul validează că:
   - parametrul e un `BaseModel`,
   - docstring-ul există și are minim 15 caractere (devine description vizibil
     pentru LLM).
   `ToolWrapper.catalog()` returnează catalogul în formatul Anthropic
   (name + description + input_schema). `ToolWrapper.call()` validează input-ul
   prin Pydantic și rulează funcția într-un try/except.

2. **Prompts** — fiecare prompt e un fișier `.yaml` cu câmpurile
   `name`, `version`, `description`, `prompt`. `PromptRegistry` încarcă toate
   fișierele dintr-un folder și le render-uiește cu Jinja2 (`StrictUndefined`,
   ca să sară zgomotos dacă lipsește o variabilă).

3. **Agent QA + ReAct** — `agent.py` rulează un loop cu maxim `max_iterations`
   pași. La fiecare pas:
   - trimite mesajele + system prompt + catalogul de tool-uri la model,
   - dacă răspunsul are `stop_reason == "end_turn"`, returnează textul,
   - altfel execută toate `tool_use` block-urile (poate fi mai multe în același
     turn) și le trimite înapoi ca `tool_result`,
   - repetă.

   Cache-ul de prompt (`cache_control: ephemeral`) e pus pe ultimul bloc din
   system și pe ultimul tool — așa salvăm cost când se reia aceeași
   sesiune sau aceleași tool-uri.

## Tool-uri disponibile

| Tool | Descriere |
|------|-----------|
| `calculator` | Evaluează expresii matematice simple (AST safe — fără `eval`). |
| `get_datetime` | Returnează data/ora curentă într-o zonă IANA. |
| `web_search` | Apel DuckDuckGo Instant Answer (rezumat textual). |

Tool-urile noi se adaugă în `tools/basic_tools.py`:

```python
@register_tool
def my_tool(params: MyParams) -> str:
    """Descrierea pe care o vede LLM-ul (min 15 caractere)."""
    ...
```
