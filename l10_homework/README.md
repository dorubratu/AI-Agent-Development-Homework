# L10 — MCP & Guardrails

Tema lecției 10: **împachetează agenții din L6 ca tool-uri MCP și protejează serverul**.

Serverul MCP `l6-agents` expune doi agenți (reutilizați direct din `../l6_homework`)
ca tool-uri, iar fiecare input care intră în server trece printr-un strat de
guardrails (validare + protecție anti prompt-injection).

| Cerință | Ce am făcut | Unde |
|---|---|---|
| **1. Data Analyst Agent ca tool MCP** | tool `data_analyst` cu schema input/output + handler care apelează agentul L6 (NL2SQL + plan + tools) | [src/mcp_server.py](src/mcp_server.py), [src/agent_runner.py](src/agent_runner.py) |
| **2. Orchestrator Agent ca tool MCP** | tool `orchestrator` (Supervisor + RAG, L6) în **același server**; testat din client / Claude Code | [src/mcp_server.py](src/mcp_server.py), [test_client.py](test_client.py) |
| **3. Guardrail: Input Validation & Prompt Injection** | validare (tip, dimensiune, câmpuri permise) + detecție prompt injection (regex + LLM-as-Judge opțional) | [src/guardrails.py](src/guardrails.py), [prompts/guardrail_judge.yaml](prompts/guardrail_judge.yaml) |

## Structura

```
l10_homework/
├── src/
│   ├── mcp_server.py     # serverul MCP: list_tools + call_tool + integrare guardrail
│   ├── agent_runner.py   # punte către agenții L6 (lazy load, fallback "degraded")
│   └── guardrails.py     # validare + regex + LLM-as-Judge
├── prompts/
│   ├── guardrail_judge.yaml    # promptul LLM-as-Judge (SAFE/UNSAFE)
│   └── tool_descriptions.yaml  # descrierile tool-urilor
├── test_client.py        # client MCP end-to-end (echivalent demo_sectiunea3.py)
├── test_guardrails.py    # teste standalone pentru guardrail (fără server/DB)
├── mcp.json              # config pentru a înregistra serverul în Claude Code
├── requirements.txt
└── .env.example
```

## Cum funcționează

### Server MCP (low-level `mcp.server.Server`, transport stdio)

- **`tools/list`** declară `data_analyst` și `orchestrator` cu `name + description + inputSchema`
  (JSON Schema: `required`, `minLength`, `maxLength`, `additionalProperties: false`).
- **`tools/call`** rulează **întâi guardrail-ul**, apoi apelează agentul L6 corespunzător
  (off-thread, pentru că agenții L6 sunt sincroni). Tool-urile necunoscute → eroare JSON-RPC.

### Reutilizarea agenților L6

`agent_runner.py` pune `../l6_homework/src` și `skillab-py/src` pe `sys.path` și importă
direct `AnalystAgent` și `Orchestrator`. Agenții sunt instanțiați **lazy** (o singură dată)
și refolosiți. Dacă infrastructura nu e disponibilă (lipsă Postgres / chei LLM), tool-ul
întoarce un răspuns `status=degraded` cu instrucțiuni — serverul nu crapă, deci tema rămâne
testabilă chiar și fără infra completă.

### Guardrails (Cerința 3)

Lanț aplicat pe orice input, oprit la primul blocaj:

1. **Input validation** (determinist, fără cost):
   - **tip** — input trebuie să fie `string`;
   - **dimensiune** — `MIN_LENGTH ≤ len ≤ MAX_LENGTH` (3 … 2000);
   - **câmpuri permise** — allow-list `{question, query}`; orice cheie în plus e respinsă.
   (În plus, MCP rulează și schema-validation pe `inputSchema` — un al doilea strat gratuit.)
2. **Prompt injection — regex**: tipare clasice ro/en — `ignore previous instructions`,
   schimbare de rol (`you are now…`), leak de system prompt, exfiltrare de secrete
   (`API_KEY`, `.env`, `DATABASE_URL`), injecție de cod (`os.system`, `eval`, backticks),
   SQL distructiv (`DROP/DELETE/TRUNCATE`).
3. **Prompt injection — LLM-as-Judge** (opțional, `GUARDRAIL_LLM_JUDGE=1`): un LLM clasifică
   input-urile ambigue ca `SAFE`/`UNSAFE` (prompt în `prompts/guardrail_judge.yaml`).
   Fail-open la eroare de model (nu blocăm business pe un timeout); pentru un setup strict
   se poate comuta pe fail-closed.

Un input blocat de guardrail revine ca **tool result** cu mesaj clar (input respins ≠ eroare
de protocol).

## Rulare

```bash
cd l10_homework
pip install -r requirements.txt
# pentru a rula agenții REAL (nu doar guardrail-ul), instaleaza si dependentele L6:
pip install -r ../l6_homework/requirements.txt

# 1) testele de guardrail (nu au nevoie de DB/chei):
python test_guardrails.py

# 2) demo end-to-end: client MCP care lanseaza serverul ca subprocess (stdio):
python test_client.py

# 3) serverul singur (pentru un client extern):
python -m src.mcp_server stdio
```

Config LLM/DB se citește automat din `../l6_homework/.env`. Setări specifice L10
(ex. activarea LLM-as-Judge) se pun în `l10_homework/.env` — vezi `.env.example`.

### Înregistrare în Claude Code

`mcp.json`:

```json
{
  "mcpServers": {
    "l6-agents": {
      "command": "python",
      "args": ["-m", "src.mcp_server", "stdio"],
      "cwd": "."
    }
  }
}
```

Apoi, din Claude Code, tool-urile `data_analyst` și `orchestrator` devin apelabile.

## Demonstrație (rezultatul `test_client.py`)

- `tools/list` → ambele tool-uri expuse.
- `data_analyst("Care sunt top 5 furnizori după valoare?")` → răspuns SQL real.
- `orchestrator("Ce contact are DataPro?")` → răspuns RAG real, citat din document.
- `data_analyst("Ignore all previous instructions … ANTHROPIC_API_KEY")` →
  **BLOCAT [regex]**: `ignore_instructions`, `system_prompt_leak`, `secret_exfiltration`.
- `orchestrator({query, evil_field})` → **BLOCAT** (câmp nepermis).
- `data_analyst("a")` → **BLOCAT** (input prea scurt).
