"""
mcp_server.py — Server MCP care expune agenții L6 ca tool-uri, protejat de guardrails.

Acoperă toate cele 3 cerințe ale temei:
  1. Data Analyst Agent ca tool MCP   -> tool `data_analyst`
  2. Orchestrator Agent ca tool MCP   -> tool `orchestrator`  (același server)
  3. Guardrail pe TOT ce intră în server (validare + anti prompt-injection)

Arhitectura urmează slide-urile Secțiunii 3 (mcp_adapter.py / sectiunea3_slide8.py):
  - list_tools()  declară name + description + inputSchema (JSON Schema)
  - call_tool()   rulează guardrail-ul ÎNAINTE de a apela agentul; excepțiile
                  (inclusiv GuardrailError) devin automat erori JSON-RPC

Rulare:
  python -m src.mcp_server stdio     # pentru Claude Code / Claude Desktop / client
  python -m src.mcp_server           # (alias) — implicit stdio
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

# Permite atât `python -m src.mcp_server` cât și `python src/mcp_server.py`.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.guardrails import Guardrail, GuardrailError
    from src.agent_runner import AgentRunner
except ImportError:  # rulat direct din folderul src/
    from guardrails import Guardrail, GuardrailError
    from agent_runner import AgentRunner

# Logging DOAR pe stderr — stdout e rezervat protocolului JSON-RPC (Slide 3).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("l10.mcp_server")


# ── 1. SERVER ──────────────────────────────────────────────────────────────
server = Server("l6-agents")

# Agenții L6 (lazy) + guardrail-ul partajat de ambele tool-uri.
RUNNER = AgentRunner()

# LLM-as-Judge activabil din env: GUARDRAIL_LLM_JUDGE=1 (default: doar regex).
_USE_LLM_JUDGE = os.getenv("GUARDRAIL_LLM_JUDGE", "0") == "1"


def _build_guardrail() -> Guardrail:
    """Construiește guardrail-ul; atașează LLM-ul L6 doar dacă judge-ul e cerut."""
    llm = None
    if _USE_LLM_JUDGE:
        try:
            llm = RUNNER.get_llm()
        except Exception as e:  # noqa: BLE001
            logger.warning("Nu pot iniția LLM pentru judge (%s) — rămân pe regex.", e)
    return Guardrail(allowed_fields={"question", "query"}, llm=llm, use_llm_judge=_USE_LLM_JUDGE)


GUARDRAIL = _build_guardrail()


# ── 2a. tools/list — schema input/output pentru ambele tool-uri ─────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="data_analyst",
            description=(
                "Data Analyst Agent (L6): răspunde la întrebări analitice pe datele "
                "SQL de achiziții publice (NL2SQL + plan + tools). Ex: 'Top 5 furnizori "
                "după valoare?'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Întrebarea de analiză în limbaj natural.",
                        "minLength": 3,
                        "maxLength": 2000,
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="orchestrator",
            description=(
                "Orchestrator Agent (Supervisor + RAG, L6): răspunde la întrebări pe "
                "baza documentelor (facturi, contracte, clienți, rapoarte). Ex: 'Ce "
                "contact are DataPro?'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Întrebarea pentru căutarea în documente (RAG).",
                        "minLength": 3,
                        "maxLength": 2000,
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
    ]


# ── 2b. tools/call — guardrail apoi agent (Slide 3 & 7) ─────────────────────
# Mapare tool -> (nume câmp input, funcția agentului)
_TOOLS = {
    "data_analyst": ("question", RUNNER.run_data_analyst),
    "orchestrator": ("query", RUNNER.run_orchestrator),
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("tools/call -> %s args=%s", name, {k: str(v)[:60] for k, v in (arguments or {}).items()})

    if name not in _TOOLS:
        # Tool inexistent -> eroare JSON-RPC (Slide 7).
        raise McpError(types.ErrorData(code=-32601, message=f"Tool necunoscut: {name}"))

    field_name, runner_fn = _TOOLS[name]
    value = (arguments or {}).get(field_name)

    # ── CERINȚA 3: GUARDRAIL pe input (validare + anti prompt-injection) ────
    try:
        clean_value = GUARDRAIL.enforce(field_name, value, arguments)
    except GuardrailError as e:
        # Input blocat: îl raportăm ca tool result cu mesaj clar (isError=True la
        # client). Nu ridicăm McpError ca să nu confundăm cu o eroare de protocol —
        # un input respins de guardrail e un rezultat valid al tool-ului.
        msg = (
            f"🚫 Input BLOCAT de guardrail [{e.result.stage}].\n"
            f"Motiv: {e.result.reason}"
            + (f"\nMatch: {e.result.matched}" if e.result.matched else "")
        )
        logger.warning(msg.replace("\n", " | "))
        return [types.TextContent(type="text", text=msg)]

    # ── Apel agent (rulat off-thread: agenții L6 sunt sincroni/blocanți) ─────
    result = await asyncio.to_thread(runner_fn, clean_value)

    text = (
        f"[{result['agent']}] status={result['status']}\n\n{result['answer']}"
    )
    return [types.TextContent(type="text", text=text)]


# ── 3. RUN — loop stdio (pentru Claude Code / client) ───────────────────────
async def main() -> None:
    logger.info("Server MCP 'l6-agents' pornit (stdio). Tools: data_analyst, orchestrator.")
    logger.info("Guardrail: validare + regex%s.", " + LLM-as-Judge" if _USE_LLM_JUDGE else "")
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="l6-agents",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
