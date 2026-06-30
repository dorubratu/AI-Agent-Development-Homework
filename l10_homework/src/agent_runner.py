"""
agent_runner.py — încarcă și rulează agenții din L6 (Cerințele 1 & 2).

Acest modul e puntea dintre serverul MCP și agenții reali din `l6_homework`:
  - Data Analyst Agent  (NL2SQL + tools)         -> run_data_analyst(question)
  - Orchestrator Agent  (Supervisor + RAG)        -> run_orchestrator(query)

Agenții L6 au nevoie de o bază Postgres + chei LLM. Ca tema să fie un cod
FUNCȚIONAL și testabil chiar fără infrastructură completă, folosim lazy-loading
și un fallback explicit: dacă agentul real nu poate fi instanțiat (lipsă DB /
dependențe / chei), tool-ul întoarce un răspuns marcat `degraded` în loc să
crape serverul. Când infra e prezentă, rulează agentul real din L6.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Rădăcina proiectului și a lui L6 (refolosim codul existent, nu îl duplicăm).
ROOT = Path(__file__).parent.parent.parent
L6 = ROOT / "l6_homework"

load_dotenv(L6 / ".env")   # config LLM/DB din L6
load_dotenv(ROOT / "l10_homework" / ".env", override=False)

# Pune pe sys.path codul L6 + librăria skillab.
for p in (str(L6 / "skillab-py" / "src"), str(L6 / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Rezolvare provider/model (aliniat cu l6_homework/src/main.py) ──────────
_PROVIDER_ALIASES = {"gemini": "google", "ollama": "local"}
_MODEL_ENV_VARS = {
    "google": "GOOGLE_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "local": "OLLAMA_MODEL",
}


def _resolve_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    return _PROVIDER_ALIASES.get(provider.lower(), provider.lower())


def _model_from_env(provider: str | None) -> str | None:
    resolved = _resolve_provider(provider)
    if not resolved:
        return None
    env_var = _MODEL_ENV_VARS.get(resolved, f"{resolved.upper()}_MODEL")
    return os.getenv("LLM_MODEL") or os.getenv(env_var)


_DATA_DIR = L6 / "data"
_TABLES_CONFIG = {
    "achizitii_directe": {
        "schema_path": str(_DATA_DIR / "nl2sql_agent" / "schema_achizitii_directe.json"),
        "business_path": str(_DATA_DIR / "nl2sql_agent" / "business_achizitii_directe.json"),
    },
    "anunturi_initiere": {
        "schema_path": str(_DATA_DIR / "nl2sql_agent" / "schema_anunturi_initiere.json"),
        "business_path": str(_DATA_DIR / "nl2sql_agent" / "business_anunturi_initiere.json"),
    },
}


class AgentRunner:
    """
    Container leneș (lazy) pentru agenții L6. Le instanțiază o singură dată, la
    prima utilizare, și le refolosește (sunt scumpe: încarcă scheme + LLM + RAG).
    """

    def __init__(self) -> None:
        self._llm = None
        self._analyst = None
        self._orchestrator = None
        self._orch_app = None

    # ── LLM partajat ───────────────────────────────────────────────────────
    def get_llm(self):
        if self._llm is None:
            from skillab import get_llm
            provider = _resolve_provider(os.getenv("LLM_PROVIDER"))
            model = _model_from_env(os.getenv("LLM_PROVIDER"))
            self._llm = get_llm(provider=provider, model=model)
            logger.info("LLM init: provider=%s model=%s", provider, getattr(self._llm, "model", "?"))
        return self._llm

    # ── Cerința 1: Data Analyst ──────────────────────────────────────────────
    def run_data_analyst(self, question: str) -> dict[str, Any]:
        """Apelează agentul Data Analyst din L6 și normalizează răspunsul."""
        try:
            if self._analyst is None:
                from analyst_agent import AnalystAgent
                self._analyst = AnalystAgent(tables_config=_TABLES_CONFIG, llm=self.get_llm())
            result = self._analyst.chat(question)
            return {
                "agent": "data_analyst",
                "status": result.get("status", "unknown"),
                "answer": result.get("answer", ""),
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("Data Analyst indisponibil")
            return self._degraded("data_analyst", question, e)

    # ── Cerința 2: Orchestrator (Supervisor + RAG) ───────────────────────────
    def run_orchestrator(self, query: str) -> dict[str, Any]:
        """Apelează agentul Orchestrator din L6 și normalizează răspunsul."""
        try:
            if self._orch_app is None:
                from orchestrator import Orchestrator
                self._orchestrator = Orchestrator(llm=self.get_llm())
                self._orch_app = self._orchestrator.build_graph()
            from state import OrchestratorState
            result = self._orch_app.invoke(OrchestratorState(query=query))
            return {
                "agent": "orchestrator",
                "status": result.get("status", "unknown"),
                "answer": result.get("answer", ""),
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("Orchestrator indisponibil")
            return self._degraded("orchestrator", query, e)

    @staticmethod
    def _degraded(agent: str, prompt: str, error: Exception) -> dict[str, Any]:
        """Răspuns de tip 'degraded' când agentul real nu poate rula (lipsă infra)."""
        return {
            "agent": agent,
            "status": "degraded",
            "answer": (
                f"[{agent}] Input acceptat și trecut de guardrails, dar agentul L6 nu a "
                f"putut rula în acest mediu ({type(error).__name__}: {error}). "
                f"Pornește baza de date și setează cheile LLM (vezi README) pentru răspuns real.\n"
                f"Input primit: {prompt!r}"
            ),
        }
