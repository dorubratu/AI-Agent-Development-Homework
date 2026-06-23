"""
Test runner - rulează ambele sisteme și salvează rezultatele COMPLETE
(răspunsuri netrunchiate) în test-output/results_<timestamp>.md
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Reduce logging noise în consolă
logging.basicConfig(level=logging.WARNING)

from skillab import get_llm
from state import OrchestratorState
from orchestrator import Orchestrator
from analyst_agent import AnalystAgent

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent

_ALIASES = {"gemini": "google", "ollama": "local"}
_MODEL_ENV = {"google": "GOOGLE_MODEL", "anthropic": "ANTHROPIC_MODEL", "openai": "OPENAI_MODEL", "local": "OLLAMA_MODEL"}


def _provider():
    p = os.getenv("LLM_PROVIDER")
    return _ALIASES.get(p.lower(), p.lower()) if p else None


def _model(provider):
    if not provider:
        return None
    return os.getenv("LLM_MODEL") or os.getenv(_MODEL_ENV.get(provider, f"{provider.upper()}_MODEL"))


def main():
    provider = _provider()
    model = _model(provider)
    llm = get_llm(provider=provider, model=model)

    lines = []
    lines.append(f"# Test Output — Multi-Agent System")
    lines.append("")
    lines.append(f"- **Date:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- **LLM Provider:** {provider}")
    lines.append(f"- **Model:** {llm.model}")
    lines.append("")

    # ---- Test 1: Orchestrator + RAG ----
    lines.append("## 1. Orchestrator + RAG")
    lines.append("")
    orch = Orchestrator(llm=llm)
    app = orch.build_graph()

    rag_queries = [
        "Care e totalul facturilor TechSoft?",
        "Ce contact are DataPro?",
    ]
    for q in rag_queries:
        print(f"[RAG] {q}")
        result = app.invoke(OrchestratorState(query=q))
        lines.append(f"### Query: {q}")
        lines.append(f"- **Status:** `{result['status']}`")
        lines.append(f"- **Iterations:** {result.get('iteration', '?')}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(result["answer"])
        lines.append("")
        lines.append("---")
        lines.append("")

    # ---- Test 2: Analyst + NL2SQL ----
    lines.append("## 2. Analyst + NL2SQL")
    lines.append("")
    analyst = AnalystAgent(
        tables_config={
            "achizitii_directe": {
                "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_achizitii_directe.json"),
                "business_path": str(DATA_DIR / "nl2sql_agent" / "business_achizitii_directe.json"),
            },
            "anunturi_initiere": {
                "schema_path": str(DATA_DIR / "nl2sql_agent" / "schema_anunturi_initiere.json"),
                "business_path": str(DATA_DIR / "nl2sql_agent" / "business_anunturi_initiere.json"),
            },
        },
        llm=llm,
    )

    sql_queries = [
        "Care sunt top 5 furnizori după valoare?",
        "Câte contracte are fiecare autoritate contractantă? Top 5.",
    ]
    for q in sql_queries:
        print(f"[ANALYST] {q}")
        result = analyst.chat(q)
        lines.append(f"### Question: {q}")
        lines.append(f"- **Status:** `{result['status']}`")
        lines.append(f"- **Reasoning:** {result.get('reasoning', '')}")
        lines.append("")
        lines.append("**Plan steps:**")
        for step in result.get("plan", []):
            lines.append(f"- `{step.id}` ({step.action})")
        lines.append("")
        lines.append("**Step results:**")
        for sr in result.get("step_results", []):
            lines.append(f"- `{sr.step_id}`: {sr.status} ({sr.row_count} rows) {('- ' + sr.error) if sr.error else ''}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(result["answer"])
        lines.append("")
        lines.append("---")
        lines.append("")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"results_{ts}.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Results saved to: {out_file}")


if __name__ == "__main__":
    main()
