"""
Orchestrator - Supervizor care coordonează RAG Agent

Flow (L8 - cu memorie):
    load_memory → call_rag → evaluate ──┬──→ answer → save_memory → END
                       ↑                │
                       │   can_answer   │
                       │   = false      │
                       └────────────────┘

L8 - extensii față de L6:
  • Task 1 (Conversation Memory): nodurile load_memory / save_memory persistă
    istoricul conversației în PostgreSQL (per session_id). Istoricul e injectat
    în node_answer → context persistent între request-uri, supraviețuiește
    restart-urilor.
  • Task 2 (Prompt Caching): node_answer trimite system prompt-ul + contextul
    fix marcat cu cache_control: ephemeral și măsoară tokenii economisiți.
"""
import logging
import re
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import OrchestratorState, OrchestratorFeedback
from rag_agent import RAGAgent, RAGAgentConfig
from memory import PersistentMemory

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# System prompt STATIC (prefixul care se cache-uiește). Identic la fiecare
# request → candidat perfect pentru prompt caching.
ANSWER_SYSTEM_PROMPT = (
    "Ești un asistent pentru analiză de documente. Răspunzi la întrebări "
    "folosind EXCLUSIV contextul furnizat (chunks din documente) și istoricul "
    "conversației. Răspunzi concis, în limba întrebării, și menționezi sursa "
    "(numele fișierului) când citezi o informație. Dacă informația nu există "
    "în context, spui clar că nu o ai."
)


class OrchestratorConfig:
    max_iterations: int = 3
    min_score: float = 0.25
    memory_window: int = 10        # L8: câte mesaje reinjectăm din DB
    use_prompt_cache: bool = True  # L8: activează prompt caching (Anthropic)


class Orchestrator:
    """
    Supervizor care coordonează RAG Agent, cu memorie persistentă și caching.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
        memory: PersistentMemory | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.memory = memory or PersistentMemory(window=self.config.memory_window)

        # Statistici cumulate de prompt caching (pentru raportare).
        self.cache_stats = {
            "calls": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # RAG Agent - graf separat
        rag_config = RAGAgentConfig()
        rag_config.default_threshold = self.config.min_score
        self.rag = RAGAgent(self.llm, rag_config)

    # === NODES ===

    def node_load_memory(self, state: OrchestratorState) -> dict:
        """L8: încarcă istoricul conversației din PostgreSQL (ultimele N mesaje)."""
        history = self.memory.load_messages(state.session_id)
        return {"history": history}

    def node_call_rag(self, state: OrchestratorState) -> dict:
        """Apelează RAG Agent pentru căutare. COMPLET - nu modifica."""
        logger.info(f"[CALL_RAG] iter {state.iteration + 1}")

        rag_result = self.rag.run(
            query=state.query,
            feedback=state.feedback,  # None prima dată
        )

        return {
            "rag_result": rag_result["result"],
            "iteration": state.iteration + 1,
        }

    def node_evaluate(self, state: OrchestratorState) -> dict:
        """Evaluează dacă contextul RAG e suficient pentru a răspunde."""
        logger.info(f"[EVALUATE] iter {state.iteration}")

        results = state.rag_result.results if state.rag_result else []
        context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in results)

        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=context,
            max_score=state.rag_result.max_score if state.rag_result else 0.0,
            avg_score=state.rag_result.avg_score if state.rag_result else 0.0,
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        json_str = match.group(1) if match else response.strip()
        feedback = OrchestratorFeedback.model_validate_json(json_str)

        logger.info(f"[EVALUATE] can_answer={feedback.can_answer}")
        return {"feedback": feedback}

    def node_answer(self, state: OrchestratorState) -> dict:
        """
        Generează răspunsul final.

        L8 Task 2: trimitem un system prompt STATIC + contextul documentelor
        marcate cu cache_control: ephemeral (prefix cacheabil). Istoricul
        conversației (Task 1) e injectat ca mesaje user/assistant.
        """
        logger.info("[ANSWER]")

        results = state.rag_result.results if state.rag_result else []

        if not results:
            return {
                "answer": "Nu am găsit informații relevante pentru întrebarea ta.",
                "status": "failed",
            }

        context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in results)

        # Prefix static (cacheabil): instrucțiuni + contextul documentelor.
        system_prompt = (
            ANSWER_SYSTEM_PROMPT
            + "\n\nCONTEXT DOCUMENTE:\n"
            + context
        )

        # Mesaje: system (cacheabil) + istoric conversație + întrebarea curentă.
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state.history)  # L8 Task 1: context persistent
        messages.append({"role": "user", "content": state.query})

        # L8 Task 2: dacă providerul suportă caching și e activat, folosim
        # generate_with_usage cu cache_system=True ca să măsurăm economia.
        use_cache = self.config.use_prompt_cache and self.llm.supports_prompt_caching()
        if use_cache:
            answer, usage = self.llm.generate_with_usage(messages, cache_system=True)
            self._record_usage(usage)
            logger.info(
                f"[ANSWER] cache: creation={usage.get('cache_creation_input_tokens')} "
                f"read={usage.get('cache_read_input_tokens')} "
                f"fresh_input={usage.get('input_tokens')}"
            )
        else:
            answer = self.llm.generate_sync(messages)

        if state.feedback and state.feedback.can_answer:
            status = "success"
        else:
            status = "partial"

        logger.info(f"[ANSWER] status={status}")
        return {"answer": answer, "status": status}

    def node_save_memory(self, state: OrchestratorState) -> dict:
        """L8: persistă turul (user + assistant) în PostgreSQL."""
        self.memory.save_turn(state.session_id, state.query, state.answer)
        return {}

    # === HELPERS ===

    def _record_usage(self, usage: dict) -> None:
        self.cache_stats["calls"] += 1
        for k in ("cache_creation_input_tokens", "cache_read_input_tokens",
                  "input_tokens", "output_tokens"):
            self.cache_stats[k] += usage.get(k, 0) or 0

    # === ROUTING ===

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        """Decide dacă continuăm căutarea sau răspundem."""
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.config.max_iterations:
            logger.info(f"[ROUTING] Max iterations ({self.config.max_iterations}) reached")
            return "answer"
        return "call_rag"

    # === GRAPH ===

    def build_graph(self):
        """Construiește graful Orchestrator (cu noduri de memorie)."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("load_memory", self.node_load_memory)
        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)
        graph.add_node("save_memory", self.node_save_memory)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "call_rag")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"call_rag": "call_rag", "answer": "answer"}
        )
        graph.add_edge("answer", "save_memory")
        graph.add_edge("save_memory", END)

        return graph.compile()
