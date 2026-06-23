"""
NL2SQL Agent - TODO: Implementează nodurile
"""
import json
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlparse
from langgraph.graph import StateGraph, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import NL2SQLState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class NL2SQLAgent:
    """
    Flow:
        get_context → generate_sql → validate → execute
                                        ↓
                                    handle_error
    """

    def __init__(
        self,
        table_name: str,
        schema_path: str,
        db_url: str = "postgresql://demo:demo123@localhost:5433/rag_demo",
        max_retries: int = 2,
        llm: LLMProvider | None = None,
    ):
        self.table_name = table_name
        self.db_url = db_url
        self.max_retries = max_retries
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load schema
        self.schema = json.loads(Path(schema_path).read_text())

        self.graph = self._build_graph()

    # === NODES ===

    def node_get_context(self, state: NL2SQLState) -> dict:
        """COMPLET."""
        return {
            "schema_context": self.schema,
            "table_name": self.table_name,
        }

    def node_generate_sql(self, state: NL2SQLState) -> dict:
        """Generează SQL din întrebare naturală."""
        logger.info(f"[GENERATE] {state.question}")

        schema = state.schema_context
        prompt = self.prompts.render(
            "nl2sql_generate",
            table_name=state.table_name,
            table_description=schema.get("description", ""),
            columns=schema.get("columns", {}),
            business_rules=self.schema.get("rules", {}),
            question=state.question,
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        # Strip ```sql ... ``` or ``` ... ``` blocks
        match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        sql = match.group(1).strip() if match else response.strip()

        logger.info(f"[GENERATE] sql={sql[:80]}...")
        return {"sql_query": sql}

    def node_validate_sql(self, state: NL2SQLState) -> dict:
        """Validează SQL-ul generat."""
        sql = state.sql_query
        logger.info(f"[VALIDATE] {sql[:50]}...")

        # Check for dangerous patterns
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE", "EXEC", ";--", "/*"]
        sql_upper = sql.upper()
        for pattern in dangerous:
            if pattern in sql_upper:
                return {"is_valid": False, "validation_error": f"Forbidden keyword: {pattern}"}

        # Parse with sqlparse
        parsed = sqlparse.parse(sql)
        if not parsed:
            return {"is_valid": False, "validation_error": "Could not parse SQL"}

        stmt = parsed[0]
        if stmt.get_type() != "SELECT":
            return {"is_valid": False, "validation_error": f"Only SELECT is allowed, got: {stmt.get_type()}"}

        return {"is_valid": True, "validation_error": ""}

    def node_execute_sql(self, state: NL2SQLState) -> dict:
        """Execută SQL-ul validat și returnează rezultat ca DataFrame."""
        logger.info("[EXECUTE]")

        from database import transaction
        from sqlalchemy import text

        try:
            with transaction() as session:
                result = session.execute(text(state.sql_query))
                df = pd.DataFrame(result.mappings().all())
            logger.info(f"[EXECUTE] success, {len(df)} rows")
            return {"result": df, "execution_error": "", "status": "success"}
        except Exception as e:
            logger.error(f"[EXECUTE] error: {e}")
            return {"result": pd.DataFrame(), "execution_error": str(e), "status": "failed"}

    def node_handle_error(self, state: NL2SQLState) -> dict:
        """Gestionează eroarea și încearcă să corecteze SQL-ul."""
        new_retry = state.retry_count + 1
        logger.info(f"[ERROR] retry {new_retry}/{state.max_retries}")

        if new_retry >= state.max_retries:
            return {"retry_count": new_retry, "status": "failed"}

        error_message = state.execution_error or state.validation_error
        prompt = self.prompts.render(
            "nl2sql_error",
            table_name=state.table_name,
            question=state.question,
            failed_sql=state.sql_query,
            error_message=error_message,
            columns=state.schema_context.get("columns", {}),
        )

        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        new_sql = match.group(1).strip() if match else response.strip()

        logger.info(f"[ERROR] corrected sql={new_sql[:80]}...")
        return {"sql_query": new_sql, "retry_count": new_retry, "is_valid": False, "validation_error": "", "execution_error": ""}

    # === ROUTING ===

    def _route_after_validate(self, state: NL2SQLState) -> str:
        return "execute_sql" if state.is_valid else "handle_error"

    def _route_after_execute(self, state: NL2SQLState) -> str:
        return END if not state.execution_error else "handle_error"

    def _route_after_error(self, state: NL2SQLState) -> str:
        return "generate_sql" if state.retry_count < self.max_retries else END

    # === GRAPH ===

    def _build_graph(self):
        graph = StateGraph(NL2SQLState)

        graph.add_node("get_context", self.node_get_context)
        graph.add_node("generate_sql", self.node_generate_sql)
        graph.add_node("validate_sql", self.node_validate_sql)
        graph.add_node("execute_sql", self.node_execute_sql)
        graph.add_node("handle_error", self.node_handle_error)

        graph.set_entry_point("get_context")
        graph.add_edge("get_context", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_conditional_edges("validate_sql", self._route_after_validate, ["execute_sql", "handle_error"])
        graph.add_conditional_edges("execute_sql", self._route_after_execute, [END, "handle_error"])
        graph.add_conditional_edges("handle_error", self._route_after_error, ["generate_sql", END])

        return graph.compile()

    def run(self, question: str) -> NL2SQLState:
        """Execută agentul."""
        initial = NL2SQLState(
            question=question,
            table_name=self.table_name,
            max_retries=self.max_retries,
        )
        return self.graph.invoke(initial)
