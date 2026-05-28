from __future__ import annotations

import json
import logging
import os
from typing import Type

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from prompts.registry import get_prompt_registry
from .schemas import Contract, DocumentExtraction, Invoice

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
MAX_CONTENT_CHARS = 8000


class StructuredExtractor:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.prompts = get_prompt_registry()

    def _call_with_schema(
        self,
        schema_cls: Type[BaseModel],
        tool_name: str,
        tool_description: str,
        system: str,
        user_text: str,
    ) -> BaseModel:
        schema = schema_cls.model_json_schema()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_text}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema_cls(**block.input)

        raise RuntimeError(
            f"Model did not call the forced tool '{tool_name}'. "
            f"stop_reason={response.stop_reason}"
        )

    def classify(self, filename: str, content: str) -> str:
        class Classification(BaseModel):
            doc_type: str
            summary: str

        system = self.prompts.render("classify", filename=filename)
        snippet = content[:MAX_CONTENT_CHARS]

        try:
            result = self._call_with_schema(
                Classification,
                tool_name="set_classification",
                tool_description="Set the document type and a short summary.",
                system=system,
                user_text=snippet,
            )
        except (ValidationError, RuntimeError) as exc:
            logger.warning("classification failed for %s: %s", filename, exc)
            return "other"

        doc_type = result.doc_type.lower().strip()
        if doc_type not in {"invoice", "contract", "other"}:
            doc_type = "other"
        # summary is captured separately via extract()
        return doc_type

    def extract(self, filename: str, content: str) -> DocumentExtraction:
        snippet = content[:MAX_CONTENT_CHARS]
        doc_type = self.classify(filename, content)

        invoice: Invoice | None = None
        contract: Contract | None = None
        summary = ""

        if doc_type == "invoice":
            system = self.prompts.render("extract_invoice", filename=filename)
            try:
                invoice = self._call_with_schema(  # type: ignore[assignment]
                    Invoice,
                    tool_name="set_invoice_fields",
                    tool_description="Capture invoice fields from the text.",
                    system=system,
                    user_text=snippet,
                )
            except (ValidationError, RuntimeError) as exc:
                logger.warning("invoice extraction failed for %s: %s", filename, exc)
        elif doc_type == "contract":
            system = self.prompts.render("extract_contract", filename=filename)
            try:
                contract = self._call_with_schema(  # type: ignore[assignment]
                    Contract,
                    tool_name="set_contract_fields",
                    tool_description="Capture contract fields from the text.",
                    system=system,
                    user_text=snippet,
                )
            except (ValidationError, RuntimeError) as exc:
                logger.warning("contract extraction failed for %s: %s", filename, exc)

        # Summary — un singur call indiferent de tip, e ieftin și util
        try:
            class _Summary(BaseModel):
                summary: str

            sys_sum = self.prompts.render("summarize", filename=filename)
            summary_obj = self._call_with_schema(
                _Summary,
                tool_name="set_summary",
                tool_description="Set a 1-3 sentence summary in the document language.",
                system=sys_sum,
                user_text=snippet,
            )
            summary = summary_obj.summary  # type: ignore[attr-defined]
        except (ValidationError, RuntimeError) as exc:
            logger.warning("summary failed for %s: %s", filename, exc)

        return DocumentExtraction(
            filename=filename,
            content=content,
            doc_type=doc_type,  # type: ignore[arg-type]
            invoice=invoice,
            contract=contract,
            summary=summary,
        )

    def to_json(self, extraction: DocumentExtraction) -> str:
        return json.dumps(
            extraction.model_dump(exclude={"content"}),
            ensure_ascii=False,
            indent=2,
        )
