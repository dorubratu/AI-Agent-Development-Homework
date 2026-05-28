from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DocType = Literal["invoice", "contract", "other"]


class InvoiceLineItem(BaseModel):
    description: str = Field(..., description="Descrierea produsului sau serviciului.")
    quantity: float | None = Field(default=None, description="Cantitatea facturată.")
    unit_price: float | None = Field(default=None, description="Pretul unitar (fără TVA).")
    total: float | None = Field(default=None, description="Total pentru această linie.")


class Invoice(BaseModel):
    invoice_number: str | None = Field(default=None, description="Seria și numărul facturii (ex: 'FAC-2024-001').")
    issue_date: str | None = Field(default=None, description="Data emiterii facturii (YYYY-MM-DD dacă e posibil).")
    due_date: str | None = Field(default=None, description="Data scadenței (YYYY-MM-DD dacă e posibil).")
    supplier_name: str | None = Field(default=None, description="Numele furnizorului / emitentului.")
    customer_name: str | None = Field(default=None, description="Numele clientului / cumpărătorului.")
    currency: str | None = Field(default=None, description="Codul valutei (RON, EUR, USD).")
    subtotal: float | None = Field(default=None, description="Suma fără TVA.")
    tax_amount: float | None = Field(default=None, description="Valoarea TVA.")
    total_amount: float | None = Field(default=None, description="Total de plată cu TVA inclus.")
    line_items: list[InvoiceLineItem] = Field(default_factory=list)


class Contract(BaseModel):
    title: str | None = Field(default=None, description="Titlul contractului.")
    parties: list[str] = Field(default_factory=list, description="Părțile semnatare ale contractului.")
    effective_date: str | None = Field(default=None, description="Data intrării în vigoare.")
    termination_clause: str | None = Field(default=None, description="Rezumat scurt al clauzei de reziliere.")
    notice_period_days: int | None = Field(default=None, description="Perioada de preaviz, în zile (dacă e menționată).")
    payment_terms: str | None = Field(default=None, description="Rezumat al termenilor de plată / penalități.")
    governing_law: str | None = Field(default=None, description="Jurisdicția aplicabilă (ex: 'România').")


class DocumentExtraction(BaseModel):
    filename: str = Field(..., description="Fișierul original.")
    content: str = Field(..., min_length=1, description="Text complet extras.")
    doc_type: DocType = Field(default="other")
    invoice: Invoice | None = Field(default=None, description="Set doar dacă doc_type == 'invoice'.")
    contract: Contract | None = Field(default=None, description="Set doar dacă doc_type == 'contract'.")
    summary: str = Field(default="", description="Rezumat scurt al documentului în 1-3 propoziții.")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content cannot be empty or whitespace-only")
        return v

    def to_db_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"summary": self.summary}
        if self.invoice is not None:
            meta["invoice"] = self.invoice.model_dump(exclude_none=True)
        if self.contract is not None:
            meta["contract"] = self.contract.model_dump(exclude_none=True)
        return meta


class ExtractionResult(BaseModel):
    success: bool
    filename: str
    document_id: int | None = None
    extraction: DocumentExtraction | None = None
    chunk_count: int = 0
    error: str | None = None
