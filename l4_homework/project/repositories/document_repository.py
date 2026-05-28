from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import Document


class DocumentRepository:
    _UPDATABLE = {"filename", "content", "doc_type", "doc_metadata"}

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        filename: str,
        content: str,
        doc_type: str = "other",
        doc_metadata: dict[str, Any] | None = None,
    ) -> Document:
        doc = Document(
            filename=filename,
            content=content,
            doc_type=doc_type,
            doc_metadata=doc_metadata or {},
        )
        self.db.add(doc)
        self.db.flush()  # populează doc.id înainte de a-l folosi pentru chunks
        return doc

    def get_by_id(self, doc_id: int) -> Document | None:
        return self.db.get(Document, doc_id)

    def get_by_filename(self, filename: str) -> Document | None:
        return self.db.query(Document).filter(Document.filename == filename).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Document]:
        return (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count(self) -> int:
        return self.db.query(func.count(Document.id)).scalar() or 0

    def delete(self, doc_id: int) -> bool:
        doc = self.db.get(Document, doc_id)
        if doc is None:
            return False
        self.db.delete(doc)
        return True
