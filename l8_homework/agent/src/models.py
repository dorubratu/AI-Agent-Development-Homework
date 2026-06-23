"""
SQLAlchemy Models
"""
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, Numeric, Index, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


# ============================================================
# CONVERSATION MEMORY (L8 - Task 1)
# Long-term memory persistat în PostgreSQL: o conversație
# (session) are mai multe mesaje. Memoria supraviețuiește
# restart-urilor pentru că trăiește în DB, nu în RAM.
# ============================================================

class ChatSession(Base):
    """O conversație / sesiune, identificată printr-un session_id."""
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)            # session_id (ex: "user-andrei")
    user_id = Column(String(64), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    """Un mesaj dintr-o conversație (user / assistant)."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id"), index=True, nullable=False)
    role = Column(String(16), nullable=False)            # "user" / "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DocumentChunk(Base):
    """Chunk pentru RAG cu pgvector."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50))
    content = Column(Text, nullable=False)
    summary = Column(Text)
    metadata_json = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_document_chunks_embedding', 'embedding', postgresql_using='ivfflat'),
    )


class AchizitieDirecta(Base):
    """Achiziție directă din SEAP."""
    __tablename__ = "achizitii_directe"

    id = Column(Integer, primary_key=True)
    castigator = Column(String(500))
    castigator_cui = Column(String(50))
    castigator_tara = Column(String(100))
    castigator_localitate = Column(String(200))
    castigator_adresa = Column(Text)
    tip_procedura = Column(String(200))
    autoritate_contractanta = Column(String(500))
    autoritate_contractanta_cui = Column(String(50))
    numar_anunt = Column(String(100))
    data_anunt = Column(DateTime)
    descriere = Column(Text)
    tip_incheiere_contract = Column(String(200))
    numar_contract = Column(String(100))
    data_contract = Column(DateTime)
    titlu_contract = Column(Text)
    valoare = Column(Numeric(15, 2))
    moneda = Column(String(10))
    valoare_ron = Column(Numeric(15, 2))
    valoare_eur = Column(Numeric(15, 2))
    cpv_code_id = Column(String(50))
    cpv_code = Column(String(200))


class AnuntInitiere(Base):
    """Anunț inițiere licitație."""
    __tablename__ = "anunturi_initiere"

    id = Column(Integer, primary_key=True)
    tip_anunt = Column(String(200))
    numar_anunt_invitatie = Column(String(100))
    data_publicare = Column(DateTime)
    denumire_ac = Column(String(500))
    cui = Column(String(50))
    judet = Column(String(100))
    tip_contract = Column(String(200))
    utilitati = Column(String(100))
    tip_procedura = Column(String(200))
    criteriu_atribuire = Column(String(200))
    valoare_estimata = Column(Numeric(15, 2))
    moneda = Column(String(10))
    modalitate_desfasurare = Column(String(200))
    trimis_ojeu = Column(String(50))
    fonduri_comunitare = Column(String(50))
    main_cpv_code = Column(String(50))
    main_cpv_name = Column(String(500))
