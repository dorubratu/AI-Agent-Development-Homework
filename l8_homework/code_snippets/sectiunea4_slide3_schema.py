# ─────────────────────────────────────────────────────────
# Slide 3 · Schema SQLAlchemy 2.0  (PostgreSQL)
# 3 modele ORM relaționate, stil modern Mapped[] / mapped_column.
# ─────────────────────────────────────────────────────────
from datetime import datetime
from sqlalchemy import ForeignKey, String, Text, JSON, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarativ comun tuturor modelelor."""
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)   # ex: UUID
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    meta: Mapped[dict] = mapped_column(JSON, default=dict)          # 'metadata' e rezervat în SQLAlchemy

    # 1:N — o sesiune are multe mesaje și multe entități
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    entities: Mapped[list["Entity"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))                  # "user" / "assistant"
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    session: Mapped["Session"] = relationship(back_populates="messages")


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)     # index pt. lookup rapid
    info: Mapped[dict] = mapped_column(JSON, default=dict)         # {type, facts: [...]}
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    session: Mapped["Session"] = relationship(back_populates="entities")
