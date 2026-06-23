# ─────────────────────────────────────────────────────────
# Slide 4 · Memory Manager — Repository pattern + transaction
# Separăm persistența (repository) de business logic (manager).
# Tranzacția e gestionată explicit prin context manager (unit of work).
# ─────────────────────────────────────────────────────────
from contextlib import contextmanager
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession, sessionmaker
from sectiunea4_slide3_schema import ChatMessage


# ── Unit of Work: o tranzacție = un context. Commit la succes, rollback la eroare.
@contextmanager
def unit_of_work(session_factory: sessionmaker):
    db: DBSession = session_factory()
    try:
        yield db
        db.commit()              # totul a mers → persistăm atomic
    except Exception:
        db.rollback()            # ceva a crăpat → anulăm tot
        raise
    finally:
        db.close()


# ── Repository: TOATĂ logica de DB pentru mesaje, izolată aici.
#    Business logic-ul nu vede niciodată SQL/ORM direct.
class ChatMessageRepository:
    def __init__(self, db: DBSession):
        self.db = db

    def add(self, session_id: str, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(session_id=session_id, role=role, content=content)
        self.db.add(msg)
        return msg

    def latest(self, session_id: str, limit: int) -> list[ChatMessage]:
        # ORDER BY timestamp DESC + LIMIT N  → ultimele N (window)
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            # id ca tiebreaker: mesajele scrise în aceeași secundă au timestamp
            # identic, iar id-ul auto-increment garantează ordinea reală.
            .order_by(ChatMessage.timestamp.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        rows = self.db.execute(stmt).scalars().all()
        return list(reversed(rows))      # reverse → cronologic pentru LLM


# ── Manager: API curat pentru restul aplicației. Nu știe de SQL.
class PersistentMemory:
    def __init__(self, session_factory: sessionmaker, window: int = 10):
        self.session_factory = session_factory
        self.window = window

    def load_messages(self, session_id: str) -> list[dict]:
        with unit_of_work(self.session_factory) as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            return [{"role": m.role, "content": m.content} for m in rows]

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with unit_of_work(self.session_factory) as db:     # tranzacție atomică
            ChatMessageRepository(db).add(session_id, role, content)
