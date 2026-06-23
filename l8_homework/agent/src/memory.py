"""
Conversation Memory - L8 Task 1
================================

PersistentMemory: un manager care încarcă istoricul conversației din
PostgreSQL și salvează după fiecare interacțiune. API curat pentru restul
aplicației — nu știe de SQL.

Pattern de folosire (load → invoke → save):

    memory = PersistentMemory(window=10)
    history = memory.load_messages("user-andrei")     # ultimele N mesaje
    # ... rulează agentul cu `history` injectat în context ...
    memory.save_message("user-andrei", "user", user_input)
    memory.save_message("user-andrei", "assistant", reply)

Pentru că totul trăiește în DB, memoria supraviețuiește restart-urilor
aplicației — exact ce cere tema.
"""
import logging

from database import transaction
from repositories import ChatMessageRepository

logger = logging.getLogger(__name__)


class PersistentMemory:
    """Manager de memorie conversațională, persistat în PostgreSQL."""

    def __init__(self, window: int = 10):
        # window = câte mesaje (user+assistant) reinjectăm în context.
        # Strategie tip ConversationBufferWindowMemory: doar ultimele N.
        self.window = window

    def load_messages(self, session_id: str) -> list[dict]:
        """
        Încarcă ultimele `window` mesaje pentru sesiune, în ordine cronologică.

        Returnează formatul standard de mesaje: [{"role", "content"}, ...]
        gata de injectat în lista trimisă la LLM.
        """
        with transaction() as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            messages = [{"role": m.role, "content": m.content} for m in rows]
        logger.info(f"[MEMORY] load session={session_id!r} → {len(messages)} msg")
        return messages

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Persistă un mesaj. Tranzacție atomică (commit/rollback automat)."""
        with transaction() as db:
            ChatMessageRepository(db).add(session_id, role, content)
        logger.info(f"[MEMORY] save session={session_id!r} role={role}")

    def save_turn(self, session_id: str, user_input: str, reply: str) -> None:
        """Salvează un tur complet (mesajul user + răspunsul assistant)."""
        self.save_message(session_id, "user", user_input)
        self.save_message(session_id, "assistant", reply)

    def history_count(self, session_id: str) -> int:
        with transaction() as db:
            return ChatMessageRepository(db).count(session_id)

    def clear(self, session_id: str) -> int:
        """Șterge toată conversația (util la teste)."""
        with transaction() as db:
            return ChatMessageRepository(db).delete_session(session_id)
