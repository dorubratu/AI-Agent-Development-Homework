# ─────────────────────────────────────────────────────────
# Slide 2 · Window Memory  →  doar ultimele N mesaje (k=5)
# Modern: extindem InMemoryChatMessageHistory ca să taie la k.
# (ConversationBufferWindowMemory e deprecated din v0.3.1)
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

K = 5   # câte mesaje recente păstrăm

class WindowHistory(InMemoryChatMessageHistory):
    """La fiecare adăugare, ține DOAR ultimele K mesaje."""
    def add_messages(self, messages: list[BaseMessage]) -> None:
        super().add_messages(messages)
        self.messages = self.messages[-K:]   # ← fereastra glisantă

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ești un asistent util."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | llm

store = {}
def get_history(session_id: str) -> WindowHistory:
    if session_id not in store:
        store[session_id] = WindowHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input",
    history_messages_key="history",
)
cfg = {"configurable": {"session_id": "andrei"}}
chat.invoke({"input": "Mă numesc Andrei."}, config=cfg)
print(chat.invoke({"input": "Cum mă numesc?"}, config=cfg).content)
# → "Te numești Andrei."  (tot istoricul e în store)

# ✓ Size fix, cost predictibil   ✗ Pierde contextul vechi
