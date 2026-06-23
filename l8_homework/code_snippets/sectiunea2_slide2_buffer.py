# ─────────────────────────────────────────────────────────
# Slide 2 · Buffer Memory  →  păstrează TOATE mesajele
# Modern (LangChain v0.3+). Clasicul ConversationBufferMemory
# e deprecated din v0.3.1 → folosim RunnableWithMessageHistory.
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ești un asistent util."),
    MessagesPlaceholder("history"),   # ← aici se injectează tot istoricul
    ("human", "{input}"),
])
chain = prompt | llm

# Buffer = un singur store care reține TOT, fără limită.
store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
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

# ✓ Simplu, păstrează tot   ✗ Crește nelimitat → cost mare
