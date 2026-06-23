# ─────────────────────────────────────────────────────────
# Slide 5 · LangChain · LCEL
# Pattern modern: memory + prompt + invoke, totul ca un singur
# pipeline (prompt | llm) împachetat cu istoricul automat.
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# 1. prompt | llm  → pipeline-ul LCEL
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ești un asistent pentru analiză de documente."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | llm

# 2. RunnableWithMessageHistory înlocuiește vechiul
#    load_memory_variables() + save_context() — le face automat.
store = {}
chat = RunnableWithMessageHistory(
    chain,
    lambda sid: store.setdefault(sid, InMemoryChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history",
)

# 3. invoke — istoricul se încarcă ÎNAINTE și se salvează DUPĂ, transparent
cfg = {"configurable": {"session_id": "doc-analyst"}}
chat.invoke({"input": "Procesez 5 facturi de la ACME."}, config=cfg)
print(chat.invoke({"input": "Câte facturi am?"}, config=cfg).content)
