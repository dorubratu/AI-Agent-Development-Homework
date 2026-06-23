# ─────────────────────────────────────────────────────────
# Slide 5 · LangGraph · State
# Memory built-in în state: mesajele se acumulează automat
# (reducer add_messages) și sunt persistate între invocări
# de către checkpointer — fără management manual.
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# MessagesState = state cu un câmp `messages` adnotat cu reducer-ul
# add_messages → mesajele noi se APPENDează, nu se suprascriu.
def agent(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}     # ← reducer-ul îl adaugă la istoric

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# Checkpointer = persistența. Thread-ul (session) e identificat prin thread_id.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "andrei"}}
graph.invoke({"messages": [("user", "Mă numesc Andrei.")]}, config=cfg)
out = graph.invoke({"messages": [("user", "Cum mă numesc?")]}, config=cfg)
print(out["messages"][-1].content)   # → "Te numești Andrei."

# ✓ Memory built-in   ✓ Acumulare automată   ✓ Persistat între noduri
