# ─────────────────────────────────────────────────────────
# Slide 3 · Summary Memory  →  comprimă istoricul cu un LLM
# Modern: custom history care, când trece de prag, cere LLM-ului
# un rezumat și înlocuiește mesajele vechi cu el.
# (ConversationSummaryMemory e deprecated din v0.3.1)
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

MAX_MESSAGES = 6          # prag: peste atâtea mesaje → sumarizăm
summarizer = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

class SummaryHistory(InMemoryChatMessageHistory):
    """Când istoricul crește, comprimăm mesajele vechi într-un rezumat."""
    def add_messages(self, messages: list[BaseMessage]) -> None:
        super().add_messages(messages)
        if len(self.messages) > MAX_MESSAGES:
            old = self.messages[:-2]                  # tot, mai puțin ultimul schimb
            convo = "\n".join(f"{m.type}: {m.content}" for m in old)
            summary = summarizer.invoke(
                f"Rezumă concis conversația, păstrând faptele cheie:\n{convo}"
            ).content
            # înlocuim mesajele vechi cu un singur SystemMessage-rezumat
            self.messages = [SystemMessage(f"Rezumat anterior: {summary}")] + self.messages[-2:]

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ești un asistent util."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chat = RunnableWithMessageHistory(
    prompt | llm, lambda sid: store.setdefault(sid, SummaryHistory()),
    input_messages_key="input", history_messages_key="history",
)
store = {}

# Mesaje originale (~150 tokeni) → Summary (~25 tokeni), 6× mai compact.
# ✨ SummaryBuffer = rezumat pt. mesajele vechi + buffer pt. cele recente.
