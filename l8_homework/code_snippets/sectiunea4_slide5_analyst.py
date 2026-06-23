# ─────────────────────────────────────────────────────────
# Slide 5 · Integrare în Document Analyst
# Pattern complet: load → invoke → save. Memoria supraviețuiește
# restart-urilor pentru că trăiește în PostgreSQL, nu în RAM.
# ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sectiunea4_slide4_memory_manager import PersistentMemory


class DocumentAnalyst:
    def __init__(self, memory: PersistentMemory):
        self.memory = memory
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Ești un asistent pentru analiză de documente."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

    def chat(self, session_id: str, user_input: str) -> str:
        # 1. LOAD — istoricul din DB (ultimele N mesaje)
        history = self.memory.load_messages(session_id)

        # 2. INVOKE — prompt | llm cu istoricul injectat
        chain = self.prompt | self.llm
        reply = chain.invoke({"history": history, "input": user_input}).content

        # 3. SAVE — persistăm ambele mesaje (×2), fiecare în tranzacția lui
        self.memory.save_message(session_id, "user", user_input)
        self.memory.save_message(session_id, "assistant", reply)

        return reply


# Folosire:
#   analyst.chat("sesiune-andrei", "Procesează facturile de la Mega Image")
#   --- RESTART APP ---
#   analyst.chat("sesiune-andrei", "Cât am plătit la Mega Image?")
#   → își amintește, pentru că a încărcat din PostgreSQL.
#
# ⭐ Bonus: LangGraph oferă PostgresSaver built-in pentru checkpoints de workflow.
