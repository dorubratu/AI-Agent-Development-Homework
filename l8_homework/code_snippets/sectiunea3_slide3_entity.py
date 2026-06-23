# ─────────────────────────────────────────────────────────
# Slide 3 · Entity Memory
# Extrage entități din conversație și le ține structurat,
# accesibile direct din entity_store.
#
# ConversationEntityMemory e deprecated din v0.3.1 (eliminat la v1.0.0).
# Mai jos: (A) varianta clasică de pe slide, (B) echivalentul modern.
# ─────────────────────────────────────────────────────────

# ── (A) CLASIC — așa apare în multe tutoriale (DEPRECATED v0.3.1) ──
from langchain.memory import ConversationEntityMemory
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
memory = ConversationEntityMemory(llm=llm)

# save_context() trimite mesajul către LLM, care extrage entitățile
memory.save_context(
    {"input": "Andrei de la TechCorp are 5 facturi de la Furnizor ABC"},
    {"output": "Am notat."},
)

# entity_store.store = dict {entitate: descriere}, accesibil direct
print(memory.entity_store.store)
# {'Andrei': 'Andrei lucrează la TechCorp...', 'ABC': 'Furnizor cu 5 facturi...'}

# load_memory_variables() returnează DOAR entitățile relevante pt. query
print(memory.load_memory_variables({"input": "Cât am la ABC?"}))


# ── (B) MODERN — extracție cu structured output + store pe entitate ──
# Recomandat azi: LLM-ul extrage entități structurate, le ții într-un
# dict (sau LangGraph BaseStore pt. persistență cross-sesiune).
from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(description="numele entității")
    type: str = Field(description="persoană / companie / furnizor")
    facts: list[str] = Field(description="fapte despre entitate")

class Entities(BaseModel):
    entities: list[Entity]

extractor = llm.with_structured_output(Entities)

entity_store: dict[str, Entity] = {}
def remember(text: str):
    for e in extractor.invoke(f"Extrage entitățile din: {text}").entities:
        entity_store[e.name] = e        # update/insert pe nume

remember("Andrei de la TechCorp are 5 facturi de la Furnizor ABC")
print(entity_store["ABC"])              # lookup instant, structurat
