# Document Analyst cu RAG (L3 + L4)

Pipeline de ingestion (load → chunk → extract structurat → embed → store)
și agent ReAct conectat la un tool RAG peste pgvector. Continuă agentul din
L1–L2 (Anthropic + Pydantic tools + YAML prompts), adăugând layer-ul de
storage și retrieval.

## Stack

- **LLM**: Anthropic `claude-opus-4-7` (reutilizat din L2). Output structurat
  prin `tool_choice` forced — un singur tool cu `input_schema` derivat din
  Pydantic, modelul e obligat să-l apeleze.
- **Embeddings**: `sentence-transformers / paraphrase-multilingual-MiniLM-L12-v2`
  (384 dim, suportă bine româna).
- **Storage**: PostgreSQL 16 + extensia `pgvector`, prin Docker.
- **ORM + migrații**: SQLAlchemy 2.0 + Alembic.
- **Index ANN**: HNSW cu `vector_cosine_ops` (m=16, ef_construction=64),
  creat la prima migrație.
- **Loaders**: `pypdf` (PDF), `docx2txt` (DOCX), citire directă (TXT/MD).
- **Chunker**: `langchain-text-splitters.RecursiveCharacterTextSplitter`
  (default 800/100).

## Structura

```
project/
├── docker-compose.yml          # Postgres 16 + pgvector
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/0001_initial_schema.py   # documents + chunks + HNSW
├── database.py                 # engine, SessionLocal, transaction()
├── models.py                   # Document, DocumentChunk (one-to-many)
├── pipeline.py                 # entry point: process(file) → ExtractionResult
├── agent.py                    # DocumentAnalystAgent (ReAct, Anthropic)
├── demo.py                     # ingestion + Q&A pe sample_docs/
├── extraction/
│   ├── schemas.py              # DocumentExtraction, Invoice, Contract
│   ├── loaders.py              # LOADER_REGISTRY pe extensie
│   ├── chunker.py              # split_text(text, size, overlap)
│   └── extractor.py            # StructuredExtractor (Anthropic forced tool)
├── rag/
│   ├── embedder.py             # singleton SentenceTransformer
│   └── service.py              # RAGService.search / search_with_threshold
├── repositories/
│   ├── document_repository.py  # CRUD pe Document
│   └── chunk_repository.py     # batch insert + similarity_search
├── tools/                      # registry pattern din L2
│   ├── basic_tools.py          # calculator, get_datetime
│   └── rag_tools.py            # search_documents (tool nou)
├── prompts/                    # YAML + Jinja2 (StrictUndefined)
│   ├── planner_rag.yaml
│   ├── classify.yaml
│   ├── extract_invoice.yaml
│   ├── extract_contract.yaml
│   └── summarize.yaml
└── sample_docs/                # contract, factură, NDA — pentru demo
```

## Setup

```bash
cd l4_homework/project

# 1. Postgres + pgvector
docker compose up -d
docker compose ps                       # health=healthy

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Variabile
cp .env.example .env
# editează .env și pune ANTHROPIC_API_KEY

# 4. Migrare schema
alembic upgrade head
```

Migrația 0001 creează:
- extensia `vector`
- tabelele `documents` și `document_chunks` (FK + cascade delete)
- index unic pe `(document_id, chunk_index)`
- index HNSW pe `document_chunks.embedding` cu `vector_cosine_ops`

## Rulare

### Demo end-to-end

```bash
python demo.py
```

Demo-ul ingestă `sample_docs/` (contract, factură, NDA) și pune agentului 4
întrebări de test. Rularea e idempotentă pe filename: a doua rulare va sări
peste documentele deja încărcate.

### Ingestă manuală

```bash
python pipeline.py sample_docs/                  # un folder
python pipeline.py path/to/contract.pdf          # un fișier
python pipeline.py a.pdf b.docx c.txt            # mai multe
```

Pentru fiecare document se salvează un dump JSON cu metadatele extrase în
`output/<stem>.json` (fără full content — doar invoice/contract/summary).

### Agent

```bash
python agent.py
```

Sau direct din Python:

```python
from agent import DocumentAnalystAgent

agent = DocumentAnalystAgent()
print(agent.run("Ce clauze de reziliere avem?", verbose=True))
```

## Cum funcționează — pe componente

### 1. Extraction Pipeline (L3)

`Pipeline.process(file)`:

```python
docs   = load(file)                              # registry pe extensie
chunks = split_text(docs, 800, 100)              # RecursiveCharacterTextSplitter
extr   = extractor.extract(filename, content)    # LLM cu forced tool -> Invoice/Contract
emb    = embedder.encode(chunks)                 # sentence-transformers, batch
repo.save(extr, chunks, emb)                     # tot într-o singură tranzacție
```

Pași-cheie:
- **Loader registry**: `@register_loader(".pdf", ".docx", ...)` adaugă
  în `LOADER_REGISTRY` (dict pe extensie). `load_document(path)` face
  dispatch pe `path.suffix.lower()`.
- **Idempotență pe filename**: dacă există deja un Document cu același
  filename, ingestion-ul îl sare.
- **Structured output cu Anthropic**: SDK-ul Anthropic nu are
  `with_structured_output`. Trick-ul folosit aici e `tool_choice = "tool"`
  cu un singur tool al cărui `input_schema` e
  `Pydantic.model_json_schema()`. Modelul *trebuie* să apeleze tool-ul, iar
  argumentele sunt validate apoi prin Pydantic.

### 2. PostgreSQL + Repository Pattern (L4)

- `Document` (one) ↔ `DocumentChunk` (many), cascade delete pe `document_id`.
- `transaction()` — context manager care face commit / rollback / close
  automat.
- `DocumentRepository.create()` folosește `flush()` (nu `commit()`) ca să
  populeze `doc.id` înainte ca `ChunkRepository.create_chunks_batch()` să
  insereze chunks în aceeași tranzacție. Commit-ul se face la ieșirea din
  `with transaction()`.

### 3. RAG cu Embeddings (L4)

- `Embedder` cache-uiește modelul sentence-transformers ca singleton de proces
  (~80 MB, scump să-l reinstanțiezi).
- `ChunkRepository.similarity_search(emb, top_k)` ordonează după cosine
  distance (operatorul `<=>` din pgvector, traducerea SQLAlchemy:
  `embedding.cosine_distance(...)`) și calculează `1 - cos_dist` ca
  `score` în [0, 1]. `joinedload(DocumentChunk.document)` evită N+1.
- `RAGService.search_with_threshold()` taie chunks-urile cu scor sub 0.35
  (default — configurabil prin tool).
- Indexul HNSW e creat în migrația inițială, deci search-ul nu face seq
  scan nici la primul query.

### 4. Conectare la agent (L1-L2)

- Tool-ul `search_documents` e un Pydantic-validated wrapper peste
  `RAGService.search_with_threshold`. Tot ce face e să deschidă o
  tranzacție, să caute, și să serializeze rezultatele într-un string
  (`render_context`) — format-ul așteptat de Anthropic Messages API
  pentru `tool_result`.
- `DocumentAnalystAgent` este aceeași clasă ReAct din L2 (Think → Act →
  Observe), dar cu un system prompt diferit (`planner_rag.yaml`) care îi
  spune să apeleze ÎNTÂI `search_documents` și să citeze sursele.
- Tool registry-ul e shared pentru `calculator`, `get_datetime` și
  `search_documents` — same pattern, same `ToolWrapper.catalog()`.

## Exemplu output

După `python demo.py` (cu sample_docs ingest-ate):

```
>>> Ce clauze de reziliere avem în contracte și care e preavizul?
<<< Conform contract_servicii.txt (chunk 3), oricare parte poate denunța
unilateral contractul cu preaviz scris de 30 de zile calendaristice...
Pentru NDA (nda_acord.txt, chunk 1), preavizul este de 15 zile, dar
obligațiile de confidențialitate supraviețuiesc rezilierii încă 3 ani.
```

## Note de design

- **Anthropic vs Gemini**: Lectorul a folosit Gemini cu `response_schema`
  în slide-uri. Am ales Anthropic pentru continuitate cu L1-L2 — același
  client, același API, același pattern de cache. Trick-ul cu
  `tool_choice = "tool"` produce același efect ca `response_schema`:
  output forțat conform unei scheme JSON.
- **Migrații Alembic vs `create_all`**: Am preferat Alembic + migrație
  explicită (slide 55) pentru că `Base.metadata.create_all()` nu creează
  extensia `vector` și nici indexul HNSW — și ambele sunt critice.
- **HNSW de la prima migrație**: indexul HNSW se creează inline în
  `0001_initial_schema.py`, nu lazy. Pentru un dataset de produse real
  aș folosi `CREATE INDEX CONCURRENTLY` într-o migrație separată ca să nu
  blochez tabela, dar pentru homework-ul ăsta varianta inline e mai
  ușor de verificat.
- **Threshold 0.35**: cu `paraphrase-multilingual-MiniLM-L12-v2`,
  scorurile pe limba română stau de obicei între 0.4 și 0.7 pentru
  query-uri relevante. Threshold-ul e configurabil prin tool, deci LLM-ul
  îl poate ajusta la nevoie.

## Reset complet

```bash
docker compose down -v   # șterge volumul → pierzi datele
docker compose up -d
alembic upgrade head
```
