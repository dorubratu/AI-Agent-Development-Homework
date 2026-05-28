# Document Analyst cu RAG

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
- **ORM + migrations**: SQLAlchemy 2.0 + Alembic.
- **Index ANN**: HNSW cu `vector_cosine_ops` (m=16, ef_construction=64),
  creat la prima migrare.
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

docker compose up -d
docker compose ps       # check health

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # add ANTHROPIC_API_KEY to .env

alembic upgrade head
```

### Demo end-to-end

```bash
python demo.py
```

Demo proceseazǎ `sample_docs/` (contract, factură, NDA) și pune agentului 4
întrebări de test. A doua rulare va sări peste documentele deja încărcate.

### Manual

```bash
python pipeline.py sample_docs/                  # un folder 
python pipeline.py path/to/contract.pdf          # un fișier
python pipeline.py a.pdf b.docx c.txt            # mai multe documente
```

Pentru fiecare document se salvează un dump JSON cu metadatele extrase în
`output/<stem>.json` (fără full content, doar invoice/contract/summary).

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

## Componente

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

## Reset complet

```bash
docker compose down -v  
docker compose up -d
alembic upgrade head
```
