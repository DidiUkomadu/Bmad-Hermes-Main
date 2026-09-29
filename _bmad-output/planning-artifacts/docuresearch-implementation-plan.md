# DocuResearch — Implementation Plan

**Status:** Draft v1 — implementation-planning stage  
**Last updated:** 2026-09-21  
**Source of truth:** DocuResearch PRFAQ (finalized) · Architecture v1 · Epics & Stories (finalized)  
**Implementation start:** Not yet — this document is the planning artifact  
**Implementation language:** Python 3.12  
**Application framework:** FastAPI  
**Storage:** SQLite (local, file-based)  
**MVP document formats:** text-based PDF, Markdown, TXT  
**Out of MVP scope:** DOCX, scanned PDFs/OCR, dedicated contradiction detection, reranking, authentication, billing, multi-tenancy, numerical confidence scores  

---

## 1. Repository / Project Structure

### Top-level layout

```
docuresearch/
├── app/                      # FastAPI application package
│   ├── __init__.py
│   ├── main.py               # FastAPI app factory, router registration, lifespan
│   ├── config.py             # Settings (Pydantic/Attrs), validated at startup
│   ├── api/                  # API route handlers
│   │   ├── __init__.py
│   │   ├── documents.py      # document upload, list, removal
│   │   ├── query.py          # query + follow-up
│   │   └── citations.py     # passage/citation lookup
│   ├── ingestion/            # document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # routes uploaded file → format handler → chunker → store
│   │   ├── pdf.py            # text-based PDF extraction (page-preserving)
│   │   ├── markdown.py       # Markdown block parsing with heading structure
│   │   └── text.py           # plain text paragraph splitting
│   ├── chunking/             # passage creation from ingestion output
│   │   ├── __init__.py
│   │   └── strategy.py       # configurable chunker: size, overlap, boundary preference
│   ├── store/                # SQLite content store
│   │   ├── __init__.py
│   │   ├── schema.py         # table definitions, migrations / initial schema
│   │   └── repository.py     # document and passage CRUD operations
│   ├── retrieval/            # retrieval pipeline
│   │   ├── __init__.py
│   │   ├── interface.py      # abstraction: text → vector; query + index → ranked IDs
│   │   ├── semantic.py       # embedding-based retrieval (all-MiniLM-L6-v2 default)
│   │   ├── keyword.py        # BM25 retrieval (rank_bm25 default)
│   │   └── hybrid.py         # combine semantic + keyword with configurable weighting
│   ├── generation/           # LLM-backed answer generation
│   │   ├── __init__.py
│   │   ├── interface.py      # model-agnostic LLM contract
│   │   ├── prompt.py         # prompt construction (8-section structure)
│   │   └── schema.py         # structured output: answer, citations, evidence quality
│   ├── conversation/         # conversation handling
│   │   ├── __init__.py
│   │   └── session.py        # session storage, history management, 5-turn default limit
│   └── citation/             # citation resolution (backend side)
│       ├── __init__.py
│       └── resolver.py       # passage ID → passage text + metadata
├── evaluation/               # evaluation dataset and runner (parallel workstream)
│   ├── __init__.py
│   ├── schema.py             # evaluation question / result data models
│   ├── runner.py             # evaluation runner: submit question, collect answer, compare
│   ├── methodology.md        # evaluation methodology document (story 0.1 output)
│   └── dataset/              # living dataset directory
│       ├── documents/        # curated representative documents (PDFs, MD, TXT)
│       ├── questions.json    # curated questions with gold answers and source locations
│       └── results/          # evaluation run results, timestamped
├── tests/                    # test suite
│   ├── __init__.py
│   ├── conftest.py           # shared fixtures (app, test DB, test documents)
│   ├── unit/                 # unit tests by module
│   │   ├── test_config.py
│   │   ├── test_chunking.py
│   │   ├── test_store.py
│   │   ├── test_retrieval.py
│   │   ├── test_generation.py
│   │   └── test_conversation.py
│   ├── integration/          # integration tests
│   │   ├── test_ingestion.py
│   │   ├── test_retrieval_integration.py
│   │   └── test_api.py
│   └── evaluation/           # evaluation-runner tests
│       └── test_runner.py
├── data/                     # local data directory (git-ignored)
│   ├── storage/              # SQLite database file(s)
│   ├── embeddings/           # embedding cache, if any
│   └── uploads/              # staging for uploaded files (transient)
├── config.example.toml       # example configuration (not secret, checked in)
├── pyproject.toml            # project metadata, dependencies, tool config
├── .gitignore
└── README.md
```

### Package boundaries

- `app.config` — all runtime configuration, validated at startup. No other module imports from config in a way that bypasses the settings object.
- `app.ingestion` — format handlers and orchestration. Produces passages in the data model defined in Section 4. Consumes nothing from retrieval or generation.
- `app.chunking` — operates on ingestion output. Configurable; does not depend on storage or retrieval.
- `app.store` — SQLite content store. Owns the schema. Provides repository functions consumed by ingestion (write), retrieval (read passages by ID, list passages for a document), and citation resolution.
- `app.retrieval` — retrieval pipeline. Consumes passages from the store (or an indexed representation built from them). Produces ranked passage IDs. Does not depend on generation.
- `app.generation` — LLM-backed generation. Consumes retrieval results and conversation context. Produces structured answers. Does not depend on the API layer.
- `app.conversation` — session and history management. Consumed by generation (context) and the API layer (session lifecycle).
- `app.api` — FastAPI route handlers. Thin layer: validates requests, calls domain functions, returns responses. No business logic.
- `evaluation` — separate package, parallel workstream. Owns the dataset, runner, and methodology. Consumes the API (or component interfaces) for evaluation runs. Does not affect runtime application code.

---

## 2. Component Contracts / Interfaces

### 2.1 Ingestion → chunking

**Producer:** `app.ingestion` (format handlers: pdf.py, markdown.py, text.py; orchestrator.py)  
**Consumer:** `app.chunking.strategy`  

**Contract:** Ingestion produces a `RawDocument` structure:

```
RawDocument
  - document_name: str
  - format: "pdf" | "markdown" | "txt"
  - source_text: str                 # full extracted text, cleaned
  - pages: list[Page] | None         # PDF only: list of {page_number, text}
  - blocks: list[Block] | None       # Markdown only: list of {level, heading, text}
  - paragraphs: list[str] | None     # TXT only: list of paragraph strings
  - upload_timestamp: datetime
```

The chunker accepts a `RawDocument` and a `ChunkingConfig`, and produces a list of `Passage` objects (see Section 4). The chunker does not need to know how the RawDocument was produced — it only needs the structured content and metadata.

**Invariance:** Every `Passage` produced must carry a stable identifier, the document reference, a location string, start/end offsets within `source_text`, and the passage text. No passage may be produced without all of these fields.

### 2.2 Chunking → content store

**Producer:** `app.chunking.strategy`  
**Consumer:** `app.store.repository`  

**Contract:** The chunker produces a list of `Passage` objects. The store's repository accepts these and persists them alongside the `Document` record.

```
Passage (as produced by chunking, persisted by store)
  - id: str                          # stable passage identifier
  - document_id: str                 # references Document.id
  - text: str                        # passage text
  - location: str                    # page number, section path, or paragraph index
  - start_offset: int                # character offset within RawDocument.source_text
  - end_offset: int                  # character offset within RawDocument.source_text
  - embedding: list[float] | None    # computed at indexing time; stored when available
```

The store assigns or validates the `id`. The store is responsible for ensuring that a passage's `document_id` matches an existing document.

### 2.3 Content store → retrieval

**Producer:** `app.store.repository`  
**Consumer:** `app.retrieval` (semantic.py, keyword.py, hybrid.py)  

**Contract:** Retrieval needs access to passage text and metadata for indexing and for resolving passage IDs. The store provides:

- `get_document(doc_id) → Document | None`
- `list_passages_for_document(doc_id) → list[Passage]`
- `get_passage(passage_id) → Passage | None`
- `list_all_passages() → list[Passage]` (used to build the retrieval index)

Retrieval does not mutate the store. The store does not depend on retrieval.

**Indexing contract:** Retrieval builds and maintains its own index structures (embedding index, BM25 index) from passage data. When passages are added or removed, the retrieval index must be updated. The mechanism for this (e.g. a callback, an explicit reindex, or an event) is an implementation detail; the contract is that retrieval sees a consistent view of the passages it has indexed.

### 2.4 Semantic retrieval + BM25 → hybrid retrieval

**Producers:** `app.retrieval.semantic`, `app.retrieval.keyword`  
**Consumer:** `app.retrieval.hybrid`  

**Semantic contract:**

```
SemanticRetriever (abstract)
  - embed(text: str) → list[float]
  - index_passages(passages: list[Passage]) → None
  - remove_passages(passage_ids: list[str]) → None
  - search(query: str, top_n: int) → list[RetrievalResult]
```

`RetrievalResult`:
```
RetrievalResult
  - passage_id: str
  - score: float                     # normalized 0–1 semantic similarity
  - index_type: "semantic"
```

**BM25 contract:**

```
KeywordRetriever (abstract)
  - index_passages(passages: list[Passage]) → None
  - remove_passages(passage_ids: list[str]) → None
  - search(query: str, top_n: int) → list[RetrievalResult]
```

`RetrievalResult` for keyword:
```
RetrievalResult
  - passage_id: str
  - score: float                     # normalized 0–1 BM25 score
  - index_type: "keyword"
```

**Hybrid contract:**

```
HybridRetriever
  - search(query: str, scope: DocumentScope, top_n: int, weighting: HybridWeighting) → list[RetrievalResult]
```

`DocumentScope`:
```
DocumentScope
  - mode: "all" | "specific"
  - document_id: str | None          # present when mode == "specific"
```

`HybridWeighting`:
```
HybridWeighting
  - semantic_weight: float           # default 0.5
  - keyword_weight: float            # default 0.5
```

Hybrid search combines results from both retrievers, normalizes scores to a common 0–1 scale, applies the weighting, applies scope filtering, and returns the top N results. The combined `RetrievalResult` carries a `score` that is the weighted combination.

**Invariance:** Both retrievers must return `passage_id`s that exist in the store. Hybrid must not return duplicate `passage_id`s (deduplicate by taking the higher combined score).

### 2.5 Retrieval → generation

**Producer:** `app.retrieval.hybrid`  
**Consumer:** `app.generation` (prompt.py, interface.py)  

**Contract:** Generation receives a `RetrievedContext`:

```
RetrievedContext
  - query: str
  - scope: DocumentScope
  - passages: list[RetrievedPassage]
  - conversation_history: list[ConversationTurn] | None
  - retrieval_latency_seconds: float
```

```
RetrievedPassage
  - passage_id: str
  - document_name: str
  - location: str
  - text: str
  - score: float
```

Generation does not call the store or retrieval directly. It receives the `RetrievedContext` from the caller (the API layer or evaluation runner). This makes generation testable in isolation — a test can construct a `RetrievedContext` with known passages and verify the generated answer.

### 2.6 Generation → citation resolution

**Producer:** `app.generation`  
**Consumer:** `app.citation.resolver`, `app.api`  

**Contract:** Generation produces a `GeneratedAnswer`:

```
GeneratedAnswer
  - answer_text: str
  - citations: list[Citation]        # each citation has a passage_id
  - evidence_quality: EvidenceQuality
  - evidence_quality_narrative: str
  - is_abstention: bool
  - generation_latency_seconds: float
```

```
Citation
  - passage_id: str
  - document_name: str
  - location: str
  - excerpt: str                     # actual passage text from the store
```

**Resolution contract:** Citation resolution is the process of verifying that each `Citation.passage_id` exists in the store and populating `Citation.excerpt` from the store's record for that passage. This is done by `app.citation.resolver` using `app.store.repository.get_passage()`. If a citation's `passage_id` does not exist in the store, that is an error condition — the generation layer must not produce citations for non-existent passages.

The generation layer produces `Citation.passage_id`, `document_name`, and `location` from the `RetrievedPassage` data it received. The `excerpt` is populated by the resolver from the store, not generated by the LLM.

### 2.7 Conversation → retrieval/generation

**Producer:** `app.conversation.session`  
**Consumer:** `app.retrieval`, `app.generation`, `app.api`  

**Contract:** A conversation session has a `session_id`. The session store holds a list of `ConversationTurn`:

```
ConversationTurn
  - turn_index: int                  # 0-based, sequential within a session
  - user_query: str
  - answer_text: str
  - citations: list[Citation]
  - evidence_quality: EvidenceQuality
  - is_abstention: bool
  - created_at: datetime
```

The conversation module provides:

- `create_session() → Session`
- `add_turn(session_id, turn) → None`
- `get_history(session_id, max_turns: int) → list[ConversationTurn]`  # applies pruning limit
- `prune_history(session_id) → None`  # called when max exceeded

The retrieval and generation layers receive conversation history as a list of `ConversationTurn` objects included in the `RetrievedContext`. They do not manage session state themselves.

### 2.8 Backend ↔ UI/API

**Producer:** `app.api`  
**Consumer:** external client (UI, evaluation runner)  

The API is the sole external interface. See Section 5 for full contracts. The API consumes domain functions from ingestion, store, retrieval, generation, conversation, and citation resolver. It does not contain business logic.

### 2.9 System ↔ evaluation runner

**Producer:** `evaluation.runner`  
**Consumer:** the application (API or component interfaces)  

The evaluation runner is a separate package that can run against the API (end-to-end) or against individual component interfaces (component-level evaluation). It consumes the evaluation dataset (questions, gold answers, source locations) and produces `EvaluationResult` objects.

**Contract:** The runner must be able to:

- Load questions from the dataset.
- Submit a query to the system (via API or directly to retrieval + generation).
- Collect the system's answer, citations, and metadata.
- Compare against the gold answer and source locations.
- Compute metrics and produce a result.

The runner does not modify the application's data. It may use a separate test database or a controlled test environment.

---

## 3. Core Data Models / Entities

### 3.1 Document

```
Document
  - id: str                          # UUID4, primary key
  - name: str                        # user-visible document name
  - format: str                      # "pdf" | "markdown" | "txt"
  - uploaded_at: datetime
  - page_count: int | None           # PDF: number of pages; other formats: None
  - section_count: int | None        # Markdown: number of top-level sections; other: None
  - status: str                      # "available" | "processing" | "failed"
  - error_message: str | None
```

### 3.2 Passage

```
Passage
  - id: str                          # stable passage identifier (see Section 4)
  - document_id: str                 # FK to Document.id
  - text: str                        # passage text (the citable unit)
  - location: str                    # page number, section path, or paragraph index
  - start_offset: int                # char offset within cleaned source text
  - end_offset: int                  # char offset within cleaned source text
  - embedding: list[float] | None    # vector representation; stored when computed
  - indexed_at: datetime | None
```

### 3.3 Citation

```
Citation
  - passage_id: str                  # references Passage.id
  - document_name: str               # human-readable document name (from Document.name)
  - location: str                    # from Passage.location
  - excerpt: str                     # actual passage text from the store
```

Citation is a view-model, not a persisted entity. It is produced by generation (from RetrievedPassage data) and populated by the citation resolver (excerpt from the store).

### 3.4 RetrievalResult (runtime)

```
RetrievalResult
  - passage_id: str
  - score: float                     # 0–1 combined hybrid score
  - index_type: str                  # "semantic" | "keyword" | "hybrid"
```

### 3.5 EvidenceQuality

```
EvidenceQuality (enumeration)
  - SUFFICIENT                       # documents support the answer directly
  - PARTIAL                          # documents support part of the answer
  - INSUFFICIENT                     # documents do not contain enough to answer
  - CONFLICTING                      # documents disagree on the point
```

This is the same enumeration from the PRFAQ and Architecture. It is used as the evidence quality field in `GeneratedAnswer` and `ConversationTurn`.

### 3.6 Answer

```
Answer (runtime / API response)
  - answer_text: str
  - citations: list[Citation]
  - evidence_quality: EvidenceQuality
  - evidence_quality_narrative: str
  - is_abstention: bool
  - generation_latency_seconds: float
  - retrieval_latency_seconds: float
  - total_latency_seconds: float
```

### 3.7 Conversation / Session

```
Session
  - session_id: str                  # UUID4
  - created_at: datetime
  - current_document_scope: DocumentScope | None  # scope from the most recent query

ConversationTurn
  - turn_index: int
  - user_query: str
  - answer: Answer                   # full answer object from the generation layer
  - created_at: datetime
```

### 3.8 EvaluationQuestion

```
EvaluationQuestion
  - id: str                          # stable question identifier
  - question_text: str
  - question_type: str               # "single_source" | "multi_passage" | "partial" | "insufficient" | "conflicting"
  - gold_answer: str | None          # expected answer text; None for insufficient-evidence questions
  - gold_passage_ids: list[str]      # passage IDs that support the gold answer
  - conflicting_sources: list[ConflictingSource] | None
  - closest_relevant_passage_id: str | None  # for insufficient-evidence questions
  - metadata: dict | None            # free-form notes (e.g. which document, what to look for)

ConflictingSource
  - source_label: str                # e.g. "Document A", "Section 3 of Document B"
  - claim: str                       # what this source says
  - passage_id: str                  # passage containing this claim
```

### 3.9 EvaluationResult

```
EvaluationResult
  - run_id: str                      # UUID4, identifies this evaluation run
  - run_timestamp: datetime
  - question_id: str
  - question_type: str
  - system_answer_text: str
  - system_citations: list[Citation]
  - system_evidence_quality: str
  - system_is_abstention: bool
  - retrieval_precision: float | None
  - retrieval_recall: float | None
  - citation_correctness: float | None
  - answer_faithful: bool | None
  - was_abstention_correct: bool | None
  - conflict_handled_correctly: bool | None
  - total_latency_seconds: float
  - notes: str | None
```

---

## 4. Stable Identifiers and Metadata Requirements

### 4.1 Document IDs

- **Type:** UUID4, generated at document upload time.
- **Stability:** Stable for the lifetime of the document in the system. Deleted documents' IDs must not be reused.
- **Usage:** Used in the API (document endpoints), in passage references, in evaluation dataset source references.

### 4.2 Passage IDs

- **Type:** `doc_id:passage_index` where `doc_id` is the document UUID and `passage_index` is a sequential integer within that document (0-based, assigned in passage creation order).
- **Example:** `a1b2c3d4-...:0`, `a1b2c3d4-...:1`
- **Stability:** Stable for the lifetime of the passage. If a document is re-ingested, new passages get new IDs; old IDs are not reused.
- **Usage:** Used in citations, retrieval results, evaluation gold references, citation lookup endpoint.

**Rationale:** The `doc_id:passage_index` format is human-readable in logs and debuggable, unambiguous across documents, and stable as long as the document exists. It does not encode location (which can vary with chunking strategy) — location is a separate field.

### 4.3 Location identifiers

Location is format-specific and recorded as a human-readable string:

- **PDF:** `p.<page_number>` (e.g. `p.5`). Page numbers are 1-based.
- **Markdown:** Section path derived from heading hierarchy, e.g. `§3` for a top-level heading "3.", `§3.2` for a second-level heading under "3.". The exact format is: take the heading hierarchy from the Markdown parser, build a dotted path from heading numbers or heading text. The format must be consistent within a document and derivable from the parsing output.
- **TXT:** `¶<paragraph_index>` (e.g. `¶12`). Paragraph indices are 0-based or 1-based; the choice is consistent within the document and documented.

**Invariance:** A location string must be sufficient to identify the passage in the original document for a human reviewer. It does not need to be machine-navigable to a character offset (that is what `start_offset`/`end_offset` are for).

### 4.4 Offsets

- **Type:** Non-negative integers, character offsets within the cleaned/processed source text (`RawDocument.source_text`).
- **start_offset:** inclusive start of the passage text within the source text.
- **end_offset:** exclusive end of the passage text within the source text.
- **Invariance:** `source_text[start_offset:end_offset] == passage.text` must hold for every passage. This is testable.

### 4.5 Citation resolution requirement

A citation in an answer is resolvable if and only if:
1. `Citation.passage_id` exists in the store as a `Passage.id`.
2. `Citation.document_name` matches the `Document.name` for the document that contains that passage.
3. `Citation.location` matches the `Passage.location` for that passage.
4. `Citation.excerpt` is the exact `Passage.text` from the store.

The citation resolver enforces (1)–(4) at citation resolution time. The evaluation runner uses these same checks for citation correctness.

### 4.6 Metadata requirements summary

| Entity | Required metadata | Notes |
|---|---|---|
| Document | id, name, format, uploaded_at, page_count, section_count, status | page_count and section_count are format-specific; None for formats that don't have them |
| Passage | id, document_id, text, location, start_offset, end_offset, embedding | embedding is optional at passage creation; computed at indexing time |
| Citation | passage_id, document_name, location, excerpt | excerpt populated from store at resolution time; not generated by LLM |
| Answer | answer_text, citations, evidence_quality, evidence_quality_narrative, is_abstention, latencies | latencies: retrieval, generation, total |
| EvaluationQuestion | id, question_text, question_type, gold_answer, gold_passage_ids, conflicting_sources, closest_relevant_passage_id | gold_answer is None for insufficient-evidence questions |

---

## 5. API Contracts

All endpoints return JSON. All endpoints are unauthenticated (single-user, local-first MVP). Base path: `/api/v1`.

### 5.1 Document upload

```
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Form fields:
  - file: the document file (PDF, Markdown, or TXT)
  - name: optional user-visible name; if absent, derived from the filename

Response 202 Accepted:
{
  "document_id": "<UUID4>",
  "name": "<document name>",
  "format": "<pdf|markdown|txt>",
  "status": "processing",
  "uploaded_at": "<ISO 8601>"
}

Response 200 OK (when processing completes synchronously — implementation choice):
{
  "document_id": "<UUID4>",
  "name": "<document name>",
  "format": "<pdf|markdown|txt>",
  "status": "available",
  "page_count": <int|null>,
  "section_count": <int|null>,
  "uploaded_at": "<ISO 8601>"
}

Response 400 Bad Request:
{
  "error": "Invalid file format",
  "detail": "..."
}

Response 422 Unprocessable Entity:
{
  "error": "Extraction failed",
  "detail": "..."
}
```

**Notes:** Upload triggers ingestion (epic 1, story 1.5). The status field reflects whether ingestion has completed. If ingestion is synchronous for the MVP, the endpoint returns 200 with status "available". If asynchronous, returns 202 and the client polls or waits. The implementation choice is an engineering decision, not a product requirement.

### 5.2 Document listing

```
GET /api/v1/documents

Response 200 OK:
{
  "documents": [
    {
      "document_id": "<UUID4>",
      "name": "<document name>",
      "format": "<pdf|markdown|txt>",
      "uploaded_at": "<ISO 8601>",
      "page_count": <int|null>,
      "section_count": <int|null>,
      "status": "available"
    },
    ...
  ]
}
```

### 5.3 Document removal

```
DELETE /api/v1/documents/{document_id}

Response 200 OK:
{
  "document_id": "<UUID4>",
  "status": "removed"
}

Response 404 Not Found:
{
  "error": "Document not found"
}
```

Removal deletes the document and all its passages from the store. After removal, the document is no longer listable and its passages are not retrievable.

### 5.4 Query

```
POST /api/v1/query
Content-Type: application/json

Request body:
{
  "question": "<question text>",
  "document_scope": {
    "mode": "all" | "specific",
    "document_id": "<UUID4>"        # required when mode == "specific"
  },
  "session_id": "<UUID4>"           # optional; for conversation context
}

Response 200 OK:
{
  "answer": {
    "answer_text": "<generated answer>",
    "citations": [
      {
        "passage_id": "<doc_id:passage_index>",
        "document_name": "<document name>",
        "location": "<location string>",
        "excerpt": "<passage text>"
      }
    ],
    "evidence_quality": "sufficient" | "partial" | "insufficient" | "conflicting",
    "evidence_quality_narrative": "<narrative>",
    "is_abstention": false,
    "retrieval_latency_seconds": 0.123,
    "generation_latency_seconds": 0.456,
    "total_latency_seconds": 0.579
  },
  "session_id": "<UUID4>"
}
```

**Notes:** The `document_scope` is optional; when absent, defaults to `"all"`. The `session_id` is optional; when provided, the conversation turn is recorded. When absent, the query is treated as a standalone question.

### 5.5 Citation / passage lookup

```
GET /api/v1/citations/{passage_id}

Response 200 OK:
{
  "passage_id": "<doc_id:passage_index>",
  "document_id": "<UUID4>",
  "document_name": "<document name>",
  "location": "<location string>",
  "text": "<passage text>",
  "start_offset": <int>,
  "end_offset": <int>
}

Response 404 Not Found:
{
  "error": "Passage not found"
}
```

### 5.6 Conversation / follow-up

```
POST /api/v1/conversation/{session_id}/follow-up
Content-Type: application/json

Request body:
{
  "question": "<follow-up question>",
  "document_scope": {
    "mode": "all" | "specific",
    "document_id": "<UUID4>"        # optional; changes scope if provided
  }
}

Response 200 OK: same shape as query response, plus the session_id.

Response 404 Not Found if session_id does not exist.
```

Follow-up records a new turn in the session and returns the answer. The conversation history is included in the generation context automatically.

### 5.7 Session creation (explicit, for clients that want a fresh session)

```
POST /api/v1/conversation
Content-Type: application/json

Request body:
{
  "initial_document_scope": {
    "mode": "all" | "specific",
    "document_id": "<UUID4>"
  }
}

Response 201 Created:
{
  "session_id": "<UUID4>",
  "created_at": "<ISO 8601>"
}
```

### 5.8 API design invariants

- No endpoint accepts or returns a numerical confidence score.
- No endpoint requires authentication.
- All endpoints return clear error responses for invalid input, missing resources, or processing failures.
- The `answer` object in query/follow-up responses is identical in shape to the `GeneratedAnswer` domain model (Section 3.6), with `Citation` objects (Section 3.3).

---

## 6. SQLite Storage Schema

### 6.1 Documents table

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,              -- UUID4
    name TEXT NOT NULL,
    format TEXT NOT NULL CHECK (format IN ('pdf', 'markdown', 'txt')),
    uploaded_at TEXT NOT NULL,        -- ISO 8601
    page_count INTEGER,
    section_count INTEGER,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'available', 'failed')),
    error_message TEXT
);
```

### 6.2 Passages table

```sql
CREATE TABLE passages (
    id TEXT PRIMARY KEY,              -- doc_id:passage_index
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    location TEXT NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
    embedding TEXT,                   -- JSON array of floats, or NULL
    indexed_at TEXT
);
```

Index:
```sql
CREATE INDEX idx_passages_document_id ON passages(document_id);
CREATE INDEX idx_passages_embedding ON passages(embedding);  -- for embedding lookups, if needed
```

**Notes:** `ON DELETE CASCADE` ensures that deleting a document removes all its passages. `embedding` is stored as a JSON array of floats (e.g. `[0.123, -0.456, ...]`) to avoid requiring a vector extension for the MVP. The BM25 index and semantic index are built in-memory or in a separate structure by the retrieval layer, not in SQLite.

### 6.3 Conversations table

```sql
CREATE TABLE conversations (
    session_id TEXT PRIMARY KEY,      -- UUID4
    created_at TEXT NOT NULL,
    current_document_scope TEXT       -- JSON encoding of DocumentScope, or NULL
);
```

### 6.4 Conversation turns table

```sql
CREATE TABLE conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    turn_index INTEGER NOT NULL,
    user_query TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    citations TEXT NOT NULL,          -- JSON array of Citation objects
    evidence_quality TEXT NOT NULL CHECK (evidence_quality IN ('sufficient', 'partial', 'insufficient', 'conflicting')),
    is_abstention INTEGER NOT NULL CHECK (is_abstention IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(session_id, turn_index)
);
```

**Notes:** `ON DELETE CASCADE` ensures that deleting a session removes its turns. `citations` is stored as a JSON array. `turn_index` is managed by the conversation module and enforced as unique per session.

### 6.5 Schema versioning

The initial schema is created on first startup (if the database file does not exist). For the MVP, a simple "create tables if not exist" approach is acceptable. If schema changes are needed later, a basic migration mechanism (version table + incremental migrations) should be added. This is not an MVP requirement but should be considered when the schema is first defined.

---

## 7. Ingestion Interfaces

### 7.1 PDF ingestion

**Function signature:**
```python
def extract_text_from_pdf(file_content: bytes, document_name: str) -> RawDocument:
```

**Returns:** `RawDocument` with `format="pdf"`, `source_text` containing the full extracted text, `pages` populated with `{page_number, text}` for each page.

**Behavior:**
- Extracts text page by page, preserving page boundaries.
- Records page number for each page's text.
- `source_text` is the concatenation of all page texts (with page boundaries recorded in `pages`).
- Offsets in passages reference positions within `source_text`.
- If the PDF has no text layer (scanned/image PDF), the function returns an error or an empty result — it does not attempt OCR.
- If the PDF is corrupt or unreadable, returns a clear error.

**Failure mode:** On failure, the orchestrator sets the document status to `"failed"` with an error message. The API returns a 422 or 400 accordingly.

**Libraries (initial direction, not final):** pypdf or pdfplumber — text-based PDF extraction. The choice is recorded as part of story 1.1 delivery.

### 7.2 Markdown ingestion

**Function signature:**
```python
def parse_markdown(file_content: bytes, document_name: str) -> RawDocument:
```

**Returns:** `RawDocument` with `format="markdown"`, `source_text` containing the full text, `blocks` populated with `{level, heading, text}` for each block.

**Behavior:**
- Parses the Markdown into blocks: headings (with level), paragraphs, code blocks, lists.
- Records heading structure: each heading has a level (1, 2, 3, ...) and text. Section locators are derived from the heading hierarchy.
- `source_text` is the concatenated text of all blocks.
- Offsets in passages reference positions within `source_text`.

**Block model:**
```python
Block = dict(level=None, heading=None, block_type="heading"|"paragraph"|"code"|"list", text=str)
```

For headings, `level` is the heading level (1–6) and `heading` is the heading text. For non-headings, `level` and `heading` are None.

**Libraries (initial direction):** A Python Markdown parser that exposes block structure. The specific choice is recorded as part of story 1.2 delivery.

### 7.3 Plain text ingestion

**Function signature:**
```python
def parse_plain_text(file_content: bytes, document_name: str) -> RawDocument:
```

**Returns:** `RawDocument` with `format="txt"`, `source_text` containing the full text, `paragraphs` populated with a list of paragraph strings.

**Behavior:**
- Splits the text on paragraph boundaries (blank-line-separated groups).
- `source_text` is the full text.
- `paragraphs` is the list of paragraph strings, in order.
- Offsets in passages reference positions within `source_text`.
- If paragraph boundaries are ambiguous, falls back to a line-group or fixed-size split — documented in the function.

**Libraries:** Standard Python text handling. No external library required.

### 7.4 Orchestration

**Function signature:**
```python
def ingest_document(file_content: bytes, file_name: str, document_name: str | None) -> Document:
```

**Behavior:**
- Detects format from file extension and/or content inspection.
- Routes to the correct format handler.
- Passes the `RawDocument` to the chunker.
- Passes the resulting passages to the store.
- Returns the `Document` record with status `"available"` (or `"failed"` on error).

**Format detection:**
- Priority: file extension (`.pdf`, `.md`, `.markdown`, `.txt`).
- Fallback: content inspection (e.g. Markdown headers, PDF magic bytes) if extension is ambiguous.
- Unrecognized format: raise an error that the orchestrator converts to a `"failed"` status.

---

## 8. Configurable Retrieval Parameters

All retrieval parameters are configurable via the application settings (Section 1, `app/config.py`). They have defaults as specified in the Epics & Stories decision-resolution pass, and are explicitly adjustable.

### 8.1 Embedding model

```
retrieval.embedding_model = "all-MiniLM-L6-v2"   # model name identifier
retrieval.embedding_device = "cpu"                # or "cuda" if available
retrieval.embedding normalize = True              # normalize embeddings to unit length
```

**Default:** all-MiniLM-L6-v2 via sentence-transformers, CPU, normalized.  
**Swappability:** The `SemanticRetriever` interface (Section 2.4) insulates the rest of the system. Changing the model is a configuration change plus, if the new model has different output dimensions, a re-indexing of passages.

### 8.2 BM25 implementation

```
retrieval.keyword_library = "rank_bm25"
retrieval.keyword_k1 = 1.5                        # BM25 k1 parameter
retrieval.keyword_b = 0.75                        # BM25 b parameter
```

**Default:** rank_bm25, k1=1.5, b=0.75.  
**Swappability:** The `KeywordRetriever` interface (Section 2.4) insulates the rest of the system.

### 8.3 Hybrid weighting

```
retrieval.hybrid_semantic_weight = 0.5
retrieval.hybrid_keyword_weight = 0.5
```

**Default:** 0.5 / 0.5 (equal).  
**Normalization:** Both signals normalized to 0–1 before combination. The combination is `semantic_weight * semantic_score + keyword_weight * keyword_score`.

### 8.4 Candidate set size

```
retrieval.max_candidates = 20
```

**Default:** 20.  
**Behavior:** Hybrid search returns no more than `max_candidates` results. Applies after scope filtering and deduplication.

### 8.5 Chunking parameters

```
chunking.target_length_min = 250      # characters
chunking.target_length_max = 600      # characters
chunking.overlap_min = 80             # characters
chunking.overlap_max = 150            # characters
chunking.prefer_semantic_boundaries = True   # use paragraph/heading boundaries when in range
```

**Defaults:** As specified in Epics & Stories story 1.4 decision.  
**Behavior:** The chunker targets a passage length in `[target_length_min, target_length_max]`, uses overlap in `[overlap_min, overlap_max]`, and prefers semantic boundaries (paragraphs for PDF/TXT, headings for Markdown) when they fall within the target range. When no semantic boundary is available, uses a sized chunk with overlap.

### 8.6 Configuration file

Configuration is loaded from a TOML file (default: `config.toml` in the project root, or a path specified at startup). An `config.example.toml` is checked in with sane defaults and is not secret. Runtime overrides can be applied via environment variables or command-line arguments, following the pattern established by the settings library.

**Security note:** No secrets are required for the MVP (no authentication, no external paid services). The configuration file does not contain secrets in the MVP.

---

## 9. Generation Contract

### 9.1 Model-agnostic LLM interface

```python
class LLMInterface(Protocol):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        ...
```

```
LLMResponse
  - raw_text: str                    # the LLM's raw output
  - structured: GeneratedAnswer | None  # parsed structured output, if parsing succeeded
  - parse_error: str | None          # if parsing failed
  - latency_seconds: float
```

**Contract:** The `LLMInterface` is the only place that knows about the specific LLM provider. The generation layer (`prompt.py`, schema parsing) depends on this interface, not on any specific provider. Swapping the model is implementing a new `LLMInterface` or changing which implementation is instantiated.

**Initial implementation:** Uses the free model available through Hermes for the agent layer, as specified in the PRFAQ and Architecture. The implementation wraps the Hermes agent call in the `LLMInterface` contract.

**Error handling:** If the LLM returns an error or the response cannot be parsed, `LLMResponse.structured` is `None` and `parse_error` is set. The generation layer treats this as a generation failure and returns an appropriate error response (not a fabricated answer).

### 9.2 Prompt construction

**Function signature:**
```python
def build_prompt(context: RetrievedContext, config: GenerationConfig) -> str:
```

**Structure (8 sections, as resolved in Epics & Stories story 4.2):**

1. **Role and task:** State that the model is a research assistant that answers questions using only the provided document passages, with citations and evidence quality narration.
2. **The user's question:** The query text from `context.query`.
3. **Retrieved passages:** Each passage presented with document name, location, and text — clearly labeled as source material.
4. **Document scope context:** Which document(s) were searched.
5. **Conversation context (if any):** Previous turns from `context.conversation_history`.
6. **Answer instructions:** The five requirements from Architecture §6.
7. **Evidence quality instructions:** The four categories with the PRFAQ/architecture language, including the conservative-heuristics instruction.
8. **Output format instructions:** The structured output schema (answer text, citations, evidence quality, abstention flag).

**Invariance:** The prompt does not instruct the model to introduce outside knowledge as if from the documents. The prompt makes the source material (retrieved passages) clearly distinguishable from the instructions.

**Separation:** `build_prompt` is a pure function of `RetrievedContext` and `GenerationConfig`. It does not call the LLM. The LLM call is a separate step. This makes the prompt inspectable and testable.

### 9.3 Structured output

The generation layer parses the LLM's raw output into a `GeneratedAnswer`. The parsing is based on the output format instructions in the prompt. The exact parsing approach (JSON output from the LLM, structured text with markers, or a combination) is an implementation detail of the generation layer, but the output must conform to the `GeneratedAnswer` schema.

**Invariance:** If parsing fails, the generation layer returns a generation failure, not a partially parsed answer presented as success.

### 9.4 Evidence quality classification

The LLM is instructed to classify evidence quality into one of: `sufficient`, `partial`, `insufficient`, `conflicting`. The generation layer maps the LLM's classification to the `EvidenceQuality` enumeration.

**Conservative heuristics (per PRFAQ §3 risk mitigation and story 4.2 decision):** When the boundary between categories is unclear, err toward `"insufficient"` rather than overclaiming. This instruction is encoded in the evidence quality instructions section of the prompt.

### 9.5 Abstention

The generation layer produces `is_abstention=True` when:
- The LLM classifies evidence as `"insufficient"`, OR
- Parsing determines that the answer does not make a definitive claim about the question (i.e., it says the documents don't contain enough information).

An abstention answer:
- States that the documents do not contain enough information to answer definitively.
- Includes the closest relevant passage if one exists (from the retrieved context).
- Does not fabricate a definitive answer.
- Uses the `"insufficient"` evidence quality category and the corresponding language.

### 9.6 Partial evidence

When the LLM classifies evidence as `"partial"`:
- The answer distinguishes the part supported by the documents from the part that is inference.
- Supported claims are cited to specific passages.
- Inferences are clearly labeled as inferences, not as statements from the documents.
- Uses the `"partial"` evidence quality category and language.

### 9.7 Conflicting evidence

When the LLM classifies evidence as `"conflicting"`:
- The answer names the conflicting sources and what each says.
- Does not silently choose one side.
- Uses the `"conflicting"` evidence quality category and language.

### 9.8 Citation generation

The generation layer produces `Citation` objects for claims that are supported by retrieved passages. Each `Citation` carries:
- `passage_id` — from the `RetrievedPassage` that supports the claim.
- `document_name` — from the `RetrievedPassage`.
- `location` — from the `RetrievedPassage`.
- `excerpt` — populated later by the citation resolver from the store (not generated by the LLM).

**Invariance:** The generation layer does not produce citations for claims not supported by retrieved passages. It does not produce citations with `passage_id`s that are not in the retrieved context.

---

## 10. Evaluation Architecture

### 10.1 Dataset format

The evaluation dataset lives in `evaluation/dataset/`. It is a living artifact, maintained by the product owner with Hermes assistance.

**Documents:** Representative documents in MVP formats, stored in `evaluation/dataset/documents/`. Each document is tagged with metadata about its characteristics.

**Questions:** Stored in `evaluation/dataset/questions.json` (or questions.jsonl for larger datasets). Format:

```json
{
  "questions": [
    {
      "id": "q001",
      "question_text": "...",
      "question_type": "single_source",
      "gold_answer": "...",
      "gold_passage_ids": ["doc_id:0", "doc_id:3"],
      "metadata": {}
    },
    {
      "id": "q004",
      "question_text": "...",
      "question_type": "insufficient",
      "gold_answer": null,
      "gold_passage_ids": [],
      "closest_relevant_passage_id": "doc_id:2",
      "metadata": {}
    },
    {
      "id": "q005",
      "question_text": "...",
      "question_type": "conflicting",
      "gold_answer": "The documents disagree...",
      "gold_passage_ids": ["doc_id_a:1", "doc_id_b:0"],
      "conflicting_sources": [
        {
          "source_label": "Document A",
          "claim": "X",
          "passage_id": "doc_id_a:1"
        },
        {
          "source_label": "Document B",
          "claim": "Y",
          "passage_id": "doc_id_b:0"
        }
      ],
      "metadata": {}
    }
  ]
}
```

### 10.2 Gold answer representation

- For `single_source` and `multi_passage` questions: `gold_answer` is the expected answer text. `gold_passage_ids` lists the passage IDs that support it.
- For `partial` questions: `gold_answer` describes what is supported and what is not. `gold_passage_ids` lists the passages that support the supported part.
- For `insufficient` questions: `gold_answer` is `null`. `gold_passage_ids` is empty. `closest_relevant_passage_id` identifies the closest passage, if any.
- For `conflicting` questions: `gold_answer` describes the conflict. `gold_passage_ids` lists passages from both sides. `conflicting_sources` lists the conflicting claims.

### 10.3 Source / passage references

Gold passage references use the stable passage ID format (`doc_id:passage_index`). This allows the evaluation runner to resolve them against the store and verify citation correctness.

### 10.4 Evaluation runner

**Location:** `evaluation/runner.py`.  
**Interface:**

```python
class EvaluationRunner:
    def run_question(self, question: EvaluationQuestion, system: EvaluationSystem) -> EvaluationResult:
        ...

    def run_all(self, questions: list[EvaluationQuestion], system: EvaluationSystem) -> list[EvaluationResult]:
        ...

    def summarize(self, results: list[EvaluationResult]) -> dict:
        ...
```

**`EvaluationSystem` abstraction:** Represents the system under test. It can be:
- An API client (end-to-end evaluation), or
- A component-level interface that calls retrieval and generation directly (component evaluation).

This abstraction allows the runner to evaluate components before the full API is available.

### 10.5 Metrics

The runner computes:

- **Retrieval precision:** fraction of retrieved passages that are in `gold_passage_ids` (for questions where gold passages are defined). For insufficient-evidence questions, precision is not applicable.
- **Retrieval recall:** fraction of `gold_passage_ids` that are in the retrieved passages.
- **Citation correctness:** fraction of citations in the system's answer whose `passage_id` is in `gold_passage_ids` and whose `excerpt` matches the store's passage text.
- **Answer faithfulness:** binary — does the system's answer text contain claims not supported by the retrieved passages? (Determined by comparing answer claims to retrieved passages; the exact method is defined in the methodology document.)
- **Abstention correctness:** for insufficient-evidence questions, was the system's answer an abstention? For other questions, was the system's answer not an abstention?
- **Conflict handling:** for conflicting-evidence questions, did the system's answer name both conflicting sources?
- **Latency:** `total_latency_seconds` from the system's answer.

### 10.6 Result format

Each `EvaluationResult` (Section 3.9) is written to `evaluation/dataset/results/<run_id>.json` (or a timestamped file). The `summarize` method produces an aggregate summary: per-metric averages, per-question-type breakdown, and a list of failures for inspection.

### 10.7 Component-level evaluation

Before the full API is available, the runner can evaluate:
- **Retrieval only:** submit a question, run retrieval, compare retrieved passage IDs to gold passage IDs. Computes precision and recall.
- **Generation only:** construct a `RetrievedContext` with known passages, run generation, compare the answer to the gold answer. Computes faithfulness.

This allows retrieval and generation to be evaluated as soon as they are implemented, without waiting for the full API.

### 10.8 End-to-end evaluation

Once the API is available, the runner evaluates the full pipeline: upload documents, submit queries via the API, collect answers, compare. This evaluates the full system including ingestion, storage, retrieval, generation, and citation resolution.

---

## 11. Testing Strategy

### 11.1 Unit tests

- **config:** validate that settings load correctly, defaults are as specified, invalid configurations are rejected.
- **chunking:** given a `RawDocument` and a `ChunkingConfig`, verify that the produced passages have correct IDs, text, offsets (offset invariance: `source_text[start:end] == passage.text`), locations, and cover the source text without gaps. Test with different configs (different sizes, overlaps, boundary preferences).
- **store:** verify document and passage CRUD, cascade delete, passage lookup by ID, document lookup by ID. Test with SQLite directly.
- **retrieval:** verify semantic retrieval returns relevant passages for vocabulary-mismatch queries; verify BM25 returns passages for exact-term queries; verify hybrid combines both and applies scope filtering and deduplication; verify candidate set size limit.
- **generation:** verify that given a `RetrievedContext` with known passages, the generated answer cites the correct passages, uses the correct evidence quality category, and abstains when appropriate. Test with constructed contexts for each evidence quality category.
- **conversation:** verify session creation, turn addition, history retrieval with pruning, session deletion cascades to turns.

### 11.2 Integration tests

- **ingestion → store:** upload a document of each format, verify the document is stored, passages are stored with correct IDs and offsets, the document is listable.
- **retrieval → generation:** given a document in the store, run a query, verify the answer cites passages from the store and that citations are resolvable.
- **API end-to-end:** upload a document, query it, verify the response shape and that citations resolve.

### 11.3 Retrieval tests

Specific tests for retrieval quality:
- Vocabulary mismatch: query uses different words than the relevant passage; relevant passage is in the top N.
- Exact term: query contains a precise term that appears in a passage; that passage is returned by BM25.
- Hybrid combination: a query where semantic-only misses and keyword-only misses, but hybrid returns the relevant passage.
- Scope filtering: querying with a specific document scope returns passages only from that document.
- Candidate set size: querying returns no more than `max_candidates` results.

### 11.4 Citation tests

- Citation resolution: for a citation in an answer, verify that `passage_id` exists in the store, `document_name` matches, `location` matches, and `excerpt` is the exact passage text.
- Citation correctness (evaluation): for an evaluation question with gold passage IDs, verify that the system's citations point to gold passages.

### 11.5 Generation tests

- Groundedness: given a `RetrievedContext` with passages that do not contain the answer, verify the system abstains or says evidence is insufficient.
- Faithfulness: given a `RetrievedContext` with passages that support a specific answer, verify the generated answer does not introduce unsupported claims.
- Evidence quality: test each category (sufficient, partial, insufficient, conflicting) with constructed contexts.

### 11.6 Abstention tests

- Insufficient evidence: query a document that does not contain the answer; verify the system abstains and does not fabricate.
- Sufficient evidence: query a document that does contain the answer; verify the system does not abstain.

### 11.7 Conflict tests

- Conflicting sources: query with two documents that disagree; verify the system's answer names both sources and does not silently choose one.

### 11.8 API tests

- Test each endpoint (upload, list, remove, query, citation lookup, follow-up) for correct request/response shapes, error handling, and status codes.
- Test that the API does not expose numerical confidence scores.
- Test that the API does not require authentication.

### 11.9 Evaluation tests

- Test the evaluation runner with a small set of questions and a mock system, verifying that it computes metrics correctly.
- Test the runner's ability to run against components (retrieval only, generation only) and against the API.

### 11.10 Test data

Test documents and questions are separate from the evaluation dataset. They are smaller, simpler, and designed to exercise specific behaviors. They live in `tests/fixtures/` or are generated inline in tests.

---

## 12. Implementation Order at Story Level

### 12.1 Ordering principles

1. **Epic 0 (evaluation dataset) is parallel and starts immediately.** Story 0.1 (methodology) must be delivered before epics 2–4 proceed significantly. Story 0.4 (runner) must be available before end-to-end testing of epic 7. Stories 0.2 and 0.3 (documents and questions) are worked in parallel with implementation.
2. **Epic 1 (ingestion) → Epic 2 (content store) → Epic 3 (retrieval) → Epic 4 (generation) is the main sequential chain.** Each depends on the previous.
3. **Within epics, stories are ordered by dependency.** Where stories are independent, they can be parallelized.
4. **Epic 5, Epic 6, and Epic 7 (stories 7.1–7.5) can proceed in parallel once their dependencies are met.**
5. **Story 7.6 (application wiring) is last**, as it requires all other epics to be functional.

### 12.2 Parallelization opportunities

- **Epic 0 stories 0.1, 0.2, 0.3, 0.4** can all proceed in parallel with each other and with epics 1–7 (subject to the pacing constraints).
- **Epic 1 stories 1.1, 1.2, 1.3** (format handlers) can proceed in parallel with each other, since they are independent format-specific implementations. Story 1.4 (chunker) depends on the format handlers only in the sense that it operates on their output — it can be developed in parallel as long as the `RawDocument` structure is agreed. Story 1.5 (orchestration) depends on 1.1–1.4.
- **Epic 2 stories 2.1 and 2.2** — story 2.2 (delete) depends on story 2.1 (persist), but both are small. Can be done together.
- **Epic 3 stories 3.1 and 3.2** (semantic and keyword retrievers) can proceed in parallel, since they are independent implementations of the two retrieval paths. Story 3.3 (hybrid) depends on both. Story 3.4 (scope) and 3.5 (ranking/size) are independent of each other and of 3.1/3.2 in terms of the interface they add; they depend on the retrievers being available for full testing but can be developed in parallel with them.
- **Epic 4 stories 4.1, 4.2, 4.3** (interface, prompt, schema) can proceed in parallel — they are the definition of the generation contract and don't require a working retrieval implementation to develop (they can be tested with constructed `RetrievedContext` objects). Stories 4.4, 4.5, 4.6 depend on 4.1–4.3 and on the evaluation dataset for testing.
- **Epic 5 story 5.1** (session storage) can proceed in parallel with epics 3 and 4, since it's independent of retrieval quality. Story 5.2 depends on 5.1 and on epics 3 and 4.
- **Epic 6 stories 6.1 and 6.2** — story 6.2 depends on story 6.1. Both depend on the store (for passage lookup) and generation output (for citation structure), but not on retrieval quality. Can be developed in parallel with epics 3 and 4 once the store and generation output schema are available.
- **Epic 7 story 7.1** (API contract) can be done early, as contract-first. Stories 7.2–7.5 depend on their respective epics. Story 7.6 depends on all of the above.

### 12.3 Implementation sequence (story level)

**Phase 0 — Parallel (start immediately):**
- 0.1 Define evaluation methodology
- 0.2 Begin curating documents
- 0.3 Begin curating questions
- 0.4 Establish evaluation runner scaffolding

**Phase 1 — Core pipeline (sequential chain with internal parallelism):**

*Wave 1 (parallel):*
- 1.1 PDF ingestion
- 1.2 Markdown ingestion
- 1.3 TXT ingestion
- 2.1 Document and passage persistence (+ 2.2 Passage deletion, small, can go with 2.1)

*Wave 2 (depends on Wave 1):*
- 1.4 Chunking strategy (operates on RawDocument; can be developed alongside Wave 1 if RawDocument structure is agreed)
- 1.5 Ingestion orchestration (depends on 1.1–1.4 and 2.1)

*Wave 3 (depends on Wave 2 + store populated):*
- 3.1 Semantic retrieval
- 3.2 Keyword/BM25 retrieval
- 3.4 Document scope filtering (can develop alongside 3.1/3.2; full testing requires retrievers)

*Wave 4 (depends on Wave 3):*
- 3.3 Hybrid retrieval (depends on 3.1, 3.2, 3.4)
- 3.5 Retrieval ranking and candidate set size (depends on 3.3)

*Wave 5 (depends on Wave 3–4 for full testing, but interface/prompt/schema can be developed earlier):*
- 4.1 Model-agnostic LLM interface
- 4.2 Prompt construction
- 4.3 Structured output schema

*Wave 6 (depends on Wave 5 + evaluation dataset):*
- 4.4 Abstention logic
- 4.5 Partial and conflicting evidence handling
- 4.6 Citation generation

**Phase 2 — User-facing capabilities (parallel once dependencies met):**

*Wave 7 (parallel, depends on store + generation schema):*
- 5.1 Conversation context storage and retrieval
- 6.1 Citation rendering

*Wave 8 (depends on Wave 7):*
- 5.2 Follow-up query with conversation context (depends on 5.1, 3, 4)
- 6.2 Citation UI for multi-document answers (depends on 6.1)

**Phase 3 — Integration:**

*Wave 9 (contract-first, can be early):*
- 7.1 API contract definition

*Wave 10 (depends on epics 1–4):*
- 7.2 Document upload endpoint
- 7.3 Query endpoint
- 7.4 Citation lookup endpoint
- 7.5 Document list and removal endpoints

*Wave 11 (last):*
- 7.6 Application wiring and startup

---

## 13. Story-by-Story Implementation Map

For each story: files/modules, dependencies, interfaces consumed, interfaces produced, tests required, acceptance criteria mapping.

### Epic 0

**Story 0.1 — Define evaluation methodology and metric definitions**
- Files: `evaluation/methodology.md` (new), `evaluation/schema.py` (supports metric definitions)
- Dependencies: none
- Consumes: PRFAQ §8, Architecture §9
- Produces: methodology document
- Tests: review/validate the methodology document against the PRFAQ metrics; no code tests
- AC mapping: all 7 ACs in the story map to sections of the methodology document

**Story 0.2 — Begin curating representative documents**
- Files: `evaluation/dataset/documents/` (new documents), `evaluation/dataset/metadata.json` (document tags)
- Dependencies: none (documents are standalone files)
- Consumes: PRFAQ §8
- Produces: initial document corpus with metadata
- Tests: manual review that each document matches its metadata tags; no code tests required
- AC mapping: 6 ACs map to document selection and tagging

**Story 0.3 — Begin curating representative questions**
- Files: `evaluation/dataset/questions.json` (new), `evaluation/schema.py` (EvaluationQuestion model)
- Dependencies: story 0.2 (documents to write questions against)
- Consumes: PRFAQ §8, Architecture §9
- Produces: initial question set with gold answers and source locations
- Tests: manual review that gold answers and passage IDs are accurate against the documents
- AC mapping: 6 ACs map to question curation

**Story 0.4 — Establish evaluation run capability**
- Files: `evaluation/runner.py` (new), `evaluation/schema.py` (EvaluationResult model)
- Dependencies: story 0.3 (questions), `app.store` interface for passage resolution (for citation correctness checks), `app.generation` interface or API for answer collection
- Consumes: PRFAQ §8, Architecture §9
- Produces: evaluation runner that can run against components and (later) the API
- Tests: `tests/evaluation/test_runner.py` — test runner with a mock system and a small set of questions; verify metrics are computed correctly
- AC mapping: 6 ACs map to runner capabilities

### Epic 1

**Story 1.1 — PDF text extraction**
- Files: `app/ingestion/pdf.py` (new)
- Dependencies: none (pure extraction function)
- Consumes: Architecture §4, §11
- Produces: `extract_text_from_pdf` function, `RawDocument` with pages
- Tests: `tests/unit/test_ingestion.py` or `tests/integration/test_ingestion.py` — test with a representative text-based PDF; verify page boundaries preserved, offsets derivable; test with a non-PDF file (expect error); test with a corrupt PDF (expect clear error)
- AC mapping: 6 ACs

**Story 1.2 — Markdown parsing**
- Files: `app/ingestion/markdown.py` (new)
- Dependencies: none
- Consumes: Architecture §4, §11
- Produces: `parse_markdown` function, `RawDocument` with blocks
- Tests: test with representative Markdown documents; verify heading structure captured, section locators derivable, blocks identifiable; test edge cases (nested lists, code blocks, no heading structure)
- AC mapping: 6 ACs

**Story 1.3 — Plain text ingestion**
- Files: `app/ingestion/text.py` (new)
- Dependencies: none
- Consumes: Architecture §4, §11
- Produces: `parse_plain_text` function, `RawDocument` with paragraphs
- Tests: test with representative text files; verify paragraph splitting, offsets derivable; test edge cases (no paragraph boundaries, single paragraph)
- AC mapping: 5 ACs (the story has 5 ACs)

**Story 1.4 — Chunking strategy**
- Files: `app/chunking/strategy.py` (new), `app/chunking/__init__.py` (new)
- Dependencies: `RawDocument` structure (agreed across 1.1–1.3); `app.store` interface not required for development (chunker produces Passages; store consumes them)
- Consumes: Architecture §4, §13; Epics & Stories story 1.4 decision (250–600 char target, 80–150 overlap, semantic boundaries)
- Produces: `ChunkingConfig` dataclass, `chunk_passages(raw_doc, config) → list[Passage]` function
- Tests: `tests/unit/test_chunking.py` — test offset invariance, coverage without gaps, overlap, boundary preference, configurable parameters; test with different configs
- AC mapping: 6 ACs + the decision paragraph

**Story 1.5 — Ingestion orchestration**
- Files: `app/ingestion/orchestrator.py` (new), modifies `app/store/repository.py` (uses store for persistence)
- Dependencies: stories 1.1, 1.2, 1.3, 1.4, and story 2.1 (store)
- Consumes: Architecture §4
- Produces: `ingest_document` function, integration between format handlers, chunker, and store
- Tests: `tests/integration/test_ingestion.py` — end-to-end: upload a document of each format, verify document and passages in store, offsets correct, listable
- AC mapping: 6 ACs

### Epic 2

**Story 2.1 — Document and passage persistence**
- Files: `app/store/schema.py` (new), `app/store/repository.py` (new)
- Dependencies: none (defines the store; other epics consume it)
- Consumes: Architecture §2, §11; Epics & Stories story 2.1 decision (SQLite)
- Produces: SQLite schema, repository functions (create_document, create_passage, get_document, list_passages_for_document, get_passage, delete_document)
- Tests: `tests/unit/test_store.py` — test all repository functions, cascade delete, lookup by ID, offset storage
- AC mapping: 6 ACs + decision paragraph

**Story 2.2 — Passage deletion and document removal**
- Files: `app/store/repository.py` (modify — add delete function), `app/api/documents.py` (uses delete)
- Dependencies: story 2.1
- Consumes: Architecture §4
- Produces: `delete_document` repository function (uses CASCADE), API removal endpoint (story 7.5)
- Tests: `tests/unit/test_store.py` — test cascade delete; `tests/integration/test_api.py` — test removal endpoint
- AC mapping: 4 ACs

### Epic 3

**Story 3.1 — Semantic retrieval**
- Files: `app/retrieval/interface.py` (new — defines SemanticRetriever), `app/retrieval/semantic.py` (new)
- Dependencies: `app.store` interface for passage data; embedding model library (sentence-transformers, all-MiniLM-L6-v2)
- Consumes: Architecture §5, §10, §11; Epics & Stories story 3.1 decision (all-MiniLM-L6-v2)
- Produces: `SemanticRetriever` implementation, `embed`, `index_passages`, `search` functions
- Tests: `tests/unit/test_retrieval.py` — test embedding, indexing, search with vocabulary-mismatch queries; test that search returns passage IDs that exist in the store
- AC mapping: 6 ACs + decision paragraph

**Story 3.2 — Keyword/BM25 retrieval**
- Files: `app/retrieval/interface.py` (modify — defines KeywordRetriever), `app/retrieval/keyword.py` (new)
- Dependencies: `app.store` interface for passage text; rank_bm25 library
- Consumes: Architecture §5; Epics & Stories story 3.2 decision (rank_bm25)
- Produces: `KeywordRetriever` implementation, BM25 indexing and search
- Tests: `tests/unit/test_retrieval.py` — test BM25 indexing, search with exact-term queries; test that search returns passage IDs that exist in the store
- AC mapping: 5 ACs + decision paragraph

**Story 3.3 — Hybrid retrieval**
- Files: `app/retrieval/hybrid.py` (new), `app/retrieval/interface.py` (modify — defines HybridRetriever)
- Dependencies: stories 3.1, 3.2, 3.4
- Consumes: Architecture §5; Epics & Stories story 3.3 decision (0.5/0.5, normalized, configurable)
- Produces: `HybridRetriever` implementation, combination logic with configurable weighting
- Tests: `tests/unit/test_retrieval.py` — test hybrid combination, normalization, weighting, deduplication, scope filtering integration; test with queries where semantic-only misses and keyword-only misses but hybrid returns the relevant passage
- AC mapping: 5 ACs + decision paragraph

**Story 3.4 — Document scope filtering**
- Files: `app/retrieval/hybrid.py` (modify — add scope filtering), `app/retrieval/interface.py` (modify — scope parameter)
- Dependencies: stories 3.1, 3.2 (scope filtering is applied to their results)
- Consumes: Architecture §5
- Produces: scope filtering in the hybrid search interface
- Tests: `tests/unit/test_retrieval.py` — test that scoping to a specific document returns only passages from that document; test that "all" returns passages from all documents
- AC mapping: 5 ACs

**Story 3.5 — Retrieval result ranking and candidate set size**
- Files: `app/retrieval/hybrid.py` (modify — add max_candidates), `app/retrieval/interface.py` (modify)
- Dependencies: story 3.3
- Consumes: Architecture §5; Epics & Stories story 3.5 decision (20 passages, configurable)
- Produces: max_candidates parameter in hybrid search, ranking stability
- Tests: `tests/unit/test_retrieval.py` — test that results are limited to max_candidates, ranking is deterministic, no off-by-one in truncation
- AC mapping: 6 ACs + decision paragraph

### Epic 4

**Story 4.1 — Model-agnostic LLM interface**
- Files: `app/generation/interface.py` (new), `app/generation/__init__.py` (new)
- Dependencies: Hermes agent access (external, through Hermes); model-agnostic by design
- Consumes: Architecture §6, §10
- Produces: `LLMInterface` protocol, initial implementation using Hermes free model, `LLMResponse` model
- Tests: `tests/unit/test_generation.py` — test interface with a sample prompt; verify structured response fields; test error handling (malformed response → parse_error set)
- AC mapping: 5 ACs

**Story 4.2 — Prompt construction**
- Files: `app/generation/prompt.py` (new)
- Dependencies: story 4.1 (LLM interface not required to build the prompt, only to test it); `RetrievedContext` model (from retrieval/generation boundary); `GenerationConfig`
- Consumes: Architecture §6; Epics & Stories story 4.2 decision (8-section structure)
- Produces: `build_prompt(context, config) → str` function
- Tests: `tests/unit/test_generation.py` — test that the prompt contains all 8 sections with correct content; test that prompt does not instruct the model to introduce outside knowledge; test with and without conversation context
- AC mapping: 6 ACs + decision paragraph

**Story 4.3 — Structured output schema**
- Files: `app/generation/schema.py` (new), modifies `app/generation/interface.py` (LLMResponse.structured)
- Dependencies: story 4.1, story 4.2
- Consumes: Architecture §6
- Produces: `GeneratedAnswer`, `Citation`, `EvidenceQuality` models; parsing function from LLM raw output to `GeneratedAnswer`
- Tests: `tests/unit/test_generation.py` — test parsing of sample LLM outputs into `GeneratedAnswer`; test that parsing failure sets parse_error; test that citation excerpt is not populated here (left for resolver)
- AC mapping: 5 ACs

**Story 4.4 — Abstention logic**
- Files: `app/generation/schema.py` (modify — abstention logic), `app/generation/prompt.py` (evidence quality instructions already cover this)
- Dependencies: story 4.3
- Consumes: Architecture §6; PRFAQ §2
- Produces: abstention detection in the generation layer; abstention response generation
- Tests: `tests/unit/test_generation.py` — test with a `RetrievedContext` that does not contain the answer; verify is_abstention=True, evidence_quality=INSUFFICIENT, answer does not fabricate; test with a context that does contain the answer; verify is_abstention=False
- AC mapping: 6 ACs

**Story 4.5 — Partial and conflicting evidence handling**
- Files: `app/generation/schema.py` (modify), `app/generation/prompt.py` (evidence quality instructions)
- Dependencies: story 4.3, evaluation dataset (stories 0.2, 0.3) for test cases
- Consumes: Architecture §6; PRFAQ §2
- Produces: partial evidence and conflicting evidence behavior in the generation layer
- Tests: `tests/unit/test_generation.py` — test partial evidence with a context that supports part of an answer; verify answer distinguishes supported from inference, uses PARTIAL category; test conflicting evidence with two sources that disagree; verify answer names both, uses CONFLICTING category
- AC mapping: 5 ACs (the story has 5 ACs covering both partial and conflicting)

**Story 4.6 — Citation generation**
- Files: `app/generation/schema.py` (modify — citation generation), `app/generation/prompt.py` (citation instructions)
- Dependencies: story 4.3; `app.store` interface (for citation resolver, story 4.6 does not call the store directly but produces citations that the resolver will look up)
- Consumes: Architecture §6
- Produces: citation generation in the generation layer; citations carry passage_id, document_name, location (excerpt left for resolver)
- Tests: `tests/unit/test_generation.py` — test that citations produced by generation have correct passage_id, document_name, location from the RetrievedContext; test that no citation is produced for an unsupported claim; test that citation's excerpt is empty/None at this stage (resolver populates it)
- AC mapping: 6 ACs

### Epic 5

**Story 5.1 — Conversation context storage**
- Files: `app/conversation/session.py` (new), modifies `app/store/schema.py` and `app/store/repository.py` (conversation tables)
- Dependencies: story 2.1 (store)
- Consumes: Architecture §7
- Produces: `create_session`, `add_turn`, `get_history`, `prune_history` functions; Session and ConversationTurn models
- Tests: `tests/unit/test_conversation.py` — test session creation, turn addition, history retrieval with 5-turn limit and pruning, session deletion cascades to turns
- AC mapping: 6 ACs + decision paragraph (5-turn default)

**Story 5.2 — Follow-up query**
- Files: `app/api/query.py` or `app/api/conversation.py` (follow-up endpoint), `app/conversation/session.py` (uses history)
- Dependencies: stories 5.1, 3 (retrieval), 4 (generation)
- Consumes: Architecture §7
- Produces: follow-up endpoint that records a turn and returns an answer
- Tests: `tests/integration/test_api.py` — test follow-up with a multi-turn conversation; verify answer is grounded and consistent with initial answer; test scope change in follow-up
- AC mapping: 5 ACs

### Epic 6

**Story 6.1 — Citation rendering**
- Files: `app/citation/resolver.py` (new), `app/api/citations.py` (citation lookup endpoint, story 7.4), UI layer (not specified in detail — the clickable inline-expandable reference is the MVP mechanism)
- Dependencies: story 2.1 (store for passage lookup), story 4.3/4.6 (citation structure)
- Consumes: Architecture §8; Epics & Stories story 6.1 decision (clickable inline-expandable reference)
- Produces: `resolve_citation(passage_id) → Citation | Error` function; the UI mechanism for displaying citations (clickable, expands inline to show passage excerpt)
- Tests: `tests/unit/test_citation.py` (if a separate module) or `tests/integration/test_api.py` — test that citation lookup returns correct passage text and metadata; test that the UI mechanism displays the actual passage text for a sample of citations
- AC mapping: 5 ACs + decision paragraph

**Story 6.2 — Citation UI for multi-document answers**
- Files: UI layer (same as 6.1), `app/api/query.py` (answer response already includes document_name per citation)
- Dependencies: story 6.1
- Consumes: Architecture §8
- Produces: citation UI that correctly displays document name per citation in multi-document answers
- Tests: `tests/integration/test_api.py` — test with a multi-document answer; verify each citation displays the correct document name
- AC mapping: 4 ACs

### Epic 7

**Story 7.1 — API contract definition**
- Files: `app/api/documents.py`, `app/api/query.py`, `app/api/citations.py` (define endpoint signatures, request/response models)
- Dependencies: none (contract-first; implementation follows)
- Consumes: PRFAQ §2; Architecture §5
- Produces: API specification (this implementation plan, Section 5, is the specification); request/response Pydantic models
- Tests: no code tests (contract definition); validate the contract against the PRFAQ requirements
- AC mapping: 7 ACs

**Story 7.2 — Document upload endpoint**
- Files: `app/api/documents.py` (modify — implement upload), `app/ingestion/orchestrator.py` (consumed)
- Dependencies: story 1.5 (orchestration), story 2.1 (store)
- Consumes: Architecture §4
- Produces: POST /api/v1/documents/upload endpoint
- Tests: `tests/integration/test_api.py` — test upload of each format, verify 200/202 response, document in store, error handling for invalid format
- AC mapping: 6 ACs

**Story 7.3 — Query endpoint**
- Files: `app/api/query.py` (modify — implement query), consumes retrieval and generation
- Dependencies: stories 3 (retrieval), 4 (generation), 2.1 (store)
- Consumes: Architecture §5
- Produces: POST /api/v1/query endpoint
- Tests: `tests/integration/test_api.py` — end-to-end: upload a document, query it, verify answer shape, citations resolvable, evidence quality present, latency recorded; test with insufficient-evidence question; test with scoped query
- AC mapping: 6 ACs

**Story 7.4 — Citation lookup endpoint**
- Files: `app/api/citations.py` (modify — implement lookup), `app/citation/resolver.py` (consumed)
- Dependencies: story 2.1 (store), story 6.1 (resolver)
- Consumes: Architecture §5
- Produces: GET /api/v1/citations/{passage_id} endpoint
- Tests: `tests/integration/test_api.py` — test lookup of existing passage, verify response shape and exact text match; test lookup of non-existent passage (404)
- AC mapping: 4 ACs

**Story 7.5 — Document list and removal endpoints**
- Files: `app/api/documents.py` (modify — implement list and remove)
- Dependencies: story 2.1, story 2.2
- Consumes: Architecture §5
- Produces: GET /api/v1/documents, DELETE /api/v1/documents/{id} endpoints
- Tests: `tests/integration/test_api.py` — test list returns uploaded documents; test removal removes document and passages; test removal of non-existent document (404); test that removal does not affect other documents
- AC mapping: ACs covered by story 2.2 tests + API-specific tests

**Story 7.6 — Application wiring and startup**
- Files: `app/main.py` (new), modifies all API modules to register routers
- Dependencies: all of epic 7 (7.1–7.5), and epics 1–4 for underlying capabilities
- Consumes: Architecture §5
- Produces: FastAPI application that starts, serves all endpoints, shuts down cleanly
- Tests: `tests/integration/test_api.py` — full end-to-end: start app, upload a document, query it, verify answer and citations, look up a citation, verify passage text; test application shutdown
- AC mapping: 4 ACs

---

## 14. Remaining Technical Decisions

The following technical decisions are implementation-level choices that the approved PRFAQ, Architecture v1, and Epics & Stories deliberately leave open or give only directional guidance on. None of them is an architectural requirement. They are recorded here as explicit implementation decisions with recommended MVP defaults, consistent with the approved artifacts and the user's specified defaults. They should be confirmed when the relevant story is delivered.

**Classification key used throughout this section:**
- **Architectural requirement:** mandated by Architecture v1 or the PRFAQ; the implementation plan must satisfy it.
- **Approved MVP decision:** resolved through the Epics & Stories decision-resolution pass (items 1–10); recorded in the relevant story.
- **Implementation recommendation:** an engineering choice that the approved artifacts do not mandate; recorded here for clarity and to avoid re-litigating it during implementation.

### 14.1 PDF extraction library choice (story 1.1) — IMPLEMENTATION RECOMMENDATION

- **Decision:** pypdf.
- **Why unresolved:** Architecture §11 names "pypdf, pdfplumber" as examples without choosing. Not resolved by the Epics & Stories decision pass.
- **Impact:** Story 1.1 delivery only. The RawDocument contract is the same regardless.
- **Constraint check:** Architecture §12 explicitly excludes OCR-dependent libraries for MVP. pypdf is a text-based PDF library; it does not perform OCR. Consistent with Architecture §12.
- **Action:** Record "pypdf" as the choice when delivering story 1.1.

- **Decision:** pypdf.

### 14.2 Markdown parser choice (story 1.2)

- **Decision:** mistune v3 (Python Markdown parser with a proper block-level AST: each block is a typed object with a type — heading, paragraph, code, list — and headings expose their level directly). The parser is used to produce the structured `Block` list (`level`, `heading`, `block_type`, `text`) required by the `RawDocument` contract for Markdown.

- **Why unresolved:** Architecture §11 names "a Python Markdown parser that exposes block structure" without choosing a specific library. The Epics & Stories decision pass did not name a specific Markdown library.

- **Constraint check:** Story 1.2's AC requires heading levels (H1, H2, etc.) to be identifiable and section locators to be constructible from the heading hierarchy. mistune v3's block AST exposes heading levels natively, and its block types (heading, paragraph, code, list) map directly onto the `Block` structure needed by `RawDocument`. Consistent with Architecture §12 (no DOCX, no OCR — both irrelevant to a Markdown parser).

- **Impact:** Story 1.2 delivery. The `Block` extraction logic (transforming mistune's AST into the `RawDocument.blocks` structure) is the main implementation work.

- **Action:** Record "mistune v3" as the Markdown parser when delivering story 1.2.

### 14.3 Embedding index structure (story 3.1) — IMPLEMENTATION RECOMMENDATION

- **Decision:** Brute-force cosine similarity over all indexed passages. No separate vector database for MVP.
- **Why unresolved:** Architecture §11 says "a local vector store appropriate for a single-user MVP" without specifying the indexing approach. Architecture §13 item 6 leaves the embedding model and vector store choice open. The Epics & Stories decision pass named all-MiniLM-L6-v2 as the embedding model but did not specify the index structure.
- **Constraint check:** The architecture's "local vector store" requirement is satisfied by storing embeddings as JSON arrays in SQLite (Section 6 of the implementation plan) and performing similarity search in-process. This is a local store; no external service is required.
- **Impact:** Story 3.1 delivery. Appropriate for MVP scale (tens to low hundreds of passages). Not a long-term solution for large corpora, but the architecture explicitly defers the vector store choice and the evaluation dataset will validate whether this is sufficient.
- **Action:** Record "brute-force cosine similarity, no vector database" when delivering story 3.1.

### 14.4 BM25 index structure (story 3.2) — IMPLEMENTATION RECOMMENDATION

- **Decision:** Rebuild the BM25 index from the current passage corpus whenever passages are added or removed.
- **Why unresolved:** Architecture does not specify BM25 index maintenance.
- **Constraint check:** Architecture specifies BM25-style retrieval and hybrid retrieval. It does not mandate incremental index updates.
- **Impact:** Story 3.2 delivery. For MVP scale, full reindex on change is acceptable.
- **Action:** Record "full reindex on passage add/remove" when delivering story 3.2.

### 14.5 Synchronous vs. asynchronous ingestion (stories 1.5 / 7.2) — IMPLEMENTATION RECOMMENDATION

- **Decision:** Synchronous ingestion. Upload returns 200 with status "available" (or "failed").
- **Why unresolved:** The API contract (implementation plan Section 5.1) presents both 200 and 202 as options. The Epics & Stories story 1.5 AC includes a conditional ("if the content store is not yet available, this story produces passages in consumable form") that is compatible with either approach.
- **Constraint check:** Neither the architecture nor the Epics & Stories mandates asynchronous ingestion. The architecture's "local-first, single-user" posture and the absence of a background-task requirement in the architecture favor synchronous for MVP simplicity.
- **Impact:** Stories 1.5 and 7.2 delivery. Simpler for MVP; no background task system needed.
- **Action:** Record "synchronous ingestion, 200 response" when delivering stories 1.5 and 7.2.

### 14.6 LLM structured output parsing (story 4.3) — IMPLEMENTATION RECOMMENDATION

- **Decision:** Request JSON output from the LLM via the prompt's output format instructions; parse the JSON into the `GeneratedAnswer` model. Fallback: if the JSON is invalid, attempt to extract structured fields from the raw text; if that fails, return a generation error (not a fabricated answer).
- **Why unresolved:** Architecture §6 says the output is "structured (not free text), tied to passage IDs" and that the generation layer "produces citations that point to specific passages." It says the generation layer should produce structured output but does not specify the parsing mechanism. Architecture §10 says the model is pluggable and the output schema is the contract.
- **Constraint check:** Consistent with Architecture §6 (structured output, citation generation) and Architecture §10 (model-agnostic; output schema is the contract). The fallback does not violate the abstention requirement — a parsing failure returns an error, not a fabricated answer.
- **Impact:** Story 4.3 delivery and generation reliability.
- **Action:** Record "JSON output with fallback parsing" when delivering story 4.3.

### 14.7 Conversation history storage (story 5.1) — IMPLEMENTATION RECOMMENDATION

- **Decision:** Store conversation turns in the SQLite `conversation_turns` table (Section 6.4 of the implementation plan). Citations within each turn are stored as a JSON array. This is sufficient for the MVP.
- **Why unresolved:** The architecture does not specify the conversation storage format; it only requires that conversation history be available as context for generation.
- **Constraint check:** Consistent with Architecture §7 (conversation history is context, not a fact source) and with the local-first, SQLite-based store.
- **Impact:** Story 5.1 delivery only.
- **Action:** Record the SQLite schema choice when delivering story 5.1.

### 14.8 UI citation mechanism (story 6.1) — IMPLEMENTATION DECISION (CONFIRMED)

- **Decision:** A simple HTML + JavaScript frontend served by FastAPI. No frontend framework (React, Vue, etc.). The frontend renders answer pages with clickable citation references that expand inline to display the passage excerpt, document name, and document location.
- **Citation mechanism requirements (from Architecture §8, Epics & Stories story 6.1, and PRFAQ §2):**
  - Clickable/expandable inline citation references in the answer text.
  - Actual stored passage text displayed on expansion (from the content store, not regenerated by the LLM).
  - Document name displayed per citation.
  - Document location displayed per citation.
  - Citation resolution performed by the backend/content store (`app.citation.resolver` using `app.store`).
  - No frontend framework required for MVP.
- **Why this is an explicit decision now:** The architecture (§8) and Epics & Stories (story 6.1) leave the UI mechanism open. The user has confirmed the MVP choice: HTML + JS served by FastAPI. This is recorded as a confirmed implementation decision, not an open item.
- **Constraint check:** Consistent with Architecture §8 (resolvability and passage availability are the architectural requirements; the UI mechanism is a UI decision) and Architecture §11 (FastAPI for the application layer). The architecture does not prescribe the UI mechanism, so this choice does not conflict with any architectural requirement. Consistent with the FastAPI tech stack in the project context.
- **Impact:** Story 6.1 delivery. Defines the UI technology for the MVP.
- **Action:** Record "HTML + JS frontend served by FastAPI, no framework, inline-expandable citations" when delivering story 6.1.

---

## A. Implementation Architecture

The implementation follows the four-layer pipeline from Architecture v1:

```
Documents → Ingestion (app/ingestion) → Content Store (app/store) → Retrieval (app/retrieval) → Generation (app/generation) → API (app/api)
                                                                                              ↑            ↑
                                                                                     Citation Resolver    Conversation
                                                                                     (app/citation)       (app/conversation)
```

**Cross-cutting:**
- `evaluation/` package runs in parallel, consuming the API and component interfaces.
- `tests/` package covers all layers.

**Key architectural invariants (from approved artifacts):**
- Model-agnostic LLM layer (`app/generation/interface.py`).
- Local-first, single-user, no authentication.
- No numerical confidence scores.
- Citation resolution from the store, not from the LLM.
- Interactive citations show the actual passage text.

---

## B. Repository Structure

See Section 1 for the full layout.

Key design decisions:
- `app/` is the FastAPI application package.
- `evaluation/` is a separate package, not part of the runtime application.
- `tests/` mirrors the `app/` structure for unit tests, with integration and evaluation test directories.
- `data/` is git-ignored; holds the SQLite database, embedding cache, and transient uploads.
- Configuration is in a TOML file (`config.toml`), with an example checked in.

---

## C. Component Contracts

See Section 2 for the full contracts. Summary of the key boundaries:

| Boundary | Producer | Consumer | Contract type |
|---|---|---|---|
| Ingestion → Chunking | `app.ingestion` | `app.chunking` | `RawDocument` → `ChunkingConfig` → `list[Passage]` |
| Chunking → Store | `app.chunking` | `app.store` | `list[Passage]` → persist |
| Store → Retrieval | `app.store` | `app.retrieval` | Passage read interface; index building |
| Semantic + BM25 → Hybrid | `app.retrieval.semantic`, `app.retrieval.keyword` | `app.retrieval.hybrid` | `RetrievalResult` from each; hybrid combines |
| Retrieval → Generation | `app.retrieval` | `app.generation` | `RetrievedContext` |
| Generation → Citation Resolution | `app.generation` | `app.citation.resolver` | `GeneratedAnswer` with citation IDs; resolver populates excerpts |
| Conversation → Retrieval/Generation | `app.conversation` | `app.retrieval`, `app.generation` | `ConversationTurn` list in `RetrievedContext` |
| Backend → API | domain layers | `app.api` | Domain function calls; API is thin |
| System → Evaluation Runner | application / components | `evaluation.runner` | `EvaluationSystem` abstraction |

---

## D. Data Models

See Section 3 for the full models. The core entities are:

- **Document** — persisted in SQLite, `documents` table.
- **Passage** — persisted in SQLite, `passages` table.
- **Citation** — runtime view-model, produced by generation, populated by resolver.
- **RetrievalResult** — runtime, produced by retrieval.
- **EvidenceQuality** — enumeration: sufficient, partial, insufficient, conflicting.
- **Answer / GeneratedAnswer** — runtime, produced by generation.
- **Session / ConversationTurn** — persisted in SQLite, `conversations` and `conversation_turns` tables.
- **EvaluationQuestion** — persisted as JSON in the evaluation dataset.
- **EvaluationResult** — produced by the evaluation runner.

---

## E. API Contracts

See Section 5 for the full API specification. Six endpoints plus session creation:

- `POST /api/v1/documents/upload` — upload a document
- `GET /api/v1/documents` — list documents
- `DELETE /api/v1/documents/{id}` — remove a document
- `POST /api/v1/query` — ask a question
- `GET /api/v1/citations/{passage_id}` — look up a passage
- `POST /api/v1/conversation` — create a session
- `POST /api/v1/conversation/{session_id}/follow-up` — follow-up question

All return JSON. No authentication. No numerical confidence scores.

---

## F. Database Schema

See Section 6 for the full schema. Four tables:

- `documents` — document records.
- `passages` — passage records, CASCADE delete from documents.
- `conversations` — session records.
- `conversation_turns` — conversation turns, CASCADE delete from conversations.

Embeddings stored as JSON arrays in SQLite. BM25 and semantic indices are in-memory structures maintained by the retrieval layer, not in SQLite.

---

## G. Evaluation Design

See Section 10 for the full design. Summary:

- **Dataset location:** `evaluation/dataset/`
- **Questions format:** JSON file with question objects including id, question_text, question_type, gold_answer, gold_passage_ids, conflicting_sources, closest_relevant_passage_id.
- **Runner:** `evaluation/runner.py`, with an `EvaluationSystem` abstraction that can target the API or components directly.
- **Metrics:** retrieval precision/recall, citation correctness, answer faithfulness, abstention correctness, conflict handling, latency.
- **Result format:** `EvaluationResult` objects written to `evaluation/dataset/results/`.
- **Component-level evaluation:** runner can evaluate retrieval only and generation only before the full API is available.
- **End-to-end evaluation:** runner evaluates the full API once available.

---

## H. Testing Strategy

See Section 11 for the full strategy. Nine test categories:

1. **Unit tests** — config, chunking, store, retrieval, generation, conversation.
2. **Integration tests** — ingestion → store, retrieval → generation, API end-to-end.
3. **Retrieval tests** — vocabulary mismatch, exact term, hybrid combination, scope filtering, candidate set size.
4. **Citation tests** — resolution, correctness against gold.
5. **Generation tests** — groundedness, faithfulness, evidence quality categories.
6. **Abstention tests** — insufficient evidence (abstain), sufficient evidence (don't abstain).
7. **Conflict tests** — two disagreeing sources, verify both named.
8. **API tests** — all endpoints, error handling, no confidence scores, no auth.
9. **Evaluation tests** — runner with mock system, component-level and end-to-end.

---

## I. Story-by-Story Implementation Map

See Section 13 for the full map. All 29 stories are mapped to:
- Files/modules likely to be created or modified
- Dependencies
- Interfaces consumed
- Interfaces produced
- Tests required
- Acceptance criteria mapping

The 29 stories break down as:
- Epic 0: 4 stories (0.1–0.4)
- Epic 1: 5 stories (1.1–1.5)
- Epic 2: 2 stories (2.1–2.2)
- Epic 3: 5 stories (3.1–3.5)
- Epic 4: 6 stories (4.1–4.6)
- Epic 5: 2 stories (5.1–5.2)
- Epic 6: 2 stories (6.1–6.2)
- Epic 7: 6 stories (7.1–7.6)

---

## J. Implementation Sequence

See Section 12 for the full sequence. Summary:

**Phase 0 (parallel, start immediately):**
- 0.1, 0.2, 0.3, 0.4 — evaluation dataset workstream

**Phase 1 (core pipeline):**
- Wave 1 (parallel): 1.1, 1.2, 1.3, 2.1, 2.2
- Wave 2: 1.4, 1.5
- Wave 3 (parallel): 3.1, 3.2, 3.4
- Wave 4: 3.3, 3.5
- Wave 5 (parallel): 4.1, 4.2, 4.3
- Wave 6: 4.4, 4.5, 4.6

**Phase 2 (user-facing, parallel once deps met):**
- Wave 7 (parallel): 5.1, 6.1
- Wave 8: 5.2, 6.2

**Phase 3 (integration):**
- Wave 9: 7.1 (contract-first, can be early)
- Wave 10: 7.2, 7.3, 7.4, 7.5
- Wave 11: 7.6 (last)

Key sequencing constraints:
- Story 0.1 before epics 2–4 proceed significantly.
- Story 0.4 before end-to-end testing of epic 7.
- Main chain: 1.x → 2.x → 3.x → 4.x → 7.x.

---

## K. Remaining Technical Decisions

See Section 14 for the full list. Eight decisions remain:

1. **PDF extraction library** (story 1.1) — pypdf recommended; not resolved in approved artifacts.
2. **Markdown parser** (story 1.2) — markdown2 or mistune recommended; not resolved.
3. **Embedding index structure** (story 3.1) — brute-force cosine similarity recommended for MVP; not resolved.
4. **BM25 index maintenance** (story 3.2) — rebuild on change recommended; not resolved.
5. **Synchronous vs. asynchronous ingestion** (stories 1.5, 7.2) — synchronous recommended; not resolved.
6. **LLM output parsing approach** (story 4.3) — JSON output with fallback recommended; not resolved.
7. **Conversation history storage format** (story 5.1) — JSON array in SQLite sufficient for MVP; not a blocking decision.
8. **UI technology for citation mechanism** (story 6.1) — simple web frontend (HTML + JS served by FastAPI) recommended; this is the one decision that affects how the application is presented and should be made explicit before story 6.1 is started.

None of these decisions expand MVP scope or contradict the approved artifacts. Items 1–7 are implementation details that can be decided when the relevant story is delivered. Item 8 is a UI-technology decision that should be made before story 6.1 begins.

---

## L. Readiness Assessment for Beginning Implementation

### What is ready

- **Product scope is locked.** PRFAQ, Architecture v1, and Epics & Stories are approved and finalized. No further product decisions are needed to begin implementation, except the 8 remaining technical decisions in Section 14 (all of which are implementation details, not product scope).
- **All stories have acceptance criteria.** 50 acceptance criteria across 29 stories. The criteria are rigorous around the core reliability requirements.
- **All 10 decision-resolution items are resolved.** 9 as MVP defaults incorporated into stories; 1 (evaluation dataset ownership) by explicit product decision.
- **Component contracts are defined.** Section 2 defines the interfaces between every major component. Implementation can proceed component by component without ambiguity about what each component consumes and produces.
- **Data models are defined.** Section 3 defines every entity. The database schema (Section 6) is defined. Implementation can create the schema and models directly from this document.
- **API contracts are defined.** Section 5 defines every endpoint. Implementation can build the API layer directly from this document.
- **Evaluation architecture is defined.** Section 10 defines the dataset format, runner, metrics, and result format. The evaluation workstream can begin immediately.
- **Testing strategy is defined.** Section 11 defines the test categories and what each must cover.
- **Implementation sequence is defined.** Section 12 defines the story-level sequence and parallelization opportunities.

### What is not yet ready

- **No implementation code has been written.** This is the planning stage.
- **No dependencies have been installed.** The specific libraries (sentence-transformers, rank_bm25, PDF library, Markdown parser, FastAPI) are identified but not installed.
- **No repository has been initialized.** The project structure in Section 1 is a plan, not a created filesystem.
- **The 8 remaining technical decisions in Section 14 should be made before or during the delivery of the relevant stories.** They are not blocking for the entire implementation — each blocks only its own story. The most impactful ones for early implementation are:
  - Item 1 (PDF library) — blocks story 1.1.
  - Item 2 (Markdown parser) — blocks story 1.2.
  - Item 3 (embedding index) — blocks story 3.1.
  - Item 5 (synchronous vs. asynchronous ingestion) — blocks stories 1.5 and 7.2.
  - Item 8 (UI technology) — blocks story 6.1.
- **The evaluation dataset (Epic 0) is not yet curated.** Stories 0.2 and 0.3 require actual documents and questions to be written. Story 0.1 (methodology) can be written immediately. Story 0.4 (runner scaffolding) can be started once story 0.3 has a few questions.

### Go/no-go for beginning implementation

**Go.** The planning artifacts are complete and consistent. Implementation can begin with:
1. Repository initialization and project structure setup (Section 1).
2. Configuration setup (Section 1, `app/config.py`).
3. SQLite schema and store (stories 2.1, 2.2).
4. Evaluation methodology (story 0.1) — in parallel.
5. PDF, Markdown, and TXT ingestion handlers (stories 1.1, 1.2, 1.3) — in parallel.

The remaining technical decisions (Section 14) should be made before their respective stories are started, but none of them block the very first implementation steps (store, config, methodology, and the three ingestion handlers can all proceed with reasonable default technology choices recorded as decisions when the stories are delivered).

**Not yet begun:** No code, no installs, no repository. This document is the implementation plan. The next step after review is to initialize the repository and begin Phase 0 / Wave 1 implementation.
