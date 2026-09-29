# DocuResearch — Architecture

**Status:** Draft v1  
**Last updated:** 2026-09-21  
**Input:** DocuResearch PRFAQ (finalized)  
**Model-agnostic:** Yes — LLM layer is swappable; MVP starts with a free model through Hermes  

---

## 1. Architectural Goals

The architecture exists to serve the product's core job: find specific information quickly and show exactly where it came from. Every architectural decision should be testable against that job.

The architecture must make the following measurable from early on:

- Retrieval precision and recall
- Citation correctness
- Answer faithfulness to retrieved evidence
- Unsupported-answer / hallucination rate
- Abstention accuracy
- Handling of conflicting evidence
- Response latency

The architecture should also be model-agnostic: the LLM used for answer generation, citation formatting, and evidence-quality narration is a pluggable component, not a structural dependency.

---

## 2. High-Level Structure

The system has four logical layers:

```
Documents ──▶ Ingestion ──▶ Retrieval ──▶ Generation ──▶ Response
                │              │              │
                ▼              ▼              ▼
          Content store    Passage index   Citation + evidence
          (chunks, meta)   (embeddings,   quality layer
                             metadata)     (abstention,
                                             conflict detection)
```

### Layer responsibilities

**Document ingestion**  
Accepts text-based PDF, Markdown, and TXT. Extracts text, splits into retrievable units (passages), and records metadata — document name, page or section where available, chunk offset, and a stable identifier for each passage. The passage is the atomic unit for retrieval and citation.

**Content store**  
Persists extracted passages and their metadata. This is what the interactive citation mechanism reads from when a user inspects a cited passage. It must preserve enough of the original text and location information to render an accurate passage excerpt.

**Retrieval**  
Given a query and optional document scope (all documents or one specific document), returns a ranked set of passages that are relevant to the query. The retrieval layer is the primary lever for precision and recall. It is also what determines whether a citation can point to a specific, inspectable passage.

**Generation**  
Takes the retrieved passages, the original query, and the conversation context, and produces an answer. The generation layer is responsible for:
- Grounding the answer in the retrieved passages (not the model's prior knowledge).
- Producing citations that point to specific passages in the content store.
- Narrating evidence quality (sufficient, partial, insufficient, conflicting).
- Abstaining when evidence is insufficient rather than fabricating.
- Staying within the uploaded documents during conversational follow-up.

**Citation + evidence quality layer**  
This is not a separate process step so much as a set of constraints and checks applied during generation and after. It includes:
- Citation generation: produce document name + page/section identifier + passage reference for each claim or claim-group in the answer.
- Evidence sufficiency assessment: determine, from the retrieved passages, whether the evidence supports the answer fully, partially, or not at all.
- Conflict surfacing: when retrieved passages from different sources disagree, surface the conflict rather than choosing one silently.
- Verification path: ensure every citation is resolvable to an actual passage in the content store (the interactive citation mechanism).

---

## 3. Data Model

### Document

```
Document
  - id: stable identifier
  - name: user-visible name
  - format: pdf | markdown | txt
  - uploaded_at: timestamp
  - page_count / section_count: where derivable from the format
```

### Passage

```
Passage
  - id: stable identifier (document-scoped, e.g. doc_id:page:chunk or doc_id:section:chunk)
  - document_id: foreign key to Document
  - text: the passage text (the retrievable and citable unit)
  - location: page number, section identifier, or other locator — whatever the ingestion pipeline can reliably produce
  - start_offset / end_offset: character offsets within the document text, for precise passage reconstruction
  - embedding: vector representation (used for semantic retrieval)
```

The `location` field is the basis for citations. Its granularity is a function of what the ingestion pipeline can reliably produce, not an architectural assumption. The architecture must not require page-level granularity to function; section-level or paragraph-level locators are acceptable, and the interactive citation mechanism (showing the actual passage text) is the fallback that makes granularity less critical.

### Query / Session

```
Query
  - id: stable identifier
  - text: the user's question or search term
  - scope: all documents | specific document (document_id)
  - session_id: conversational context, if any

Answer
  - query_id: foreign key
  - text: the generated answer
  - citations: list of { passage_id, document_name, location, excerpt }
  - evidence_quality: sufficient | partial | insufficient | conflicting (with narrative)
  - generated_at: timestamp
```

The `excerpt` in each citation is the actual passage text from the content store, not a paraphrase. This is what the interactive citation renders.

---

## 4. Ingestion Pipeline

### Format handlers

**Text-based PDF**  
Extract text with a PDF text extraction library. Preserve page boundaries where possible. Split into passages at page boundaries and/or semantic breakpoints (paragraphs, headings) depending on what produces the most citable units. Record page number as the `location` for each passage.

**Markdown**  
Parse into blocks (headings, paragraphs, code blocks, lists). Use the heading structure to produce section-level locators where the document has a clear hierarchy. Split long sections into sub-passages as needed.

**Plain text (TXT)**  
Split on paragraph or line-group boundaries. Locators are less structured — paragraph index or character offset range — but the interactive citation still works because the passage text is preserved.

### Chunking strategy

The chunking strategy is a key architectural decision because it determines citation granularity and retrieval quality. Early guidance:

- Chunks should be small enough to be citable as a meaningful unit (a paragraph or a few paragraphs), but large enough to contain sufficient context for the generation layer to use.
- Chunks should overlap at boundaries to avoid cutting a relevant passage in half.
- The chunking strategy should be configurable and evaluable — the evaluation dataset should be able to tell us whether a given chunking strategy produces better citation correctness and faithfulness than another.

### Metadata extraction

For each passage, record:
- Document reference
- Location (page, section, or paragraph index)
- Text
- Offsets within the document
- Embedding (computed at indexing time)

---

## 5. Retrieval Strategy

### Candidate retrieval

The retrieval layer is a hybrid of at least two signals:

**Semantic (embedding-based) retrieval**  
Embed the query and retrieve passages by vector similarity. This handles the case where the user's question uses different words than the document.

**Keyword / lexical retrieval**  
A keyword or BM25-style signal as a complement to semantic retrieval, especially for precise terms (part numbers, standard identifiers, specific technical terms) where semantic similarity alone may be weak.

The relative weight and combination of these signals is an architectural decision to be validated against the evaluation dataset, not a fixed assumption.

### Scoped retrieval

When the user restricts the query to a specific document, retrieval is constrained to that document's passages. When the query is across all documents, retrieval spans the full passage index. The scoped/uns-scoped distinction is a filter applied to the candidate set before ranking.

### Ranking

Candidates from both signals are combined and ranked. The ranking should prefer passages that are not only relevant but also citable — i.e., passages that contain enough text to support a citation. This is a soft preference, not a hard filter; the generation layer still decides what to cite.

### Reranking (possible later optimization)

A reranking step — using a model to score candidate passages for relevance to the specific query — may improve precision. This is an optimization, not an MVP requirement. The MVP can start with the combined semantic + keyword approach and add reranking if the evaluation dataset shows precision is insufficient.

---

## 6. Generation Layer

### Inputs

- The user's query
- The retrieved passages (ranked)
- The conversation history (for follow-up questions)
- The document scope context (which document(s) were searched)

### Prompting posture

The generation layer is prompted to:
1. Answer the question using only the retrieved passages as source material.
2. Cite specific passages for claims, using the passage metadata (document name, location).
3. Assess and narrate evidence quality: does the retrieved evidence fully support the answer, partially support it, not support it, or contain conflicting information?
4. Abstain or qualify the answer when evidence is insufficient, rather than generating a plausible answer from the model's prior knowledge.
5. Stay within the uploaded documents — do not introduce outside knowledge as if it were from the documents.

### Citation generation

Citations are generated as structured output (not free text), tied to passage IDs in the content store. Each citation must resolve to an actual passage. The interactive citation UI renders the `excerpt` field from the content store for that passage.

The citation format in the answer text is a reference (e.g. a footnote, an inline expandable reference, or a highlighted passage) that the UI resolves to the passage excerpt. The architecture does not prescribe the exact UI mechanism; it requires that the citation be resolvable to a passage and that the passage text be available.

### Evidence quality narration

The generation layer produces the evidence quality narrative as part of the answer, not as a separate metadata field that the user has to look for. The narrative uses the categories defined in the PRFAQ:

- **Sufficient**: "The documents support this directly."
- **Partial**: "The documents support part of this. [X] appears in [source]; the rest is a reasonable inference but not stated explicitly."
- **Insufficient**: "The documents do not contain enough information to answer this definitively."
- **Conflicting**: "The documents disagree. [Source A] says X; [Source B] says Y."

The evidence quality assessment is based on the retrieved passages, not on a model-internal confidence score. The architecture should not expose a numerical confidence score to the user.

### Abstention

When the retrieved passages do not contain sufficient evidence to answer the question, the generation layer should say so explicitly. The abstention should identify the closest relevant passage if one exists, and explain what is and is not supported.

Abstention is a first-class output, not a failure mode. The evaluation metric "abstention accuracy" measures whether the system abstains when it should and does not abstain when it has sufficient evidence.

---

## 7. Conversational Follow-Up

Follow-up questions are treated as new queries with conversation context. The context includes:
- The original query and answer (so the system can refer back to what was already established)
- The document scope from the original query (unless the user changes it)
- Any clarifying information from the conversation

The generation layer uses the context to keep follow-up answers grounded and coherent, but each follow-up is still a retrieval + generation cycle. The system does not "remember" outside the documents; it uses the conversation history as context for the query, not as an additional source of facts.

---

## 8. Interactive Citation Mechanism

The interactive citation is the core trust mechanism. Architecturally, it requires:

1. **Citation references in the answer** that point to passage IDs in the content store.
2. **A content store query** that retrieves the passage text and metadata given a passage ID.
3. **A UI mechanism** (the exact form is an open question) that displays the passage excerpt when the user interacts with the citation.

The architecture does not prescribe the UI mechanism. It requires that the citation be resolvable and that the passage text be available at render time. The open question from the PRFAQ — whether the interaction is an inline expansion, a side panel, or a document highlight — is a UI decision that the architecture should leave open.

---

## 9. Evaluation Dataset (Early Action)

As specified in the PRFAQ (Section 8), an early project action is to begin designing and curating a representative evaluation dataset in parallel with architecture and implementation work.

### What the dataset enables

The dataset is the foundation for validating the architecture. It should allow measurement of:

- **Retrieval precision and recall**: does the retrieval layer return the right passages, and all the passages needed?
- **Citation correctness**: do citations point to passages that actually support the claims?
- **Answer faithfulness**: is the answer consistent with the retrieved passages?
- **Unsupported-answer / hallucination rate**: how often does the generation layer produce an answer not grounded in the retrieved passages?
- **Abstention accuracy**: does the system abstain when evidence is insufficient, and not abstain when evidence is sufficient?
- **Handling of conflicting evidence**: when sources conflict, does the system surface the conflict?
- **Response latency**: end-to-end time from query to answer.

### Early dataset design goals

The early effort does not build the complete dataset before architecture or implementation begins. The goals are:

1. **Establish evaluation methodology**: define what "good" looks like for each metric and how it will be scored. Agree on pass/fail or graded criteria where possible.

2. **Curate representative documents**: select documents in the MVP formats (text-based PDF, Markdown, TXT) that vary in length, structure, and the kind of verification work they represent. Include documents that are internally consistent and documents that contain conflicting information across sections or across documents.

3. **Curate representative questions**: include a range of question types:
   - Clear, single-source answer (one passage supports the answer directly).
   - Multi-passage synthesis (the answer draws on more than one passage).
   - Partial evidence (the documents support part of the answer but not all).
   - Insufficient evidence (the documents do not contain enough to answer).
   - Conflicting evidence (two sources disagree).

4. **Record gold answers and source locations**: for each question, record the expected answer and the passage(s) that support it, with their identifiers. This is what makes citation correctness and faithfulness measurable.

### How the dataset informs architecture

As the dataset grows, it should be used to validate architectural decisions:

- **Chunking strategy**: does a given chunking approach produce better citation correctness and faithfulness on the dataset than another?
- **Retrieval approach**: does semantic + keyword retrieval with a given weighting produce better precision and recall than semantic-only?
- **Evidence quality heuristics**: do the rules for "sufficient" vs "partial" vs "insufficient" produce accurate evidence quality narration on the dataset?
- **Generation prompting**: does the prompting posture produce answers that are faithful to the retrieved passages and that abstain appropriately?

The dataset is a living artifact. It grows over time, and the architecture is validated against it at milestones, not just at the end.

---

## 10. Model-Agnostic LLM Layer

The LLM used for generation (answer text, citation formatting, evidence quality narration) is a pluggable component. The architecture defines the interface — inputs (query, passages, context) and outputs (answer text, structured citations, evidence quality narrative, abstention flag) — but not the specific model.

Start with a free model through Hermes for the agent layer. If the evaluation dataset shows that citation quality, faithfulness, or evidence narration is insufficient with the free model, upgrading to a paid model is a configuration change, not an architectural rewrite.

The architecture should make the model swap straightforward: the generation layer's prompt and output schema are the contract; the model is an implementation detail.

---

## 11. Technology Choices (Initial Direction, Not Final)

These are initial directions to guide implementation, not final commitments. They should be validated as the architecture is exercised against the evaluation dataset.

### Document ingestion
- PDF text extraction: a Python PDF library that handles text-based PDFs (e.g. pypdf, pdfplumber). Avoid OCR-dependent libraries for MVP.
- Markdown parsing: a Python Markdown parser that exposes block structure.
- Plain text: standard text handling; splitting on paragraph boundaries.

### Vector storage and retrieval
- Embedding model: initially a lightweight embedding model; swappable.
- Vector store: a local vector store appropriate for a single-user MVP. The choice should support the hybrid retrieval approach (semantic + keyword).

### Generation
- LLM: model-agnostic; start with the free model through Hermes.
- Framework: the generation layer should be structured so that the prompt, retrieved passages, and output schema are clearly separated from the model call.

### Content store
- A local store (database or file-based) that persists documents, passages, and metadata. Must support passage lookup by ID for the interactive citation mechanism.

### Application layer
- FastAPI, per the project tech stack. The API exposes document upload, query, and citation lookup endpoints.

---

## 12. What Is Out of Scope for MVP Architecture

- **DOCX ingestion**: no DOCX handler in the ingestion pipeline for MVP.
- **Scanned PDFs / OCR**: no OCR pipeline for MVP. Text-based PDFs only.
- **Dedicated contradiction-detection mode**: the architecture surfaces conflicts from the retrieved passages during generation, but there is no separate contradiction-detection feature or report.
- **Authentication, billing, multi-tenancy**: single-user, local-first. No auth layer, no tenant isolation.
- **Numerical confidence scores**: not part of the generation output or the API.
- **Reranking**: optional optimization, not an MVP requirement.
- **Evaluation dataset completion**: the early action is to begin curating; the full dataset is not an MVP prerequisite.

---

## 13. Open Architectural Questions

These are decisions that the architecture document flags but does not resolve. They should be resolved as the architecture is validated against the evaluation dataset and as implementation begins.

1. **Chunking granularity and strategy.** What chunk size and overlap produce the best balance of citation usefulness and retrieval quality? The evaluation dataset should answer this.

2. **Semantic + keyword weighting.** What relative weight produces the best precision and recall on representative questions? The evaluation dataset should answer this.

3. **Citation granularity floor.** What is the minimum locator granularity the system can rely on? Page-level is the aspiration; the architecture must function with section-level or paragraph-level locators where page-level is not available. The interactive citation mechanism is the fallback.

4. **Evidence quality threshold rules.** What specific rules determine "sufficient" vs "partial" vs "insufficient" vs "conflicting"? The architecture leaves this to the generation layer's prompting and heuristics; the evaluation dataset should be used to validate and refine the rules.

5. **Interactive citation UI.** The architecture requires resolvability and passage availability; the UI mechanism (inline expansion, side panel, document highlight) is a UI decision, not an architecture decision.

6. **Embedding model and vector store choice.** Initial directions are noted in Section 11; specific choices should be made when implementation begins and validated against the dataset.

---

## 14. Summary

The DocuResearch architecture is built around a four-layer pipeline — ingestion, retrieval, generation, and citation/evidence quality — with a content store that supports interactive citations. The architecture is model-agnostic and makes the evaluation metrics from the PRFAQ measurable from early on. An early action is to begin curating a representative evaluation dataset in parallel with implementation, so that architectural decisions — chunking, retrieval strategy, evidence quality heuristics, generation prompting — can be validated against real questions and documents rather than assumptions. The MVP covers text-based PDF, Markdown, and TXT; DOCX, OCR, and dedicated contradiction detection are out of scope.
