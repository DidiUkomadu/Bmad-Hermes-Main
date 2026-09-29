# DocuResearch — Epics and Stories

**Status:** Draft v1  
**Last updated:** 2026-09-21  
**Source of truth:** DocuResearch PRFAQ (finalized) + DocuResearch Architecture (v1)  
**Implementation language:** Python  
**Framework:** FastAPI  
**MVP document formats:** text-based PDF, Markdown, TXT  
**Out of MVP scope:** DOCX, scanned PDFs/OCR, dedicated contradiction-detection feature, reranking, authentication, billing, multi-tenancy, enterprise administration, numerical confidence scores  
**Implementation start:** Not yet — this document is the planning artifact  

---

## Epic 0: Evaluation Dataset (Enabling Workstream)

**Type:** Enabling / parallel workstream  
**Priority:** Start immediately, in parallel with all other epics  
**Depends on:** Nothing — this epic has no technical dependencies and is intended to inform validation across all other epics  
**Other epics depend on:** This epic's output is used to validate epics 1–6. It should be active before implementation begins in earnest, and mature alongside implementation.

**Ownership and curation:** I (the product owner) own and maintain the DocuResearch evaluation dataset as a living project artifact. Hermes assists with the curation scaffolding, dataset structure, validation scripts, and updates required during development. The dataset is updated as new failure modes or question types are discovered. Curation happens in parallel with implementation, with enough representative documents and questions available to evaluate each major milestone. The evaluation methodology (story 0.1) must be established before epics 2–4 proceed significantly. The evaluation runner (story 0.4) must be available before end-to-end testing of Epic 7.  

**Rationale:** The PRFAQ and architecture both specify that the evaluation dataset is an early action, not a post-implementation activity. Reliability and verifiability are core to the product's value proposition, so the methodology for measuring them must be established before architectural decisions are locked in. This epic is the vehicle for that work.

### Story 0.1: Define evaluation methodology and metric definitions

**Goal:** Establish what "good" means for each evaluation metric and how it will be scored, so that later implementation decisions can be validated consistently.

**Acceptance criteria:**
- [ ] A written evaluation methodology document exists that defines each of the following metrics in operational terms:
  - Retrieval precision: the fraction of retrieved passages that are relevant to the query.
  - Retrieval recall: the fraction of all relevant passages that were retrieved.
  - Citation correctness: the fraction of citations in generated answers that point to passages that actually support the claim made.
  - Answer faithfulness: the fraction of generated answers that do not introduce claims unsupported by the retrieved passages.
  - Unsupported-answer / hallucination rate: the rate at which the system produces an answer containing claims not grounded in the retrieved passages.
  - Abstention accuracy: the rate at which the system correctly abstains when evidence is insufficient, and does not abstain when evidence is sufficient.
  - Handling of conflicting evidence: whether the system surfaces conflicts when sources disagree, and does not silently choose one side.
  - Response latency: end-to-end time from query submission to answer delivery.
- [ ] Each metric has a defined scoring approach (binary pass/fail where possible, or a clear grading rubric).
- [ ] The methodology document specifies what constitutes a "relevant passage" for retrieval precision/recall, so that gold standard judgments are consistent.
- [ ] The methodology document specifies what constitutes a "correct citation" (passage must contain the claim, not merely be topically related).
- [ ] The methodology document specifies how partial-evidence cases are labeled and how the system's evidence-quality narration is evaluated against the label.
- [ ] The methodology document specifies how conflicting-evidence cases are labeled and what "surfacing the conflict" means in an evaluable way.
- [ ] The methodology document identifies the question types that must be represented in the dataset (single-source, multi-passage synthesis, partial evidence, insufficient evidence, conflicting evidence).
- [ ] The methodology document is reviewed and accepted as the evaluation standard before implementation of epics 2–6 proceeds significantly.

**Ownership note:** Per the product decision on Epic 0, I own and maintain the evaluation dataset as a living project artifact. Hermes assists with the curation scaffolding, dataset structure, validation scripts, and updates required during development. This story's output (the methodology document) is the foundation; the commitment to maintain the dataset over time is a product-level decision settled separately.

**Output:** Evaluation methodology document (living artifact).

### Story 0.2: Begin curating representative documents

**Goal:** Start building a corpus of representative documents in MVP formats that exercise the full range of retrieval and verification scenarios the MVP must handle.

**Acceptance criteria:**
- [ ] A document curation plan exists that identifies the characteristics the corpus must cover: varying length, varying structure (well-sectioned vs. flat), technical specificity, internal consistency, and cross-document conflict.
- [ ] At least a small initial set of documents is curated in each MVP format (text-based PDF, Markdown, TXT).
- [ ] The curated documents include at least one case where two documents (or two sections of one document) contain conflicting information on the same question.
- [ ] The curated documents include at least one case where the answer to a reasonable question is genuinely not present in the document (for abstention evaluation).
- [ ] The curated documents include at least one case where evidence is clearly partial (the document supports part of a reasonable answer but not all of it).
- [ ] Each curated document is tagged with metadata about its characteristics (format, approximate length, structure type, known conflict/partial/insufficient cases it contains).
- [ ] Documents are stored in a location accessible to the ingestion pipeline (epic 1) once it is ready.

**Output:** Initial document corpus with metadata, stored for ingestion.

### Story 0.3: Begin curating representative questions with gold answers and source locations

**Goal:** Start building the question set that the evaluation dataset uses to measure retrieval, citation, faithfulness, abstention, and conflict handling.

**Acceptance criteria:**
- [ ] A question curation plan exists that specifies the distribution of question types the dataset must contain: single-source answers, multi-passage synthesis, partial evidence, insufficient evidence, and conflicting evidence.
- [ ] For each curated question, the following are recorded:
  - The question text.
  - The type of answer expected (single-source, multi-passage, partial, insufficient, conflicting).
  - The gold answer (what a fully correct, fully grounded answer would say).
  - The passage identifier(s) that support the gold answer, with their document and location.
  - For insufficient-evidence questions: a record that no passage supports a definitive answer, and the closest relevant passage if one exists.
  - For conflicting-evidence questions: the two (or more) sources that conflict, what each says, and the passages that contain each claim.
- [ ] The initial question set includes at least one question of each type (single-source, multi-passage, partial, insufficient, conflicting).
- [ ] The gold answers and source locations are reviewed for accuracy against the curated documents before being used as evaluation references.
- [ ] Questions and gold data are stored in a structured format (e.g. JSON or structured markdown) that can be read by an evaluation script once the system is implemented.

**Output:** Initial question set with gold answers and source locations, in structured form.

### Story 0.4: Establish evaluation run capability (scaffolding)

**Goal:** Create the minimal scaffolding needed to run the evaluation dataset against the system once parts of it are implemented, so that validation can begin early rather than at the end.

**Acceptance criteria:**
- [ ] An evaluation runner script or structure exists that can, given a system endpoint or interface, submit a question from the dataset, collect the system's answer and citations, and compare them against the gold answer and source locations.
- [ ] The runner can compute at least the following from a single evaluation run: retrieval precision, retrieval recall, citation correctness, answer faithfulness (binary: faithful or not), and abstention classification (did the system abstain, and was that correct?).
- [ ] The runner can record response latency for each question.
- [ ] The runner produces a results summary that can be read and interpreted without specialized tooling.
- [ ] The runner is structured so that new questions and new metrics can be added without rewriting the core logic.
- [ ] The runner does not require the full system to be complete — it can run against individual components (e.g. retrieval only) as those become available.

**Output:** Evaluation runner scaffolding, ready to use against implemented components.

---

## Epic 1: Document Ingestion Pipeline

**Type:** Core capability  
**Priority:** High — foundational; must be available before the content store (epic 2) can be populated and before retrieval (epic 3) can have anything to retrieve  
**Depends on:** None (this epic has no internal technical dependencies on other epics)  
**Other epics depend on:** Epic 2 (content store) requires ingestion output; epic 3 (retrieval) requires content store; epic 5 (generation) requires retrieval.  

**Scope:** Accept text-based PDF, Markdown, and TXT; extract text; split into passages; record metadata including document name, location (page/section/paragraph), chunk offset, and stable passage identifier. Does not include DOCX, scanned PDFs, or OCR.

### Story 1.1: PDF text extraction with page boundaries (text-based PDFs only)

**Goal:** Extract text from text-based PDFs while preserving page boundaries, so that page-level citations are available for PDFs that contain text layering.

**Acceptance criteria:**
- [ ] A PDF ingestion function accepts a text-based PDF file and returns extracted text organized by page.
- [ ] Page boundaries are preserved: the output identifies which text belongs to which page number.
- [ ] The function handles PDFs that contain a text layer (the MVP format). It does not claim to handle scanned/image PDFs; behavior on such PDFs is to either fail clearly or extract nothing, not to silently return garbage.
- [ ] Extracted text is returned in a form suitable for chunking (passage creation), with page number recorded for each segment.
- [ ] Metadata recorded for the document includes: document name, format (pdf), page count (where derivable), and upload timestamp.
- [ ] The extraction behavior is tested against at least one representative text-based PDF and the output is verified to contain the expected text organized by page.
- [ ] Error handling is defined for corrupt or unreadable PDFs: the system returns a clear error, not a silent failure or partial result presented as success.

**Out of scope:** OCR, scanned PDFs, form filling, table extraction (unless it falls out of text extraction naturally).

### Story 1.2: Markdown parsing with heading-based sectioning

**Goal:** Parse Markdown documents into structured blocks (headings, paragraphs, code blocks, lists) and use heading structure to produce section-level locators where the document has a clear hierarchy.

**Acceptance criteria:**
- [ ] A Markdown ingestion function accepts a Markdown file and parses it into blocks.
- [ ] Heading structure is captured: each heading and its level (H1, H2, etc.) is identifiable, so that section locators can be constructed from the heading hierarchy.
- [ ] Non-heading blocks (paragraphs, code blocks, lists) are identifiable as distinct units for chunking.
- [ ] Section locators are produced where the document has headings: a passage's location can reference the nearest enclosing heading(s) as a section identifier.
- [ ] For Markdown documents without clear heading structure, a fallback locator (paragraph index or sequential block identifier) is available.
- [ ] The output is suitable for chunking: each block or group of blocks can be converted into a passage with an associated location.
- [ ] The parsing handles common Markdown constructs expected in technical documents (nested lists, code blocks, headings of multiple levels).

**Out of scope:** Rendering Markdown to HTML or other formats (this is ingestion, not presentation).

### Story 1.3: Plain text ingestion with paragraph-based chunking

**Goal:** Ingest plain text (TXT) files, splitting into passage units at paragraph or line-group boundaries, with a usable locator (paragraph index or offset range).

**Acceptance criteria:**
- [ ] A TXT ingestion function accepts a plain text file and returns the text organized into passage-able units.
- [ ] The splitting strategy uses paragraph boundaries (blank-line-separated groups) as the primary chunking unit.
- [ ] For documents where paragraph boundaries are ambiguous or absent, a fallback splitting strategy (e.g. fixed-size chunks with overlap, or line-group boundaries) is available and documented.
- [ ] Each unit has an associated locator: paragraph index, line range, or character offset range, whichever is most useful and reliably derivable.
- [ ] The output is suitable for passage creation: each unit can become a passage with text, location, and offsets.
- [ ] Metadata recorded for the document includes: document name, format (txt), and upload timestamp.

### Story 1.4: Chunking strategy — configurable passage creation from ingestion output

**Goal:** Convert ingestion output (per-page text for PDFs, per-block text for Markdown, per-paragraph text for TXT) into passages with stable identifiers, location metadata, and offsets, using a configurable chunking strategy.

**Acceptance criteria:**
- [ ] A chunking function takes ingestion output and produces a list of passages.
- [ ] Each passage has: a stable identifier (document-scoped, incorporating location information), the passage text, a location string (page number, section identifier, or paragraph index as appropriate to the format), start and end character offsets within the document text, and the document reference.
- [ ] The chunking strategy is configurable: chunk size (target passage length), overlap between adjacent chunks, and boundary preferences (e.g. prefer paragraph boundaries, allow mid-paragraph splits) can be adjusted without rewriting the function.
- [ ] Overlap is implemented: adjacent chunks share some text at their boundary, so that a relevant passage is not lost because it falls on a chunk boundary.
- [ ] The chunking function is tested with at least one representative document per format and the resulting passages are verified to cover the document text without gaps and with acceptable overlap.
- [ ] The chunking strategy is documented enough that it can be changed and re-evaluated against the evaluation dataset (epic 0) — i.e., the parameters are visible and adjustable, not hardcoded in a way that prevents experimentation.

Initial default chunking parameters (to be refined against the evaluation dataset): target passage length 250–600 characters; overlap 80–150 characters between adjacent passages; boundary preference — use paragraph boundaries (PDF, TXT) or heading/section boundaries (Markdown) as chunk breakpoints when they fall within the target range, otherwise use a sized chunk with overlap. These defaults are documented here so they can be adjusted and re-evaluated; they are not architectural commitments.

### Story 1.5: Ingestion orchestration — end-to-end document intake

**Goal:** Wire the format-specific handlers (stories 1.1, 1.2, 1.3) and the chunking strategy (story 1.4) into an end-to-end ingestion flow that accepts an uploaded document and produces stored passages.

**Acceptance criteria:**
- [ ] An orchestration function accepts an uploaded document (with its format identified) and routes it to the correct format handler.
- [ ] The orchestration produces a complete set of passages for the document, using the chunking strategy.
- [ ] The orchestration records document-level metadata (name, format, upload timestamp, page/section count where derivable).
- [ ] The orchestration handles format detection: given a file, it identifies whether it is PDF, Markdown, or TXT (by extension and/or content inspection) and routes accordingly. Behavior on an unrecognized format is a clear error, not silent fallback.
- [ ] The orchestration is integrated with the content store (epic 2) so that produced passages are persisted — this story's acceptance criteria assume the content store is available; if the content store is not yet implemented, this story produces passages and metadata in a form that the content store can consume when ready.
- [ ] End-to-end ingestion is tested with at least one document of each MVP format and the resulting stored passages are verified to exist, be retrievable by ID, and carry correct metadata.

**Depends on:** Stories 1.1, 1.2, 1.3, 1.4; and on epic 2 (content store) for persistence. If epic 2 is not yet available, this story's acceptance criteria are adjusted to "produces passages and metadata in consumable form."

---

## Epic 2: Content Store

**Type:** Core capability  
**Priority:** High — required for persistence of ingested documents and passages, and for the interactive citation mechanism (epic 6)  
**Depends on:** Epic 1 (ingestion) for the data to store.  
**Other epics depend on:** Epic 3 (retrieval) reads from the content store; epic 5 (generation) consumes passages from retrieval; epic 6 (citation UI) reads passages by ID from the content store.  

**Scope:** Persist documents and passages with metadata. Support retrieval of a passage by its stable identifier, and retrieval of all passages for a given document. Single-user, local-first — no multi-tenancy.

### Story 2.1: Document and passage persistence

**Goal:** Provide a store that persists documents and their passages with full metadata, so that ingested content survives across queries and sessions.

**Acceptance criteria:**
- [ ] The content store accepts a document record (ID, name, format, upload timestamp, page/section count) and stores it durably.
- [ ] The content store accepts a passage record (ID, document ID, text, location, start offset, end offset, embedding placeholder or null) and stores it durably, associated with its document.
- [ ] A document can be retrieved by its ID, returning its metadata.
- [ ] All passages for a given document can be retrieved by the document ID, returning each passage's full data (text, location, offsets).
- [ ] A single passage can be retrieved by its passage ID, returning its full data including the text — this is the operation the interactive citation mechanism (epic 6) depends on.
- [ ] The store is local-first and single-user: no tenant isolation, nomulti-user concurrency controls beyond what is needed for a single-user application.
- [ ] The choice of storage backend (file-based, SQLite, or other local store) is made and documented as part of this story. The backend choice should be appropriate for the MVP scale and should support the required operations without requiring external services.

Initial storage backend: SQLite, via the Python standard library sqlite3 module. This is the MVP default — it is file-based and local (consistent with the architecture's "local store" requirement and the single-user, local-first posture), requires no separate server, and supports all required content store operations natively. The schema (documents table, passages table, relationships, indexes for passage lookup by ID) is defined as part of this story. This is the initial default; the architecture requires only that the store be local and support the specified operations — SQLite is a concrete choice that satisfies both.

### Story 2.2: Passage deletion and document removal

**Goal:** Support removing a document and all its passages from the store, so that users can clean up uploaded documents.

**Acceptance criteria:**
- [ ] Removing a document by ID also removes all passages associated with that document.
- [ ] After removal, the document and its passages are no longer retrievable from the store.
- [ ] Removal does not affect other documents or their passages.
- [ ] The removal operation is tested: after removing a document, querying for that document returns a clear "not found" result and querying for passages by the removed document's ID returns an empty set.

---

## Epic 3: Retrieval Pipeline

**Type:** Core capability  
**Priority:** High — required for the generation layer to have anything to work with; primary lever for precision and recall  
**Depends on:** Epic 2 (content store) for the passage corpus; epic 1 (ingestion) indirectly, since ingestion populates the store.  
**Other epics depend on:** Epic 5 (generation) consumes retrieval output.  

**Scope:** Given a query and an optional document scope, return a ranked set of relevant passages. Hybrid semantic + keyword retrieval. Does not include reranking (deferred).

### Story 3.1: Semantic (embedding-based) retrieval

**Goal:** Implement a retrieval path that embeds the query and retrieves passages by vector similarity, so that questions using different words than the document can still find relevant passages.

**Acceptance criteria:**
- [ ] An embedding function is available that converts a text query into a vector representation.
- [ ] An embedding function is available that converts a passage's text into a vector representation (or passages are embedded at indexing time and stored).
- [ ] A semantic retrieval function accepts a query embedding and returns a ranked list of passages by vector similarity, with similarity scores.
- [ ] The embedding model is swappable: the retrieval function depends on an abstraction (embed text → vector) rather than a specific model, so that the model can be changed without rewriting the retrieval logic.
- [ ] Semantic retrieval is tested with at least one query where the relevant passage uses different words than the query, and the relevant passage is returned with a higher rank than irrelevant passages.
- [ ] Semantic retrieval returns passage identifiers (not just scores), so that the results can be resolved to full passages via the content store.

Initial embedding model: all-MiniLM-L6-v2 via the sentence-transformers library. This is the MVP default and is swappable — the semantic retrieval path depends on the embedding abstraction (text → vector), not on this specific model. The model is chosen as a lightweight, locally runnable option consistent with the local-first MVP posture; it should be validated against the evaluation dataset and can be replaced if the dataset shows it is insufficient for retrieval quality.

### Story 3.2: Keyword / lexical retrieval (BM25-style)

**Goal:** Implement a retrieval path that finds passages by term matching, so that precise technical terms, identifiers, and exact phrases are found even when semantic similarity is weak.

**Acceptance criteria:**
- [ ] A keyword retrieval function is available that indexes passage text and retrieves passages matching a query's terms.
- [ ] The keyword retrieval returns a ranked list of passages with relevance scores based on term matching.
- [ ] Keyword retrieval is tested with at least one query containing a precise technical term or identifier that appears verbatim in a passage, and that passage is returned.
- [ ] Keyword retrieval returns passage identifiers, not just scores.
- [ ] The keyword retrieval implementation is swappable at the abstraction level: the retrieval function depends on an abstraction (query + indexed passages → ranked passage IDs + scores) rather than a specific library, so the implementation can be changed without rewriting consumption logic.

Initial keyword retrieval implementation: rank_bm25 Python library. This is the MVP default and is swappable — the keyword retrieval path depends on the abstraction (query + indexed passages → ranked passage IDs + scores), not on this specific library. The library is chosen as a pure-Python BM25 implementation consistent with the local-first MVP posture.

### Story 3.3: Hybrid retrieval — combining semantic and keyword signals

**Goal:** Combine semantic and keyword retrieval into a single retrieval operation that uses both signals, with a configurable relative weighting.

**Acceptance criteria:**
- [ ] A hybrid retrieval function accepts a query and returns a ranked list of passage IDs, combining results from the semantic path (story 3.1) and the keyword path (story 3.2).
- [ ] The relative weighting of semantic vs. keyword signals is configurable (e.g. a parameter that controls the blend), so that the weighting can be adjusted and evaluated against the evaluation dataset.
- [ ] The combination logic is documented: how scores from the two paths are normalized and combined into a single ranking.
- [ ] Hybrid retrieval is tested with at least one query where semantic-only retrieval would miss a relevant passage (because of vocabulary mismatch) and keyword-only retrieval would miss a relevant passage (because the exact term is not present but the concept is), and the combined retrieval returns the relevant passage in both cases.
- [ ] Hybrid retrieval returns passage identifiers with combined scores, and the results can be resolved to full passages via the content store (epic 2).

Initial default hybrid weighting: 0.5 semantic + 0.5 keyword, with scores from each signal normalized to a common 0–1 scale before combination. This is the MVP starting baseline and is explicitly provisional — the weighting parameter must be configurable and should be adjusted based on evaluation dataset results. Equal weighting is the neutral starting point; it is not an architectural assertion that both signals are equally valuable.

### Story 3.4: Document scope filtering

**Goal:** Support restricting retrieval to a specific document or searching across all documents, as required by the PRFAQ.

**Acceptance criteria:**
- [ ] The retrieval function accepts an optional document scope parameter: either "all documents" or a specific document ID.
- [ ] When scoped to a specific document, retrieval returns passages only from that document, regardless of the hybrid retrieval signals.
- [ ] When scoped to all documents, retrieval searches the full passage corpus.
- [ ] Scope filtering is applied as a constraint on the candidate set, not as a post-filter that silently drops results without affecting ranking — i.e., the scope is respected in a way that is consistent with the retrieval logic.
- [ ] Scoped retrieval is tested: querying with a specific document scope returns passages only from that document, and querying with all-documents scope returns passages from all documents.
- [ ] The query interface (UI and/or API) exposes the scope choice to the user: the user can choose to search all documents or a specific document.

### Story 3.5: Retrieval result ranking and candidate set size

**Goal:** Produce a ranked candidate set of passages for the generation layer, with a defined maximum size, so that generation receives a manageable and relevant set of passages.

**Acceptance criteria:**
- [ ] The retrieval output is a ranked list of passage IDs with scores, suitable for consumption by the generation layer.
- [ ] A maximum candidate set size is defined (e.g. top N passages), and the retrieval function returns no more than that number. The value of N is documented and adjustable.
- [ ] The ranking is stable and deterministic given the same query and corpus (no random tie-breaking that would make evaluation results irreproducible).
- [ ] The retrieval function does not silently drop the top-ranked passage due to a boundary condition (e.g. off-by-one in the candidate truncation).
- [ ] Retrieval latency is measured and recorded as part of the evaluation dataset's latency metric (epic 0). The retrieval function is instrumented to record its own processing time.
- [ ] Retrieval results are validated against the evaluation dataset (epic 0) for precision and recall once the dataset and runner (stories 0.2, 0.3, 0.4) are available.

Initial default maximum candidate set size: 20 passages (top 20 by combined hybrid score). This is the MVP default and is explicitly configurable — it should be adjusted based on evaluation dataset results (increase if recall is low; decrease if latency or context quality is a problem). The value is documented here as the starting point, not as an architectural commitment.

---

## Epic 4: Generation Layer — Model Integration and Output Schema

**Type:** Core capability  
**Priority:** High — responsible for grounded answers, citations, evidence quality narration, and abstention  
**Depends on:** Epic 3 (retrieval) for passages; epic 2 (content store) indirectly via retrieval; the LLM provider (external, through Hermes).  
**Other epics depend on:** Epic 6 (citation UI) consumes the generation layer's citation output; the application layer (epic 7) exposes the generation layer's answer output.  

**Scope:** Define the interface between the application and the LLM, structure the generation prompt, and define the output schema (answer text, structured citations, evidence quality narrative). Does not include the interactive citation UI (epic 6) or the API endpoints (epic 7).

### Story 4.1: Model-agnostic LLM interface

**Goal:** Define and implement an abstraction for the LLM that the generation layer uses, so that the model can be swapped without rewriting the generation logic.

**Acceptance criteria:**
- [ ] An LLM interface abstraction is defined with a clear contract: given a prompt (constructed from query, retrieved passages, and context), return a structured response containing answer text, citations, and evidence quality information.
- [ ] The interface is not tied to a specific model provider or API. Swapping the model is a configuration change that implements the same interface.
- [ ] The initial implementation uses the free model available through Hermes for the agent layer, as specified in the PRFAQ and architecture.
- [ ] The interface handles the case where the model returns an error or malformed response: the generation layer receives a clear failure signal, not a silently degraded result.
- [ ] The interface is tested with a sample prompt and the structured response is verified to contain the expected fields.

**Open decision:** The specific model is the initial choice (free model through Hermes). The interface is what makes a future swap a configuration change.

### Story 4.2: Generation prompt construction

**Goal:** Build the prompt that is sent to the LLM, incorporating the query, retrieved passages, conversation context, and the generation posture defined in the architecture.

**Acceptance criteria:**
- [ ] The prompt construction function takes as input: the user's query, the retrieved passages (with their text and metadata), the conversation context (if any), and the document scope context.
- [ ] The prompt instructs the model to: answer using only the retrieved passages as source material; cite specific passages for claims; assess and narrate evidence quality (sufficient, partial, insufficient, conflicting); abstain when evidence is insufficient rather than fabricating; and stay within the uploaded documents.
- [ ] The prompt includes the retrieved passages in a form that makes their source clear (document name and location) so that the model can cite them accurately.
- [ ] The prompt does not instruct the model to introduce outside knowledge as if it were from the documents.
- [ ] The prompt construction is separated from the model call (story 4.1): the prompt is built by one function, sent to the model by another, so that the prompt can be inspected, tested, and changed independently.
- [ ] The prompt is documented enough that it can be revised based on evaluation results (e.g. from the evaluation dataset) without requiring a rewrite of the surrounding logic.

Initial prompt structure: the prompt is built in labeled sections — (1) role and task, (2) the user's question, (3) retrieved passages with document name, location, and text clearly labeled as source material, (4) document scope context, (5) conversation context if any, (6) answer instructions implementing the five requirements from Architecture §6, (7) evidence quality instructions with the four categories and the PRFAQ/architecture language, including the conservative-heuristics instruction to err toward "insufficient" when the boundary is unclear (PRFAQ §3 risk mitigation), and (8) output format instructions implementing the structured output schema from story 4.3. The prompt construction is separated from the model call (story 4.1): prompt building is one function, model invocation is another, so the prompt can be inspected, tested, and revised independently. The exact wording of each section is an implementation detail to be written and refined against the evaluation dataset; the section structure and required content are specified here as the initial default.

### Story 4.3: Structured output schema — answer, citations, evidence quality

**Goal:** Define and enforce the structured output that the generation layer produces, so that citations and evidence quality are machine-readable and the interactive citation mechanism (epic 6) and evaluation (epic 0) can consume them reliably.

**Acceptance criteria:**
- [ ] The output schema for a generated answer includes: answer text (string), a list of citations (each with passage ID, document name, location, and excerpt), evidence quality (one of: sufficient, partial, insufficient, conflicting), an evidence quality narrative (string), and an abstention flag (boolean or equivalent, indicating whether the answer is an abstention).
- [ ] The output is structured (not free text) to the extent the model interface supports it: the generation layer parses or constrains the model's response into the schema, and does not treat a free-text answer as the final output without extracting the structured components.
- [ ] Each citation's excerpt is the actual passage text from the content store (epic 2), not a paraphrase generated by the model. The generation layer retrieves the passage text by passage ID and uses it as the excerpt.
- [ ] The evidence quality narrative uses the categories and language defined in the PRFAQ and architecture (sufficient: "The documents support this directly"; partial: "The documents support part of this. [X] appears in [source]; the rest is a reasonable inference but not stated explicitly"; insufficient: "The documents do not contain enough information to answer this definitively"; conflicting: "The documents disagree. [Source A] says X; [Source B] says Y").
- [ ] The schema is versioned or documented so that changes to it are visible and intentional.

**Depends on:** Epic 2 (content store) for passage text retrieval by ID, used to populate citation excerpts.

### Story 4.4: Abstention logic — when to say "I don't have enough"

**Goal:** Implement the logic that determines when the generation layer should abstain rather than produce a definitive answer, and produce an appropriate abstention response.

**Acceptance criteria:**
- [ ] The generation layer produces an abstention response when the retrieved passages do not contain sufficient evidence to answer the question definitively.
- [ ] The abstention response identifies what is and is not supported by the retrieved passages, and includes the closest relevant passage if one exists (as specified in the architecture).
- [ ] The abstention response does not fabricate an answer to fill the gap. Claims not supported by the retrieved passages are not presented as facts.
- [ ] The abstention flag in the output schema (story 4.3) is set appropriately when the response is an abstention.
- [ ] The evidence quality narration for an abstention uses the "insufficient" language from the PRFAQ/architecture.
- [ ] Abstention behavior is tested with at least one question from the evaluation dataset that has an insufficient-evidence label, and the system's response is verified to be an abstention that does not fabricate.

**Depends on:** Story 4.3 (output schema) for the abstention flag and evidence quality narration fields.

### Story 4.5: Partial evidence and conflicting evidence handling

**Goal:** Implement the logic that distinguishes partial evidence from strong evidence, and that surfaces conflicting information from multiple sources.

**Acceptance criteria:**
- [ ] When the retrieved passages support part of a reasonable answer but not all of it, the generation layer produces a response that distinguishes the supported part from the unsupported part, using the partial-evidence language from the PRFAQ/architecture.
- [ ] The partial-evidence response does not present unsupported inferences as if they were stated in the documents. Inferences are clearly labeled as inferences.
- [ ] When the retrieved passages from different sources (or different parts of the same source) disagree on a point, the generation layer produces a response that surfaces the conflict, naming the sources and what each says, using the conflicting-evidence language from the PRFAQ/architecture.
- [ ] The conflicting-evidence response does not silently choose one side. Both (or all) sides are presented.
- [ ] The evidence quality narration for partial evidence uses the "partial" category; for conflicting evidence, the "conflicting" category.
- [ ] Partial-evidence and conflicting-evidence behavior is tested with questions from the evaluation dataset that have the corresponding labels, and the system's responses are verified to match the expected behavior.

**Depends on:** Story 4.3 (output schema) for the evidence quality fields; the evaluation dataset (epic 0) for test cases.

### Story 4.6: Citation generation — producing resolvable citations

**Goal:** Ensure that every citation in a generated answer is a resolvable reference to a passage in the content store, with the correct document name and location.

**Acceptance criteria:**
- [ ] Each citation in the output includes a passage ID that exists in the content store (epic 2) and resolves to the passage that supports the claim.
- [ ] Each citation includes the document name and location (page, section, or paragraph identifier) from the passage metadata, not a fabricated locator.
- [ ] Citations are produced for claims that are supported by retrieved passages. The generation layer does not cite a passage for a claim that the passage does not support.
- [ ] The citation's excerpt is the actual passage text (story 4.3), verified to match the content store's record for that passage.
- [ ] Citation correctness is measured against the evaluation dataset (epic 0): for each citation in a generated answer, the evaluation checks that the cited passage actually supports the claim. The generation layer's behavior is validated against this metric.
- [ ] The generation layer does not produce citations with passage IDs that do not exist in the content store. If a citation cannot be resolved, that is an error condition, not a normal output.

**Depends on:** Epic 2 (content store) for passage existence and text; story 4.3 (output schema) for the citation structure.

---

## Epic 5: Conversational Follow-Up

**Type:** Core capability  
**Priority:** Medium-high — the PRFAQ specifies conversational follow-up as an MVP feature  
**Depends on:** Epic 4 (generation layer) for the answer generation; epic 3 (retrieval) for each follow-up query; epic 2 (content store) indirectly.  
**Other epics depend on:** None directly — this epic uses the existing pipeline and adds conversation context handling.  

**Scope:** Allow the user to ask follow-up questions in context, with each follow-up still a retrieval + generation cycle grounded in the uploaded documents.

### Story 5.1: Conversation context storage and retrieval

**Goal:** Store and retrieve conversation history so that follow-up questions can be answered in context.

**Acceptance criteria:**
- [ ] A conversation session has a stable identifier.
- [ ] Each turn in a conversation (user query + system answer) is stored in the conversation history, associated with the session.
- [ ] The conversation history for a session can be retrieved given the session ID.
- [ ] The conversation history is stored in a form that includes the query text, the answer text, and the citations (so that the context is complete, not just the query text).
- [ ] Conversation history is stored locally and persists across turns (single-user, local-first — no external service required).
- [ ] Conversation history has a reasonable maximum length or pruning strategy, so that context does not grow unbounded. The maximum is documented.

### Story 5.2: Follow-up query with conversation context

**Goal:** Process a follow-up question using the conversation history as context, while keeping the answer grounded in the uploaded documents.

**Acceptance criteria:**
- [ ] A follow-up query is processed as a new retrieval + generation cycle (epics 3 and 4), with the conversation history included in the generation prompt as context.
- [ ] The conversation context includes the original query and answer (so the system can refer back to what was established), the document scope from the original query (unless the user changes it), and any clarifying information from the conversation.
- [ ] The follow-up answer stays grounded in the uploaded documents: the system does not introduce outside knowledge as if it were from the documents, even in a follow-up.
- [ ] The follow-up answer includes citations, evidence quality narration, and abstention behavior consistent with the initial answer (same standards, same output schema).
- [ ] If the user changes the document scope in a follow-up (e.g. from all documents to a specific document), the new scope is respected for that follow-up.
- [ ] Follow-up behavior is tested with at least one multi-turn conversation from the evaluation dataset (or a representative scenario), and the follow-up answers are verified to be grounded and consistent with the initial answer.

**Depends on:** Epic 3 (retrieval), epic 4 (generation), story 5.1 (conversation context).

Initial default conversation history limit: 5 turns (user query + system answer pairs), with oldest-first pruning when the limit is exceeded. This is the MVP default and is explicitly configurable. Five turns supports the PRFAQ's described follow-up use cases (refining scope, clarifying a passage, narrowing to a document) without unbounded context growth. The architecture's requirement — that the system use conversation history as query context, not as an independent knowledge source — is satisfied regardless of the turn limit. The limit should be adjusted if evaluation or user feedback shows it is too short or unnecessarily long.

---

## Epic 6: Interactive Citation UI

**Type:** User-facing capability  
**Priority:** Medium-high — interactive citation is the core trust mechanism; the PRFAQ specifies that citations must be inspectable  
**Depends on:** Epic 2 (content store) for passage text by ID; epic 4 (generation) for citation output (passage IDs, document names, locations, excerpts).  
**Other epics depend on:** None.  

**Scope:** Render citations as interactive elements that display the actual passage text from the original document. The specific UI mechanism (inline expansion, side panel, document highlight) is an open decision — this epic establishes the capability, not the final UI form.

### Story 6.1: Citation rendering — display passage excerpt on interaction

**Goal:** Implement the UI capability that, when a user interacts with a citation in an answer, displays the passage excerpt (the actual text from the original document) and its metadata (document name, location).

**Acceptance criteria:**
- [ ] Each citation in a displayed answer is rendered as an interactive element (the exact form — clickable text, expandable footnote, highlighted passage — is chosen and documented as part of this story, but is not prescribed by the architecture).
- [ ] Interacting with a citation (clicking, expanding, or whatever the chosen mechanism is) displays the passage excerpt: the actual text from the original document, exactly as stored in the content store (epic 2).
- [ ] The displayed passage information includes: the document name, the location (page, section, or paragraph identifier), and the passage text.
- [ ] The displayed passage text is the content store's record for that passage, not a regenerated or paraphrased version. This is verified by comparing the displayed text against the content store's passage text for a sample of citations.
- [ ] The citation interaction works for citations from all MVP document formats (PDF with page locator, Markdown with section locator, TXT with paragraph/offset locator).
- [ ] The citation interaction does not require a network call or external service — it reads from the local content store.

MVP citation UI mechanism: clickable reference in the answer text that expands inline to show the passage excerpt (document name, location, passage text) when activated by the user. This is the MVP default, chosen as the most direct implementation of the PRFAQ's "clicking or expanding" language (PRFAQ §2) and the lowest-infrastructure option for a local-first MVP. The PRFAQ lists side panel and document highlight as alternative forms (PRFAQ §11); these are potential improvements, not MVP requirements. The architecture's requirements — resolvability to a passage and passage text availability — are satisfied by this mechanism.

### Story 6.2: Citation UI for multi-document answers

**Goal:** Ensure the citation UI works when an answer draws on multiple documents, displaying the correct document name and location for each citation.

**Acceptance criteria:**
- [ ] When an answer cites passages from multiple documents, each citation displays the correct document name for its source.
- [ ] The user can distinguish which document each citation comes from (the document name is visible in or near the citation interaction).
- [ ] The citation interaction for a multi-document answer does not conflate passages from different documents (each citation resolves to its own passage only).
- [ ] The citation UI is tested with a multi-document answer and each citation is verified to display the correct document and passage.

**Depends on:** Story 6.1.

---

## Epic 7: Application Layer — FastAPI and End-to-End Flow

**Type:** Integration / delivery  
**Priority:** Medium — wraps the core capabilities in a usable application  
**Depends on:** Epics 1–6 for the underlying capabilities. The API surface can be defined early (contract first), but the full end-to-end flow requires the underlying capabilities.  
**Other epics depend on:** None — this is the outermost layer.  

**Scope:** FastAPI application with document upload, query, citation lookup, and conversation endpoints. Single-user, local-first. No authentication.

### Story 7.1: API contract definition

**Goal:** Define the API surface before implementation, so that the endpoints are consistent with the capabilities and the evaluation runner (epic 0, story 0.4) can target a stable interface.

**Acceptance criteria:**
- [ ] An API specification exists that defines the following endpoints at minimum:
  - Document upload: accepts a file (PDF, Markdown, or TXT), returns a document identifier and status.
  - Query: accepts a question/text and an optional document scope, returns an answer with citations, evidence quality, and abstention flag.
  - Citation lookup: accepts a passage ID, returns the passage text and metadata (for the interactive citation UI and for evaluation).
  - Conversation: accepts a follow-up query in a session context, returns a follow-up answer.
  - Document list: returns the list of uploaded documents with basic metadata.
  - Document removal: removes a document and its passages.
- [ ] The API specification defines the request and response shapes for each endpoint, including the structured answer response (story 4.3).
- [ ] The API specification is consistent with the output schema from the generation layer (epic 4) and the data model from the architecture.
- [ ] The API specification does not include authentication, authorization, multi-tenancy, or billing endpoints — those are out of MVP scope.
- [ ] The API specification does not include a confidence score field — numerical confidence scores are out of scope.

### Story 7.2: Document upload endpoint

**Goal:** Implement the document upload endpoint, wiring it to the ingestion pipeline (epic 1) and content store (epic 2).

**Acceptance criteria:**
- [ ] The upload endpoint accepts a file and identifies its format (PDF, Markdown, or TXT).
- [ ] The upload endpoint routes the file to the ingestion pipeline and persists the resulting passages via the content store.
- [ ] The endpoint returns a document identifier and a status indicating success or failure.
- [ ] On success, the document and its passages are retrievable via the content store and the document list endpoint.
- [ ] On failure (unrecognized format, corrupt file, extraction error), the endpoint returns a clear error, not a silent success or a partial result.
- [ ] Upload is tested with at least one file of each MVP format and the resulting document and passages are verified to be stored correctly.
- [ ] Upload does not require authentication (single-user, local-first).

**Depends on:** Epic 1 (ingestion), epic 2 (content store).

### Story 7.3: Query endpoint — end-to-end answer with citations

**Goal:** Implement the query endpoint that accepts a question and optional scope, runs retrieval and generation, and returns a structured answer.

**Acceptance criteria:**
- [ ] The query endpoint accepts a question (text) and an optional document scope (all documents or a specific document ID).
- [ ] The endpoint runs the retrieval pipeline (epic 3) with the given scope, then the generation layer (epic 4) with the retrieved passages.
- [ ] The endpoint returns a structured answer matching the output schema (story 4.3): answer text, citations with excerpts, evidence quality, evidence quality narrative, abstention flag.
- [ ] The endpoint's answer is grounded in the retrieved passages — the evaluation dataset (epic 0) can be used to verify faithfulness and citation correctness against this endpoint.
- [ ] The endpoint handles the case where retrieval returns no relevant passages: the generation layer produces an abstention or insufficient-evidence response, and the endpoint returns that response (not an error, not a fabricated answer).
- [ ] The endpoint records response latency for evaluation purposes.
- [ ] The query endpoint is tested end-to-end with at least one question that has a single-source answer in the evaluation dataset, and the returned answer, citations, and evidence quality are verified to be correct.

**Depends on:** Epic 3 (retrieval), epic 4 (generation), story 7.2 (document upload, so there are documents to query).

### Story 7.4: Citation lookup endpoint

**Goal:** Implement the endpoint that returns a passage's text and metadata given a passage ID, used by the interactive citation UI and by the evaluation runner.

**Acceptance criteria:**
- [ ] The citation lookup endpoint accepts a passage ID and returns the passage's text, document name, location, and any other metadata the content store holds for that passage.
- [ ] The returned passage text is identical to the content store's record for that passage.
- [ ] The endpoint returns a clear error for a passage ID that does not exist.
- [ ] The endpoint is tested by looking up passages that exist (from uploaded documents) and verifying the returned text and metadata are correct.

**Depends on:** Epic 2 (content store).

### Story 7.5: Document list and removal endpoints

**Goal:** Implement endpoints that let the user see uploaded documents and remove one.

**Acceptance criteria:**
- [ ] The document list endpoint returns the list of uploaded documents with basic metadata (name, format, upload timestamp, page/section count where available).
- [ ] The document removal endpoint accepts a document ID, removes the document and its passages from the content store, and returns a status.
- [ ] After removal, the document no longer appears in the document list and its passages are no longer retrievable via the citation lookup endpoint.
- [ ] Removal does not affect other documents.
- [ ] Both endpoints are tested: upload a document, list it, remove it, confirm it is gone and other documents remain.

**Depends on:** Epic 2 (content store).

### Story 7.6: Application wiring and startup

**Goal:** Wire all endpoints into a FastAPI application that starts, serves requests, and shuts down cleanly.

**Acceptance criteria:**
- [ ] The FastAPI application starts without error and serves the defined endpoints.
- [ ] The application is local-first: it runs without external services beyond what is required for the MVP (e.g. the LLM provider through Hermes, which is external but model-agnostic).
- [ ] The application shuts down cleanly.
- [ ] The application's endpoints are consistent with the API contract (story 7.1).
- [ ] The application is tested with a minimal end-to-end flow: upload a document, query it, verify the answer and citations, look up a citation, confirm the passage text is correct.

**Depends on:** Stories 7.1, 7.2, 7.3, 7.4, 7.5; and on epics 1–4 for the underlying capabilities.

---

## Dependencies Summary

```
Epic 0 (Evaluation Dataset) ──▶ runs in parallel, no technical deps
    │
    ├── informs validation of epics 1–7
    │
Epic 1 (Ingestion) ──▶ Epic 2 (Content Store) ──▶ Epic 3 (Retrieval) ──▶ Epic 4 (Generation) ──▶ Epic 7 (Application)
                                │                                                      │
                                └──────────────────────────────────────────────────────┘
                                (generation consumes passages via retrieval, not directly)

Epic 2 (Content Store) ──▶ Epic 6 (Citation UI)
Epic 4 (Generation) ──▶ Epic 5 (Conversational Follow-Up)
Epic 2 + Epic 4 ──▶ Epic 6 (Citation UI)
Epic 0 (Evaluation Dataset) ──▶ validates epics 1–7 at milestones
```

**Key dependency chains:**
- Epic 1 → Epic 2 → Epic 3 → Epic 4 → Epic 7 (main implementation sequence)
- Epic 2 + Epic 4 → Epic 6 (citation UI needs both content store and generation output)
- Epic 4 + Epic 3 → Epic 5 (follow-up needs generation and retrieval per turn)
- Epic 0 runs in parallel and validates all of the above; it should be active before implementation of epics 2–7 proceeds significantly, and the evaluation runner (story 0.4) should be available before end-to-end testing of epic 7.

---

## What Is Out of Scope (MVP)

Confirmed against the PRFAQ and architecture. These are not represented in any story above:

- **DOCX ingestion** — no format handler, no chunking story, no API support.
- **Scanned PDFs / OCR** — PDF ingestion story (1.1) is explicitly for text-based PDFs only.
- **Dedicated contradiction-detection feature or report** — conflict surfacing is handled in the generation layer (story 4.5) as part of normal answer generation, not as a separate feature.
- **Reranking** — retrieval stories (3.1–3.5) implement hybrid semantic + keyword retrieval without a separate reranking step. Reranking is noted as a possible later optimization, not an MVP story.
- **Authentication, billing, multi-tenancy, enterprise administration** — no stories, no API endpoints.
- **Numerical confidence scores** — not in the output schema (story 4.3), not in the API contract (story 7.1).

---

## Ambiguities and Open Decisions

All 10 items from the decision-resolution pass (documented in the decision-resolution artifact) have been resolved:

- **Items 1–6, 8–10 (9 items):** Resolved as RECOMMENDED MVP DEFAULTS and incorporated into the relevant stories (stories 1.4, 2.1, 3.1, 3.2, 3.3, 3.5, 4.2, 5.1, 6.1). These are explicitly provisional where the architecture anticipates later refinement against the evaluation dataset (chunking parameters, hybrid weighting, embedding model, candidate set size). They are consistent with the PRFAQ and Architecture and introduce no new scope.
- **Item 7 (evaluation dataset ownership and curation pace):** Resolved by explicit product decision. The product owner owns and maintains the evaluation dataset as a living project artifact; Hermes assists with curation scaffolding, dataset structure, validation scripts, and updates. This is recorded in the Epic 0 header and story 0.1.

No remaining ambiguities from the original 10 items. Any new ambiguities discovered during implementation should be flagged in this section.

---

## Summary

**Epics created:** 8 (Epic 0: Evaluation Dataset as enabling workstream; Epic 1: Document Ingestion; Epic 2: Content Store; Epic 3: Retrieval Pipeline; Epic 4: Generation Layer; Epic 5: Conversational Follow-Up; Epic 6: Interactive Citation UI; Epic 7: Application Layer).

**Stories:** 29 total across the 8 epics. Every story has acceptance criteria. The acceptance criteria in epics 3, 4, and 5 are particularly rigorous around retrieval precision/recall, citation correctness, faithfulness, hallucination prevention, abstention, partial vs. strong evidence, conflicting evidence, scope selection, and inspectable citations — as requested.

**Key dependencies:**
- Epic 0 runs in parallel and is the earliest-starting workstream. The evaluation runner (story 0.4) should be available before end-to-end testing of epic 7. The evaluation methodology (story 0.1) must be established before epics 2–4 proceed significantly.
- Main sequence: Epic 1 (Ingestion) → Epic 2 (Content Store) → Epic 3 (Retrieval) → Epic 4 (Generation) → Epic 7 (Application).
- Epic 6 (citation UI) depends on Epic 2 and Epic 4.
- Epic 5 (follow-up) depends on Epic 3 and Epic 4.
- Stories within epics 3 and 4 can proceed in parallel where their dependencies are met (e.g. stories 3.1, 3.2, 3.3 in parallel once epic 2 is available; stories 4.1, 4.2, 4.3 in parallel once epic 3 is available).

**No implementation has begun at this point.** This artifact is the approved planning baseline for the DocuResearch epics and stories.
