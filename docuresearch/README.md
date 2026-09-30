# DocuResearch

Ask questions about your technical documents (PDF, Markdown, TXT) and get answers grounded in the source, with citations to the exact page or section.

Every citation is checked against the passages retrieved for the question. If an answer has no valid citation, it is withheld rather than shown. When the documents don't contain the answer, DocuResearch says so.

## How it works

```
upload → extract text (page/section aware) → chunk → embed → SQLite
question → hybrid search (semantic + BM25) → prompt → LLM → parse
        → validate citations → withhold uncited claims → answer
```

The LLM layer works with any OpenAI-compatible endpoint (Nous Portal, OpenRouter, Ollama).

## Setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Configure the LLM with environment variables (see `.env.example`), or copy `config.example.toml` to `config.toml`:

```bash
export DOCURESEARCH_LLM_BASE_URL=https://inference-api.nousresearch.com/v1
export DOCURESEARCH_LLM_MODEL=<model name>
export DOCURESEARCH_LLM_API_KEY=<key>    # not needed for a local Ollama server
```

Without an LLM, the app still starts. Upload, listing, removal and citation lookup work, and the question endpoints return 503.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the app. Upload documents, ask questions, and click a numbered reference (for example **[1] sample_spec.pdf · Page 1**) to expand the exact stored passage behind it. Follow-up questions keep the conversation's context until you click **New conversation**.

The interactive API docs are at http://localhost:8000/docs.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/documents/upload` | Upload a PDF, Markdown or TXT file (optional `name` form field) |
| `GET /api/v1/documents` | List documents |
| `DELETE /api/v1/documents/{id}` | Remove a document and its passages |
| `POST /api/v1/query` | Ask a question; optional `document_scope` and `session_id` |
| `POST /api/v1/conversation` | Start a conversation session |
| `POST /api/v1/conversation/{id}/follow-up` | Ask a follow-up in context |
| `GET /api/v1/citations/{passage_id}` | The stored text behind a citation (URL-encode the ID) |

## Test and evaluate

```bash
pytest -q        # unit and integration tests; no network or model downloads
ruff check .
python -m evaluation.pipeline_system --label baseline   # needs an LLM configured
```

The evaluation dataset (`evaluation/dataset/`) has questions with gold answers and gold passages. It covers single-source, multi-passage, partial, insufficient and conflicting evidence. See `evaluation/methodology.md` for how each metric is scored.

## Scope

This is a single-user, local-first MVP. Not included: authentication, multi-tenancy, OCR for scanned PDFs, DOCX, and numerical confidence scores.
