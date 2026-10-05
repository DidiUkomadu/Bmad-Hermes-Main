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
| `POST /api/v1/auth/register`, `/login`, `/logout`, `GET /api/v1/auth/me` | Accounts and sign-in |
| `POST /api/v1/documents/upload` | Upload a PDF, Markdown or TXT file (optional `name` form field) |
| `GET /api/v1/documents` | List documents |
| `DELETE /api/v1/documents/{id}` | Remove a document and its passages |
| `POST /api/v1/query` | Ask a question; optional `document_scope` and `session_id` |
| `POST /api/v1/conversation` | Start a conversation session |
| `POST /api/v1/conversation/{id}/follow-up` | Ask a follow-up in context |
| `GET /api/v1/citations/{passage_id}` | The stored text behind a citation (URL-encode the ID) |

## Accounts

Each person signs in with an email and password. Documents and conversations are private: a user only ever sees, searches, and gets answers from their own documents.

- **The first account** registered takes ownership of any documents and conversations created before accounts existed.
- **Registration** is open by default. Set `allow_registration = false` under `[auth]` in `config.toml` (or `DOCURESEARCH_ALLOW_REGISTRATION=false`) to close it.
- **Forgotten password:** `python -m app.auth.cli reset-password <email>` sets a new one and signs that user out everywhere. `python -m app.auth.cli list-users` lists accounts.
- **Serving over HTTPS:** set `cookie_secure = true` so the session cookie is only sent over HTTPS.

Passwords are stored as salted scrypt hashes, and sessions are random tokens of which only a hash is stored. The session cookie is HttpOnly and SameSite=Lax. Repeated failed sign-ins are throttled. Design and decisions: `_bmad-output/planning-artifacts/docuresearch-change-proposal-multi-user.md`.

## Deploy

To let other people use it, run it with Docker behind HTTPS:

```bash
DOMAIN=docs.example.com docker compose up -d --build
```

See [docs/deployment.md](docs/deployment.md) for requirements, configuration, backups, upgrades and limits.

## Test and evaluate

```bash
pytest -q        # unit and integration tests; no network or model downloads
ruff check .
python -m evaluation.pipeline_system --label baseline   # needs an LLM configured
python -m evaluation.pipeline_system --questions q-101,q-102 --no-judge   # subset, no judge
```

The evaluation dataset (`evaluation/dataset/`) has questions with gold answers and gold passages. It covers single-source, multi-passage, partial, insufficient and conflicting evidence. See `evaluation/methodology.md` for how each metric is scored. Each answer is checked claim by claim by an LLM judge, which reports faithfulness, the hallucination rate and every unsupported claim. A full judged run uses two model requests per question, so mind free-tier daily limits.

## Scope

A local-first MVP with private per-user libraries. Not included: shared workspaces, roles, single sign-on, email-based password reset, OCR for scanned PDFs, DOCX, and numerical confidence scores.
