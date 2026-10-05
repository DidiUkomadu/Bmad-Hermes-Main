# DocuResearch — AI Research Assistant for Technical Documents

## Project Overview
- **Codename:** DocuResearch
- **What it is:** AI assistant for uploading technical documents (PDFs, research papers, manuals, specifications, engineering docs) and asking questions grounded in the source material with citations to relevant sections/pages.
- **Output language:** English
- **Purpose:** Portfolio/learning product with potential to evolve into a real product.

## Tech Stack
- **Language:** Python
- **Framework:** FastAPI
- **LLM layer:** Model-agnostic architecture (any OpenAI-compatible endpoint, configured in `docuresearch/.env`).
  - **App LLM (current):** OpenRouter free tier, `nvidia/nemotron-3-super-120b-a12b:free`. Switched from Nous on 2026-09-30 because Nous Portal API access requires payment.
  - **Coding agent:** Hermes Agent (Nous) was used for Epics 0–4; later work continued without it.
- **Linting/formatting:** Ruff

## Scope (MVP)
- **Multi-user with private libraries** (changed 2026-10-02, see `_bmad-output/planning-artifacts/docuresearch-change-proposal-multi-user.md`): users sign in with email and password; each user's documents and conversations are visible only to them.
- **Excluded from MVP:** Billing, enterprise administration, roles and permissions, shared workspaces, OAuth / single sign-on, email-based password reset

## Deployment
- **Local-first**, with a production path: Docker + docker compose with Caddy (automatic HTTPS). See `docuresearch/docs/deployment.md`.

## Status
- **MVP complete (Epics 0–8).** Story status: `_bmad-output/implementation-artifacts/sprint-status.yaml`. Deviations from the plan, and why: `_bmad-output/implementation-artifacts/docuresearch-implementation-notes.md`.

## BMad Output Structure
- `_bmad-output/planning-artifacts/` — PRD, architecture, epics, stories
- `_bmad-output/implementation-artifacts/` — sprint status, implementation specs
- `docs/` — project knowledge

## Critical Pitfalls (Agents Must Know)
- **Hallucinations / ungrounded answers are a MAJOR concern.** Every answer must be grounded in source material. Citation accuracy and source-grounding take priority over fluency. Agents should flag uncertainty rather than fabricate.
- **Data isolation between users is mandatory.** Every API path must pass the signed-in user's ID as `owner_id` to the store, retrieval, citation and conversation functions. `owner_id=None` means "unfiltered" and is only for trusted internal callers (the evaluation runner). A user must never see, retrieve, or be answered from another user's documents. `tests/integration/test_multi_user.py` guards this.
