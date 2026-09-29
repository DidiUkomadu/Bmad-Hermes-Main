# DocuResearch — AI Research Assistant for Technical Documents

## Project Overview
- **Codename:** DocuResearch
- **What it is:** AI assistant for uploading technical documents (PDFs, research papers, manuals, specifications, engineering docs) and asking questions grounded in the source material with citations to relevant sections/pages.
- **Output language:** English
- **Purpose:** Portfolio/learning product with potential to evolve into a real product.

## Tech Stack
- **Language:** Python
- **Framework:** FastAPI
- **LLM layer:** Model-agnostic architecture. Start with free Nous model through Hermes for the agent layer.
- **Linting/formatting:** Ruff

## Scope (MVP)
- **Single-user** focus on core research workflow
- **Excluded from MVP:** Authentication, billing, multi-tenancy, enterprise administration

## Deployment
- **Local-first** for MVP; deployment comes later

## BMad Output Structure
- `_bmad-output/planning-artifacts/` — PRD, architecture, epics, stories
- `_bmad-output/implementation-artifacts/` — sprint status, implementation specs
- `docs/` — project knowledge

## Critical Pitfalls (Agents Must Know)
- **Hallucinations / ungrounded answers are a MAJOR concern.** Every answer must be grounded in source material. Citation accuracy and source-grounding take priority over fluency. Agents should flag uncertainty rather than fabricate.
