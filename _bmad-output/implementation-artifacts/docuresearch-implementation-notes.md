# DocuResearch — Implementation Notes and Deviations

**Status:** MVP complete (Epics 0–8)
**Last updated:** 2026-10-05
**Companion to:** `planning-artifacts/docuresearch-implementation-plan.md` (the plan) and `sprint-status.yaml` (story status)

This file records where the built product differs from the approved plan and why, and the decisions the plan left to implementation (plan §14). The planning artifacts themselves are unchanged.

## How it was built

- **Epics 0–4** were implemented by Hermes Agent (Nous Portal, `upstage/solar-pro4:free`), story by story from the BMad artifacts, with each report reviewed against the acceptance criteria.
- **From Epic 5 onwards**, work continued with another AI coding assistant after the Hermes Nous session was revoked and further use required payment. The same artifacts and acceptance criteria were followed.
- **Integration** came before Epic 5. The components from Epics 1–4 were unit-tested but not connected, so the end-to-end pipeline (`app/pipeline.py`) was built first, to measure grounding against a real model before adding features.

## Deviations from the plan

| # | Plan | Built | Why |
|---|---|---|---|
| 1 | Free Nous model through Hermes for the app's LLM (AGENTS.md) | Any OpenAI-compatible endpoint. Currently OpenRouter free tier, `nvidia/nemotron-3-super-120b-a12b:free` | Nous Portal API access requires payment. The model-agnostic interface made this a configuration change. |
| 2 | Document IDs are UUID4 (§4.1, §5) | Deterministic `<slug>-<hash>` from the filename; owner + filename for user uploads | Gold passage IDs in the evaluation dataset must survive re-ingestion. Owner scoping lets two users upload the same filename. |
| 3 | Single-user MVP; no authentication | Multi-user, with private libraries and email/password sign-in (Epic 8) | Product owner decision, 2026-10-02. See `planning-artifacts/docuresearch-change-proposal-multi-user.md`. |
| 4 | Conversation history pruned to 5 turns (story 5.1) | All turns are stored; the newest 5 are given to the model as context | Users need to reopen past conversations in full. The context bound still satisfies story 5.1. |
| 5 | Follow-ups are a fresh retrieval + generation cycle (story 5.2) | Same, plus the follow-up search is expanded with the previous question and answer, and passages cited in recent answers are carried forward (max 3) | Elliptical follow-ups ("what about it?") otherwise lose the evidence the earlier answer relied on. Citations must still resolve to the current turn's passages. |
| 6 | Hybrid weighting 0.5 / 0.5 (story 3.3, provisional) | 0.7 semantic / 0.3 keyword, with BM25 stopwords | Tuned on the evaluation dataset: recall@20 rose from 0.884 to 0.930 and MRR from 0.585 to 0.620. The plan marked the weights as provisional. |
| 7 | Prompt shows up to 10 passages (story 4.2 default) | The prompt shows every retrieved candidate (20) | Gold passages ranked 11–20 counted as "retrieved" but were never shown, producing false "not in the documents" answers. |
| 8 | No answer-level guard beyond citation validation (story 4.6) | Answers that claim evidence but have no valid citation are withheld and become an abstention | AGENTS.md: grounding takes priority over fluency. |
| 9 | Location strings `page:N` (§4.3) | Stored as `page:N:chunk:M`; shown to users as "Page N" (`location_label`) | Passage IDs are built from the location; changing it would change every ID. |
| 10 | Plain text paragraphs as stored | Running page headers/footers are dropped, and numbered headings are joined to their paragraph | On RFC 7519, 59 of 337 passages were headers/footers that outranked real content. |
| 11 | Deployment "later" | Dockerfile, docker compose with Caddy (automatic HTTPS), and `docs/deployment.md` | Needed once multi-user existed. Tested: container end to end, and HTTPS with secure cookies. |

## Plan §14 decisions, as resolved

| Item | Resolution |
|---|---|
| 14.1 PDF library | `pypdf` (text-based PDFs only; no OCR) |
| 14.2 Markdown parser | `mistune` 3 (AST), with heading paths as section locators |
| 14.3 Embedding index | Brute-force cosine similarity over embeddings stored in SQLite, rebuilt per query. Adequate to about 1,000 pages; see "Known limits". |
| 14.4 BM25 | `rank_bm25` (BM25L), with stopwords |
| 14.5 Ingestion | Synchronous: upload returns 200 "available" |
| 14.6 Structured output | JSON in the prompt; tolerant extraction; unparsable output is a 502 "Generation failed", never a fabricated answer |
| 14.7 Conversation storage | SQLite tables `conversations` and `conversation_turns`, with citations as JSON |
| 14.8 UI | Plain HTML + JS served by FastAPI; inline-expanding citation references that show the stored passage |

## Evaluation status

- **Dataset:** 36 questions over 5 documents (3 synthetic samples, release notes, and RFC 7519): 23 single-source, 5 multi-passage, 3 partial, 4 insufficient, 1 conflicting.
- **Faithfulness** is measured by an LLM judge (`evaluation/judge.py`, methodology §2.4). It checks every claim against the retrieved passages and lists each unsupported claim. The judge is currently the same model as the generator, so treat it as lenient.
- **Results** are in `evaluation/dataset/results/`. Same 18 questions (14 on RFC 7519, plus q-001, q-019, q-020 and q-022), before and after the fixes:

  | Metric | Before (2026-10-02) | After (2026-10-05) |
  |---|---|---|
  | Faithfulness (judged answers) | 76.5% (13/17) | 15/15 judged faithful* |
  | Abstention accuracy | 72.2% (13/18) | 100% (18/18) |
  | Citation correctness (gold) | 0.875 | 0.924 |
  | Conflict handling | 1/1 | 1/1 |

  \* After the judge fix for "the documents do not say X" statements. q-113 (a correct "the specification recommends no libraries") could not be re-judged because of a network outage, and q-022 and q-110 had judge errors (malformed JSON, timeout).
- **No fabricated facts** appeared in either run. Every "before" failure was a false "not in the documents", caused by deviations 7 and 10 (now fixed).
- **Remaining weakness:** retrieval of a definition split across a page break. The second half of `exp` does not mention "exp", so q-103 and q-111 still find only one of their two gold passages.

## Known limits

- **Single process.** SQLite and an in-memory sign-in throttle; suited to a team, not large scale.
- **Search scaling.** The index is rebuilt per question: about 0.6 s at 1,000 passages and 7 s at 10,000. Upload embeds at about 0.5 s per page on CPU.
- **Free LLM tier:** 50 requests per day, shared by all users. A full judged evaluation needs about 72 requests.
- **Not supported:** scanned PDFs (no OCR) and DOCX.
- **Same-model judge:** a stronger or different judge model would give more trustworthy faithfulness scores.
