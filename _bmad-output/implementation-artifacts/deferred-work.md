# Deferred Work

Items found during BMad build reviews that are real but belong outside the story that found them.

- source_spec: `_bmad-output/implementation-artifacts/spec-9-1-daily-question-limits.md`
  summary: The site question count may not match provider requests (retries and DNS failures), so the 40/day site limit may not reliably keep OpenRouter under 50/day.
  evidence: maybe-false, would be medium. `_post_with_retry` can send up to 4 requests per question, and DNS failures are counted though nothing reached OpenRouter. Settle by reading `X-RateLimit-Remaining` before and after a retried call. Belongs with Story 9.2 (provider daily-cap handling). Triage rows #8 and #26.
- source_spec: `_bmad-output/implementation-artifacts/spec-9-1-daily-question-limits.md`
  summary: The evaluation runner and LLM judge make model calls on the same key that are not counted by the daily limits; the eval command should warn that a run uses the shared allowance.
  evidence: `evaluation/pipeline_system.py` calls the pipeline without `before_model_call`, and a judged run needs about 72 requests. The change proposal §4.5 assigns the eval-command warning to Epic 9. Triage row #38.
- source_spec: `_bmad-output/implementation-artifacts/spec-9-1-daily-question-limits.md`
  summary: Requests from a user already at their limit still pay for retrieval before being refused; the app has no general request rate limiting.
  evidence: the `before_model_call` hook runs after `retrieve()` (query embedding, hybrid search). This exposure predates this story (no endpoint is rate-limited), so a request-rate limit or a cheap read-only pre-check before retrieval should be considered with Story 9.4 (abuse protection). Triage row #39.
- source_spec: `_bmad-output/implementation-artifacts/spec-9-2-openrouter-daily-cap-handling.md`
  summary: Report the provider allowance state in `GET /api/v1/health` (`llm_allowance_exhausted`, `llm_allowance_resets_at`); recommended home is Story 9.6 (deployment monitoring).
  evidence: Split from Story 9.2 at the token-count gate (spec about 2,300 tokens) at the product owner's choice. The story's core goal (detect the daily cap, stop retrying and calling, fail fast with 503) is complete without it. The data is available from `ResearchPipeline.llm_allowance`, which 9.2 adds.
