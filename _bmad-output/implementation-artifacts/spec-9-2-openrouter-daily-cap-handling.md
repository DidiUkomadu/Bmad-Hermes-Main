---
title: "Story 9.2: Handling OpenRouter's daily cap"
type: 'feature'
created: '2026-10-08'
status: 'done'
route: 'dispatch'
baseline_commit: '2e6e79116d8a36d43d30801d106ab510a6831e33'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-9-context.md'
  - '{project-root}/AGENTS.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-9-1-daily-question-limits.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** When OpenRouter's free daily allowance (50 requests) is used up, the connector still retries each question up to 3 more times, and every later question keeps calling OpenRouter. That wastes requests and returns a generic 502 instead of telling visitors the demo is out of AI questions until the reset.

**Approach:** Recognise the daily-cap 429, never retry it, and remember in memory that the allowance is exhausted until OpenRouter's reported reset. While exhausted, questions fail fast with 503 "The demo has used today's free AI allowance" and the reset time, with no OpenRouter call and no counting.

## Boundaries & Constraints

**Always:**
- **Daily signal:** a 429 (HTTP status, or the error code embedded in a 200 body) whose text contains `free-models-per-day`, or which has `X-RateLimit-Remaining: 0` together with an `X-RateLimit-Reset`. Read these headers from the HTTP headers or from the body's `error.metadata.headers`.
- Daily-cap 429s are never retried. Other 429s, 5xx responses and network errors keep today's retries.
- **Exhausted until:** `X-RateLimit-Reset` (epoch milliseconds when above 10^11, otherwise seconds). If it is missing, unparsable, in the past, or more than 48 hours ahead, and the text was `free-models-per-day`, use the next 00:00 UTC.
- The state is held in the connector instance only, so a restart clears it. Calls resume automatically after the reset time.
- While exhausted, all three question paths return 503 before retrieval and before 9.1 counting.
- The question that first hits the cap was already counted: it gets the 503 and 9.1's failed-call rule (user refund, capped at 2 per day; site count kept).
- The new error is a `GenerationError` subclass, so the evaluation runner records it as a failed generation.

**Never:** retry, call or count while exhausted; persist the state; change 9.1's limits or refunds; health-check changes (deferred to 9.6); any ungrounded fallback.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Busy 429 | per-minute 429, no daily signal, then 200 | Retried as today; answer returned | N/A |
| Daily cap (body) | 429, body `free-models-per-day`, metadata reset in ms | Exactly 1 HTTP call; exhausted until that reset | 503 with the reset time; refund per 9.1 |
| Daily cap (headers) | 429 with `X-RateLimit-Remaining: 0` and `X-RateLimit-Reset` headers | Same | 503 |
| Daily cap inside 200 | 200 `{"error": {"code": 429, ...free-models-per-day}}` | Same | 503 |
| Daily, no reset | `free-models-per-day`, no reset | Exhausted until next 00:00 UTC | 503 |
| Remaining 0, no reset, no daily text | 429 | Busy: retried | As today |
| While exhausted | any question path | 0 HTTP calls, no retrieval, not counted | 503 |
| Reset passes | clock past the reset time | Calls resume | N/A |
| Restart | new connector instance | Not exhausted | N/A |
| Eval runner | exhausted | A generation failure | No crash |

</frozen-after-approval>

## Code Map

- `app/generation/openai_compat.py` -- `_post_with_retry()` (~L125) retries every 429/5xx. Add classification, reusing `_error_body`, `_effective_status` and `ProviderError`. `generate()` (~L179) turns `httpx.HTTPError` into `LLMResponse(parse_error=...)`; keep that contract and report exhaustion through it. Fail fast in `generate()` while exhausted.
- `app/generation/interface.py` -- `LLMResponse`: add `provider_exhausted_until: datetime | None = None`.
- `app/generation/allowance.py` (new) -- `ProviderAllowance`: thread-safe `mark_exhausted(until)` and `exhausted_until(now=None)`, which clears itself once passed. Held as `OpenAICompatibleLLM.allowance`.
- `app/pipeline.py` -- `GenerationError` (L68); LLM call in `generate()` (~L300). Add `ProviderAllowanceExhausted(GenerationError)` with `resets_at`, raised when `response.provider_exhausted_until` is set. Add a `llm_allowance` property: `getattr(self._llm, "allowance", None)`, so fake LLMs give None.
- `app/api/query.py` -- `_in_session()`, `query()`, `follow_up()`: pre-check `state.pipeline.llm_allowance`, giving 503. Catch `ProviderAllowanceExhausted` before `GenerationError`, refunding through the existing `_generation_failed` path, giving 503. Use the `{"error", "detail"}` shape.
- `evaluation/pipeline_system.py` -- catches `GenerationError`; no change is needed. Verify by test.
- Tests: reuse `_llm_with(handler)` with `httpx.MockTransport` (`tests/integration/test_pipeline.py`), and `server` / `_seed` (`tests/integration/test_limits_api.py`).

## Tasks & Acceptance

**Execution:**
- [x] `app/generation/allowance.py`, `app/generation/interface.py` -- the tracker; the `LLMResponse` field.
- [x] `app/generation/openai_compat.py` -- daily-cap classification (all four signal forms), no retry, reset parsing and fallbacks, mark and report exhaustion, fail fast while exhausted, and one warning log when marking (the reset time; no secrets).
- [x] `app/pipeline.py` -- `ProviderAllowanceExhausted`; raise it; the `llm_allowance` property.
- [x] `app/api/query.py` -- the 503 pre-check on all three paths; the 503 mapping with refund; docstrings.
- [x] `tests/unit/test_allowance.py` -- marking, expiry, thread-safety, reset parsing (ms, s, missing, past, more than 48 hours ahead).
- [x] `tests/integration/test_daily_cap.py` -- every I/O matrix row: connector-level with counted HTTP calls; API-level 503 on all three paths with no LLM call and unchanged usage counts; refund of the discovering request; reset expiry with a patched clock; the evaluation runner.

**Acceptance Criteria:**
- Given OpenRouter returns its daily-cap 429, when a visitor asks, then exactly one HTTP request is made for that question, and later questions until the reset make none.
- Given 9.1 limits are on and the allowance is exhausted, when visitors ask, then usage counts do not increase.
- Given the existing suite, when done, then all tests pass unchanged.

## Implementation Notes

- Connector raises an internal `DailyCapError(ProviderError)` (Ruff N818 needs the `Error` suffix) from `_post_with_retry`; `generate()` marks the allowance, logs one warning and returns `LLMResponse(provider_exhausted_until=...)`.
- Reset parsing and the midnight fallback live in `app/generation/allowance.py` (`parse_reset`, `next_utc_midnight`); its `_now()` is the clock patched in tests.
- `remaining: 0` with a valid near reset (e.g. a per-minute window) marks exhaustion only until that reset, as the signal rule specifies.
- The 503 pre-check (`require_allowance`) runs right after `require_llm`; follow-up still returns 404 for an unknown session first.

## Spec Change Log

## Review Triage Log

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | blind, edge | A per-minute 429 with `Remaining: 0` and a reset seconds away is treated as the daily cap: no retry, and the 503 says "today's allowance" | low | Real, and the frozen signal rule allows it (Implementation Notes accept it). Retries inside a per-minute window would mostly fail anyway, and the block lasts only until that reset. With the site capped at 40 questions a day, 20 a minute is unlikely, and the fix adds a threshold guard. | reject |
| 2 | blind | `docs/deployment.md:91` still says questions fail with "Generation failed" when the provider allowance runs out | low | Confirmed at line 91. Operators are told the wrong behaviour. A direct doc fix. | patch |
| 3 | blind | The 503 has no `Retry-After` header | low | The only client is the bundled UI, which shows `detail`. Adding the header adds public surface. | reject |
| 4 | blind | The allowance state is not shown in `/health` or `/usage` | false | Intent **Never**: health-check changes are deferred to 9.6 (already in deferred-work). | reject |
| 5 | blind | The eval runner keeps running and scores failures after the cap | false | Intent matrix row "Eval runner: a generation failure, no crash" chooses this. Related to deferred #38. | reject |
| 6 | blind | The reset-time text is formatted in two places; the exception message is not used by the API | low | `pipeline.py:84` and `query.py:121`. The eval runner shows the exception text and the API its own detail; they differ in wording only, and both are correct. Cosmetic. | reject |
| 7 | blind | Private `_now` is used across modules, plus a duplicate module import | false | The module alias is needed so tests that monkeypatch `allowance._now` affect the connector; a from-import would not be patched. No bad outcome. | reject |
| 8 | blind | Unneeded `getattr` in `require_allowance` | false | It is defensive only; no wrong outcome occurs. | reject |
| 9 | blind, edge | `require_allowance` runs before `require_document`, so a missing or another user's document gets 503 instead of 404 while exhausted | low | Confirmed at `query.py:187-188` and `233-235`. Reordering is a direct correction, and it still runs before retrieval and counting. | patch |
| 10 | blind | Unlimited lock: 48 hours accepted and no manual clear | false | The 48-hour bound is frozen intent, and a restart clears the state (also frozen intent). | reject |
| 11 | blind | No test for a busy 429 followed by a daily-cap 429 inside one retry loop | low | Missing verification of the no-retry rule mid-loop. Adding the test is trivial. | patch |
| 12 | blind, gap | No test that HTTP headers win over the body's `error.metadata.headers` | low | Gap pre-verified: no test sends both sources, so swapping the merge order would pass. The filed disposition was defer, but the gap was caused by this story and the test is trivial. | patch |
| 13 | blind, gap | The midnight-fallback tests read `datetime.now` separately from the connector clock, so they are flaky across 00:00 UTC | low | Confirmed at `test_daily_cap.py:145-161`. The fix is to patch `allowance._now` to a fixed time. | patch |
| 14 | edge | A concurrent request that passes the pre-check after the cap was discovered is counted on the site | low | The window is between another request marking the cap and this request's LLM call. The user is refunded; the site count rises by at most the number of concurrent requests, and the provider is out until reset anyway. Unlikely, and the fix adds a guard. | reject |
| 15 | edge | Questions with no passages (no LLM call needed) are refused with 503 | false | The intent requires the 503 on all question paths before retrieval. | reject |
| 16 | edge | `remaining` and `reset` are merged per key across HTTP and body sources | maybe-false | It would need OpenRouter to send `Remaining: 0` in one source and an unrelated reset in the other. Even if true it would be low. | reject |
| 17 | edge | Claim: under concurrency, the site count rises while exhausted | low | Same root cause and evidence as #14. | reject |

## Design Notes

**Pre-check plus typed error:** the pre-check stops retrieval and counting once the state is known. `ProviderAllowanceExhausted` (a `GenerationError`) covers the discovering request and any concurrent race, without breaking callers that only know `GenerationError`.

**Remaining 0 is not always daily:** a per-minute 429 can also report `remaining: 0`. Requiring a reset time (or the `free-models-per-day` text) avoids blocking until midnight on a per-minute limit. Observed body:

```json
{"error": {"message": "Rate limit exceeded: free-models-per-day. ...", "code": 429,
  "metadata": {"headers": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1790985600000"}}}}
```

**Deferred #8 (site count versus provider requests):** once this story is in, the app stops calling OpenRouter as soon as the cap is reported, even if the site count undercounts. The remaining risk is only that the demo runs out before 40.

## Verification

**Commands:**
- `cd docuresearch && ../.venv/bin/ruff check .` -- expected: no errors
- `cd docuresearch && ../.venv/bin/python -m pytest -q -p no:warnings` -- expected: all pass (612 existing + new)
