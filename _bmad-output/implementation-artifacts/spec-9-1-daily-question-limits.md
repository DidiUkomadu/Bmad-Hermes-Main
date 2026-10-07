---
title: 'Story 9.1: Daily question limits'
type: 'feature'
created: '2026-10-07'
status: 'done'
route: 'dispatch'
baseline_commit: '95b6116c368cd621aa6108381d70f485155b85a8'
review_loop_iteration: 1
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-9-context.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The public demo shares one OpenRouter free key with a 50 requests/day account cap, so a single visitor can exhaust the day's model allowance for everyone.

**Approach:** Count every question that is actually sent to the model, per user and site-wide, per UTC day. Refuse further questions with a clear 429 once a configured limit is reached, and expose the counts through `GET /api/v1/usage`. Limits are off unless configured.

## Boundaries & Constraints

**Always:**
- Count at the moment a question is sent to the model, inside `ResearchPipeline.generate`, after the no-passages early return. All paths (standalone `/query`, `/query` with `session_id`, `/conversation/{id}/follow-up`) go through it, so none can bypass the limit.
- The check and the increment are one atomic step: concurrent requests can never push a count past its limit.
- Days are UTC dates, resetting at 00:00 UTC.
- Per-user counts use the signed-in user's ID (Epic 8 isolation). `/usage` shows only the caller's own count plus site totals.
- A limit of 0 or unset means unlimited. Local and VPS behaviour is unchanged by default.
- Errors use the existing `{"error", "detail"}` shape.

**Never:**
- Count questions that make no model call (no passages retrieved).
- Implement 9.2–9.6 here: no provider-cap detection, upload or sign-up limits, banner or UI counter.
- Fall back to an ungrounded answer when a limit is hit.

**Decisions (product owner, 2026-10-07):**
- **Failed AI calls (revised after review, replacing the earlier "always refund"):** if the model call is sent but fails (502 "Generation failed"), the user's count is refunded, but at most **2 refunds per user per UTC day**; after that, failures count. The site count is always kept. Reason: unlimited refunds let one user drain the site budget by provoking failures, for example with a prompt-injecting document.
- **No owner exemption:** everyone, including the site owner, has the same per-user limit.
- **Spec size:** keep as one spec (about 1,980 tokens; cohesive single goal).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Under limits | user 2/5, site 10/40 | Answer returned; counts become 3 and 11 | N/A |
| User limit hit | user 5/5 | No model call; nothing counted | 429 "Daily question limit reached", detail names *your* limit and the reset time |
| Site limit hit | site 40/40, user 1/5 | No model call | 429, detail names the *site* limit and the reset time |
| No passages | empty library | Abstention returned, counts unchanged | N/A |
| Limits off | limits 0/unset | Behaves exactly as today; `/usage` limits are null | N/A |
| Day rollover | counts from yesterday | Today starts at 0 | N/A |
| AI call fails | user 2/5, 0 refunds today, provider error | 502 "Generation failed"; user back to 2, site stays +1, refunds today = 1 | Refund applied even when the error propagates |
| Refund cap reached | user 2/5, 2 refunds today, provider error | 502; user becomes 3 (not refunded), site +1 | Third and later failures that day count |
| Refund across midnight | reserved 23:59:59 UTC, fails after 00:00 | Refund applies to the reservation's day | N/A |
| Concurrent last slot | two requests at user 4/5 | Exactly one answered | The other gets 429 |
| Bypass attempt | follow-up or session query at 5/5 | Refused like `/query` | 429 |

</frozen-after-approval>

## Code Map

- `app/pipeline.py` -- `ResearchPipeline.generate()` (~L259): no-passages early return at L266, LLM call at L279. Add an optional `before_model_call` hook, invoked after the early return and before `self._llm.generate`. Thread it through `answer()` (~L160). Do not change retrieval or post-processing.
- `app/conversation/followup.py` -- `ask()` (L66) calls `pipeline.generate(context)` at L116; accept and pass the hook.
- `app/api/query.py` -- `_in_session()` (L85, calls `ask` at L96), `query()` (L112, calls `pipeline.answer` at L135), `follow_up()` (L226). Build the hook from the request's user and settings, and map the limit exception to 429. Add `GET /usage` here.
- `app/api/schemas.py` -- add a `UsageResponse` model.
- `app/api/errors.py` -- reuse `APIError`.
- `app/store/schema.py` -- `create_schema()`: add `usage_counters (day TEXT, scope TEXT, user_id TEXT, count INTEGER, PRIMARY KEY(day, scope, user_id))` with `CREATE TABLE IF NOT EXISTS`, the same pattern as the Epic 8 additions (~L152).
- `app/config.py` -- `Settings`: add `questions_per_user_per_day` and `questions_per_site_per_day` (default 0), read from a TOML `[limits]` section plus env `DOCURESEARCH_QUESTIONS_PER_USER_PER_DAY` / `_PER_SITE_PER_DAY`. Reject negatives with `ConfigError`. Follow the existing `_flag` / `_clean` style.
- `config.example.toml` -- add a documented `[limits]` section (off by default).
- `tests/integration/fakes.py` -- reuse `sign_up`, `CiteLLM`, `ScriptedLLM`, `HashEmbedding`. The `make_client` fixture pattern is in `tests/integration/test_api.py`.

## Tasks & Acceptance

**Execution:**
- [x] `app/limits/__init__.py`, `app/limits/usage.py` -- new. `DailyLimits` config holder. `reserve_question(conn, user_id, limits, now)` atomically checks and increments both counters in one `BEGIN IMMEDIATE` transaction, and raises `QuestionLimitReached(scope, limit, resets_at)`. `refund_user_question(conn, user_id, day)` refunds the user's count (never below 0) only while that user has had fewer than `MAX_REFUNDS_PER_DAY = 2` refunds on that day; it records the refund (for example a `scope='refund'` row) and returns whether it refunded. `usage_summary(conn, user_id, limits, now)`; `next_reset(now)`. Messages: "limit of 1 question" / "limit of N questions"; a site refusal must say the site's shared daily limit is reached, without implying the user caused it. Rationale: one place owns counting.
- [x] `app/store/schema.py` -- create the `usage_counters` table.
- [x] `app/config.py`, `config.example.toml` -- the two limit settings. Reject booleans and non-integer numbers (TOML `true`, `5.9`) as well as negatives, with `ConfigError`.
- [x] `app/pipeline.py`, `app/conversation/followup.py` -- the `before_model_call` hook (default None, so existing callers are unchanged).
- [x] `app/api/query.py`, `app/api/schemas.py` -- wire the hook into all three question paths; return 429 with the scope and reset time in the detail; on `GenerationError` after a reservation, call `refund_user_question` for the reservation's day; add `GET /api/v1/usage` (auth required). Document in `UsageCountModel` that `used` is 0 while limits are off, because nothing is counted then.
- [x] `README.md` -- add `GET /api/v1/usage` to the endpoint table.
- [x] `tests/unit/test_limits.py` -- counting, rollover, scopes, limits off, atomic concurrency, config rejection (negative, bool, float), singular/plural and site wording, refund cap (third refund refused), refund across midnight.
- [x] `tests/integration/test_limits_api.py` -- every row of the I/O matrix through the API, including the bypass attempts and the usage endpoint's per-user isolation.
- [x] `tests/integration/test_api.py`, `tests/integration/test_multi_user.py` -- add `/api/v1/usage` to the OpenAPI path set and to the auth-required check. (This is the only change to existing tests: any new endpoint requires it.)

**Acceptance Criteria:**
- Given limits are configured, when a user asks via any of the three question paths, then the user and site counts each increase by exactly 1, and `GET /api/v1/usage` reflects it.
- Given two users, when user A hits their limit, then user B can still ask (until the site limit).
- Given the existing test suite, when this story is complete, then all existing tests still pass, because limits are off by default. The only edits are the endpoint lists named in the tasks.

## Implementation Notes

- **2026-10-07, implementation (dispatch subagent) and verification (orchestrator):** all tasks done. Ruff is clean, and the full suite passes: 596 passed, 1 skipped (565 existing + 31 new: 18 unit, 13 integration). Every I/O matrix row is covered by a passing test.
- **AC deviation, accepted:** "existing tests pass unchanged" could not hold for `tests/integration/test_api.py::test_openapi_matches_contract_and_has_no_confidence`, which asserts the exact set of API paths. `/api/v1/usage` was added to that set and nothing else changed. Any new endpoint requires this.
- **Counter connection:** `_QuestionCounter` (in `app/api/query.py`) opens its own short-lived connection for `reserve_question` and `refund_user_question`, so `BEGIN IMMEDIATE` never collides with a transaction already open on the request's connection.
- **Limits off means not counted:** when both limits are 0 the hook returns immediately, so `/usage` reports `used: 0` with null limits. Counting starts from zero when limits are turned on. This keeps behaviour identical to before (spec: "behaves exactly as today").
- **Refund scope:** only `GenerationError` (provider, network or parse failures, as raised by the pipeline) refunds the user. Any other unexpected exception keeps the count, which is conservative for the shared allowance.
- **Lint:** `QuestionLimitReached` triggers Ruff N818 (exception names should end in "Error"). The spec fixes the name, so a targeted `noqa` was added.
- **2026-10-07, review loop 1 (intent_gap):** review found that unlimited refunds let one user drain the site budget (Review Triage Log #1). The code was reverted (attempt 1 is kept outside the repo as a patch). The product owner revised the failed-call decision to "at most 2 refunds per user per day", and the spec was re-planned with the review's patch findings (#2–#7) built into the tasks. Attempt 1's design is otherwise sound: the hook at the single model-call point, the separate counter connection, `BEGIN IMMEDIATE` reservation, and the test structure (including the 20-thread concurrency test and seeded-count integration tests).

- **2026-10-07, implementation attempt 2 (dispatch subagent):** all tasks done against the re-planned spec. Ruff is clean; full suite 611 passed, 1 skipped (565 existing + 46 new: 27 unit, 19 integration). Refunds are capped at `MAX_REFUNDS_PER_DAY = 2` per user per UTC day via `scope='refund'` rows; a no-op refund (count already 0) does not use up the allowance. Review patches #2–#7 are built in (strict int parsing for TOML and env, singular wording, neutral site wording, README row, midnight-refund tests at unit and API level, `used` description). `utc_now` in `app/limits/usage.py` is module-level so the API midnight test can replace the clock. `test_multi_user.py`'s auth check is generic over OpenAPI paths, so its only change is `checked >= 12`.

## Spec Change Log

## Review Triage Log

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | blind | Refunding the user on every `GenerationError` (including model output that won't parse) lets one user drain the site budget while their own count never moves | high | `query.py` refunds on any `GenerationError`. Parse failures raise it (`pipeline.generate` when `parse_llm_output` returns None), and a user can induce them with a prompt-injecting document. The site count still increments, so the per-user cap does not bound the site budget. | intent_gap (decision 1a) |
| 2 | edge, blind, vgap | `_int` accepts TOML `true` as 1 and `5.9` as 5 without error | low | Confirmed: `int(True) == 1`, `int(5.9) == 5`. The fix is a direct type check. | patch |
| 3 | edge | 429 detail reads "limit of 1 questions" | low | The f-string always says "questions". Direct wording fix. | patch |
| 4 | blind | Site-limit message "You have reached the site's limit" implies the user caused it | low | The message text in `QuestionLimitReached.__init__`. Direct wording fix. | patch |
| 5 | blind | README endpoint table lacks `GET /api/v1/usage` | low | README table checked; the row is missing. Direct doc fix. | patch |
| 6 | blind | No test that a refund crossing midnight applies to the original day | low | `Reservation.day` exists for this; no test covers it. Adding the test is direct. | patch |
| 7 | blind | `used: 0` with limits off is undocumented in the schema | low | `UsageCountModel.used` has no description. A direct comment fix. | patch |
| 8 | blind | The site count may not match provider requests: up to 4 retried requests per question; DNS failures are counted though nothing was sent | maybe-false | Retries exist (`_post_with_retry`), but whether OpenRouter counts rate-limited or failed attempts toward its 50/day is unknown. Settle by observing `X-RateLimit-Remaining` across a retried call. If true: medium. Belongs with 9.2. | defer |
| 9 | edge | A refund raising `OperationalError` in the 502 handler gives a 500 | low | Possible only on a database lock; the refund's transaction is milliseconds long, single process. Unlikely, and the fix adds a guard. | reject |
| 10 | edge | Refund failure on the plain `/query` path | low | Same root cause and reasoning as #9. | reject |
| 11 | edge, blind | `BEGIN IMMEDIATE` lock timeout over 5 s gives a 500 | low | Needs a writer to hold the lock for 5 s; transactions are tiny. Unlikely, and the fix adds handling. If `BEGIN` itself fails, nothing is counted, so the "nothing counted" claim holds. | reject |
| 12 | edge, blind | A non-`GenerationError` exception after the hook keeps the count without a refund | low | Only an unexpected bug path (it is a 500 anyway). Decision 1a covers failed model calls. Unlikely, and the fix adds a branch. | reject |
| 13 | edge | `add_turn` fails after a successful model call, so the count is kept | false | The model call was sent and succeeded, so counting matches the intent ("counts when sent"); the 500 is unrelated to limits. | reject |
| 14 | edge, blind | `set_db_path` global could point to another app's database | low | Existing app-wide pattern (`get_conn` does the same). Production runs one app per process. Unlikely, and the fix adds a helper. | reject |
| 15 | blind | `reserve_question` should check `in_transaction` | false | sqlite already raises "cannot start a transaction within a transaction", a loud and correct failure. | reject |
| 16 | blind | 429 lacks a `Retry-After` header and structured fields | low | Spec requires the scope and reset in `detail`, which is met; `/usage` exposes `resets_at`. The fix adds public surface. | reject |
| 17 | blind | `usage_counters` grows forever and has no cascade on user deletion | false | Accounts cannot be deleted (no such feature); about 2 rows per user per day; Space storage resets anyway. | reject |
| 18 | blind | No test that a refused `/query` with `session_id` leaves no turn | false | `test_follow_up_and_session_query_cannot_bypass` refuses that path and asserts `turns == []` afterwards. | reject |
| 19 | blind | No unit test that the hook is skipped on no-passages | false | `test_no_passages_is_not_counted` seeds the user at the limit; a mutation running the hook earlier fails it (verification-gap layer confirmed). | reject |
| 20 | blind | `test_concurrent_last_slot` is racy (unlocked append, sleep timing) | false | `list.append` is atomic in CPython; the outcome ([200, 429], 5/5) holds whether or not the requests overlap, because reservation is atomic. | reject |
| 21 | edge | Claim "existing tests pass unchanged" is untrue | low | Already recorded in Implementation Notes; the fix would be editing this spec. | reject |
| 22 | vgap | No verification gaps (8 mutations each caught by tests) | n/a | Pre-verified by the verification-gap layer. | none |
| 23 | edge (pass 2) | `BEGIN IMMEDIATE` lock over 5 s gives a 500 | low | carried #11: same location and claim; code unchanged there. | reject |
| 24 | edge (pass 2) | Refund raising in the 502 handler gives a 500 | low | carried #9: same location and claim. | reject |
| 25 | edge (pass 2) | Non-`GenerationError` exception after the hook is not refunded | low | carried #12. | reject |
| 26 | edge, blind (pass 2) | Provider retries mean the site count is not provider requests | maybe-false | carried #8. | defer |
| 27 | edge, blind (pass 2) | `set_db_path` global in `_QuestionCounter._connect` | low | carried #14. | reject |
| 28 | edge (pass 2) | TOML `limits` as a scalar raises `AttributeError` at startup | false | A loud startup failure on malformed config; the same pattern as the existing `[storage]` and `[auth]` sections. Correct behaviour, not a silent defect. | reject |
| 29 | edge, blind (pass 2) | `used` description ("always 0 while limits are off") is wrong after limits are switched off mid-day | low | `usage_summary` reads stored rows regardless of `enabled`, so earlier counts that day are shown. Direct doc correction. | patch |
| 30 | blind (pass 2) | `docs/deployment.md` still says a used-up allowance fails with "Generation failed", and omits the `[limits]` settings and env vars | low | Checked the "Limits to know before inviting users" section. Direct doc fix. | patch |
| 31 | blind (pass 2) | Unit concurrency test `Barrier(20)` has no timeout and could hang the suite | low | A failing worker before `wait()` blocks the other 19 forever. A direct timeout fix. | patch |
| 32 | blind (pass 2) | No logging when a limit refuses a question or a refund is capped | low | No logger calls in `app/limits` or `_QuestionCounter`; the operator of a public demo cannot see budget exhaustion or refund abuse. Trivial log lines, no public surface. | patch |
| 33 | blind (pass 2) | Docstrings of `ask`, `ResearchPipeline.answer` and the routes don't mention `QuestionLimitReached` | low | Read the docstrings; only `GenerationError` and `SessionNotFoundError` are listed. Direct doc fix. (The orphan-session part is unreachable from the API: `_in_session` always passes a session_id.) | patch |
| 34 | blind (pass 2) | Limit validation is duplicated in `_limit()` and `Settings.__post_init__` | low | Two sources of truth for the same rule; the messages are written three times and will drift. The fix is a deletion. | patch |
| 35 | blind (pass 2) | `app.limits` re-exports `utc_now`, but patching it there has no effect | low | `reserve_question` looks up `app.limits.usage.utc_now`; the re-export invites patching the wrong place. The fix is a deletion. | patch |
| 36 | vgap (pass 2) | `UsageCount.remaining` clamp to 0 is not pinned by any test | low | Pre-verified gap: the mutation removing the clamp passed all 46 tests. Add a test with a limit lowered below `used`. | patch |
| 37 | blind (pass 2) | API `test_concurrent_last_slot` does not itself prove atomicity (the sleep is after reservation) | low | True, but atomicity is pinned by the unit 20-thread test (pass-1 vgap mutation of `BEGIN IMMEDIATE` failed it), and the matrix row is covered. Strengthening adds complexity. | reject |
| 38 | blind (pass 2) | Evaluation runner and judge make uncounted model calls on the same key | low | Real, but the eval path is internal and the epic assigns the eval-command warning to its own task (change proposal §4.5), not this story. | defer |
| 39 | blind (pass 2) | Refused questions still pay for retrieval (load from a user at their limit) | medium | Real: the hook runs after `retrieve()`. It is a pre-existing exposure, though: no endpoint has request rate limiting, and this change does not add the cost. | defer |
| 40 | blind (pass 2) | "Called exactly once" is not enforced by `_QuestionCounter` | false | `generate()` calls the hook once (one call site); nothing calls it twice today. | reject |
| 41 | blind (pass 2) | 429 not declared in OpenAPI `responses` | low | Follows the existing pattern (502 and 503 are undeclared); declaring it adds public surface. | reject |

## Design Notes

The hook keeps the limiter out of the pipeline's core logic and is called at exactly one point: after the no-passages early return and before the model call. That makes "counts only when sent to the model" structural rather than duplicated per endpoint:

```python
def generate(self, context, before_model_call=None):
    if not context.passages:
        return <abstention>          # no model call, nothing counted
    if before_model_call:
        before_model_call()          # may raise QuestionLimitReached
    response = self._llm.generate(prompt)
```

Atomic reservation: open with `BEGIN IMMEDIATE`, read both counts, raise if either is at its limit, otherwise upsert both (`INSERT ... ON CONFLICT DO UPDATE SET count = count + 1`), then commit.

## Verification

**Commands:**
- `cd docuresearch && ../.venv/bin/ruff check .` -- expected: no errors
- `cd docuresearch && ../.venv/bin/python -m pytest -q -p no:warnings` -- expected: all pass (565 existing + new)
