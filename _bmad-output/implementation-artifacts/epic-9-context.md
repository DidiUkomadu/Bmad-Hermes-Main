# Epic 9 Context: Public Beta on Hugging Face Spaces

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Make the completed multi-user MVP (Epics 0–8) safely usable by anyone on the internet, hosted on a free Hugging Face Space (Docker) and powered only by the owner's single OpenRouter free-tier key. That key allows just 50 model requests per day across the whole site, resetting at 00:00 UTC. Without protection, one visitor could use up the day's allowance for everyone, bots could register freely, data would disappear without warning when the Space restarts, and visitors would not know their passages go to third-party AI providers. The epic adds usage limits, handling for the provider's daily cap, upload and sign-up limits, a privacy notice with a demo banner, and a documented, repeatable Space deployment. It only adds to existing components; nothing in Epics 0–8 is reopened.

## Stories

- Story 9.1: Daily question limits
- Story 9.2: Handling OpenRouter's daily cap
- Story 9.3: Upload limits
- Story 9.4: Sign-up protection
- Story 9.5: Privacy notice and demo banner
- Story 9.6: Hugging Face Spaces deployment

## Requirements & Constraints

- **Off by default.** Every limit and demo-mode behaviour is a configuration setting (new `[limits]` and `[demo]` sections, documented in the example config). When a limit is 0 or unset there is no limit, so local and VPS installs behave exactly as they do today. The Space turns the limits on through Space variables.
- **Confirmed limit values for the Space:** 5 questions per user per day, 40 per site per day (leaving headroom under the provider's 50), 5 MB per upload, 50 pages per PDF, 400 passages per document, 5 documents per user, 3 sign-ups per IP per hour, 500 accounts in total.
- **Question counting:** a question counts only when it is actually sent to the model. Answers given without a model call (no passages retrieved) do not count. Standalone questions, follow-ups and conversation turns all count, and no endpoint may bypass the limit. Days reset at 00:00 UTC.
- **Don't waste provider requests:** once the provider's daily cap is hit, the app makes no further calls to it until the reset.
- **Uploads:** reject before anything is stored. Size is checked before parsing.
- **Per-user data isolation still applies.** Per-user counts and document counts must use the signed-in user's ID. The usage endpoint shows the caller only their own counts plus the site totals.
- **Grounding still applies.** Limit and allowance failures are explicit errors and never a fallback or fabricated answer.
- **No secrets in the repo.** The LLM API key is a Space secret only.
- **Evaluation shares the allowance.** Evaluation runs use the same 50 daily requests (a full judged run needs about 72). The evaluation command must warn about this.
- **Success criteria:** anyone can open the Space's direct address, register, upload, and get a cited answer. No visitor exceeds their share. The site never exceeds 40 questions a day or wastes requests after the cap. Every visitor sees the banner and accepts the privacy notice. All existing tests pass, with new tests for every limit, including tests that limits cannot be bypassed through follow-ups or conversations.
- **Out of scope:** payments or paid credits, other providers, invite codes, email verification, and preloaded sample documents (new accounts start empty).

## Technical Decisions

- **Usage limiter**: a new component that the query and conversation endpoints call before generation. It is backed by a SQLite `usage_counters` table (day, scope [user or site], user ID, count). Counts survive app restarts but not Space rebuilds.
- **Model connector 429 classification:** a *busy* 429 is retried as today. A *daily-cap* 429 (the body contains `free-models-per-day`, or `X-RateLimit-Remaining: 0`) is not retried. It marks the provider exhausted until the time in `X-RateLimit-Reset`, or 00:00 UTC if that header is missing. Exhaustion is held in memory, so a restart clears it and the next cap response sets it again. The health check reports whether the allowance is exhausted.
- **Auth:** a per-IP sign-up throttle and an account cap. The client IP comes from `X-Forwarded-For` only when `limits.trust_proxy_headers` is on (it is on for Hugging Face); otherwise it comes from the direct connection. The existing failed-sign-in throttle is unchanged. Users gain a `privacy_accepted_at` column, added through the existing in-place schema migration. Registration gains an `accept_privacy` field that is required in demo mode.
- **API additions** (all use the existing `{"error", "detail"}` response shape):
  - `GET /api/v1/usage`: user and site used/remaining counts and the reset time.
  - `/privacy`: a static page.
  - 429 "Daily question limit reached": says which limit (yours or the site's) and when it resets.
  - 503 "The demo has used today's free AI allowance": includes the reset time and makes no provider call.
  - 413 when a file is over the size limit; 422 when the page or passage limit is exceeded, with the limit in the message.
  - 409 "Document limit reached"; removing a document frees a slot.
  - 429 "Too many sign-ups from your network".
  - 403 "The demo is full".
  - 400 when registration is sent without accepting the privacy notice in demo mode.
- **Deployment:** Hugging Face Spaces is added alongside the existing VPS (Caddy) option. It reuses the existing Docker image with `sdk: docker` and the app port. It is deployed from the repo's `docuresearch/` folder with one documented command (for example `git subtree push`); no separate copy of the code is kept. The database must live in a directory writable by the container user the Space runs as. The session cookie must be `Secure`.
- **Known browser limitation:** the session cookie is `SameSite=Lax`, so sign-in fails inside the embedded huggingface.co frame and works only at the direct `*.hf.space` address. The Space description and README link to that address.

## UX & Interaction Patterns

There is no UX document; these changes are recorded only in the change proposal.
- **Demo banner on every screen (demo mode only):** "Public demo. Data is deleted when the demo restarts. Don't upload confidential documents: passages are sent to third-party AI providers via OpenRouter's free tier, which may log them."
- **Privacy page:** explains what is stored, how long it is kept, what is sent to OpenRouter, and how to delete documents and conversations.
- **Sign-up form:** an "I have read the privacy notice" checkbox linking to `/privacy` (demo mode).
- **Question box:** "*N* of 5 questions left today", updated after each answer.
- **Upload form:** shows the limits.
- **Limit errors** from stories 9.1–9.4 appear as friendly messages, not raw errors.

## Cross-Story Dependencies

- **Order:** 9.1 and 9.2 first (both protect the allowance), then 9.3 and 9.4, then 9.5, then 9.6.
- Story 9.5's question counter needs 9.1's usage endpoint, and its friendly messages cover the errors from 9.1–9.4.
- Story 9.6 turns on the settings from 9.1–9.5. Its first step, checking the container user and writable paths, cookie and frame behaviour, and sleep behaviour on the live Space, can start as soon as the Space exists. It ends with a live smoke test: register, upload, ask, check the citation, hit the per-user limit, and see the banner. It also adds a Hugging Face Spaces section to the deployment docs (setup, secrets, limits, redeploying, and the data-loss warning).
- The epic depends on Epic 8: accounts, sessions, owner-scoped data, and the in-place migration.
