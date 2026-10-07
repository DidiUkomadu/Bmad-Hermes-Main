# Sprint Change Proposal: Public Beta on Hugging Face Spaces (Epic 9)

**Date:** 2026-10-07
**Workflow:** BMad Correct Course (batch mode)
**Product owner:** Didi
**Status:** Approved by the product owner (2026-10-07)
**Scope classification:** Moderate (new epic; no existing work reopened)

---

## 1. Issue Summary

**Trigger:** a new requirement from the product owner. The MVP is complete (Epics 0–8, see `implementation-artifacts/sprint-status.yaml`), and the product owner wants anyone on the internet to be able to try DocuResearch.

**Constraints, decided by the product owner:**
- Hosted on **Hugging Face Spaces, free tier** (Docker).
- **OpenRouter free tier only**, using the owner's single shared key. No paid credits, and no other providers.
- **Open to anyone**: no invite codes.

**Problem statement:** DocuResearch is built and tested for a trusted, local setup with no usage limits. Exposed publicly under these constraints, one visitor can exhaust the shared daily model allowance for everyone, bots can register freely, stored data disappears without warning, and visitors are not told that their document passages go to third-party AI providers.

**Evidence, from this project:**

| # | Evidence |
|---|---|
| E1 | OpenRouter's free tier allows **50 requests per day per account**. Confirmed from OpenRouter's own error: `free-models-per-day`, `X-RateLimit-Limit: 50`, reset at 00:00 UTC. One question uses one request, so ten visitors asking five questions each would exhaust it. |
| E2 | When the daily cap is hit, the model connector retries up to 3 more times (`app/generation/openai_compat.py`), and those retries cannot succeed before the reset. One question can waste 4 requests. |
| E3 | Free Spaces storage is wiped when the Space restarts or wakes from sleep, so accounts, documents and conversations are temporary. |
| E4 | Hugging Face shows a Space inside an embedded frame on huggingface.co. DocuResearch's session cookie is `SameSite=Lax`, which browsers do not send in a cross-site frame, so sign-in would fail on the Hugging Face page. It works at the Space's direct `*.hf.space` address. (Expected browser behaviour; to be verified on a real Space.) |
| E5 | Passages are sent to free OpenRouter endpoints, some of which may log prompts or use them for training. Visitors are not currently told. |
| E6 | Registration has no throttle; only failed sign-ins are rate-limited (`app/auth/service.py`, `LoginThrottle`). |

---

## 2. Impact Analysis

### Epic impact
- **Epics 0–8:** complete and unchanged. Nothing is reopened or rolled back.
- **New Epic 9: Public Beta on Hugging Face Spaces.** Six stories (§4.1). Priority: daily limits first, because without them a single visitor can stop the demo for everyone.

### Artifact conflicts

| Artifact | Conflict | Resolution |
|---|---|---|
| PRFAQ §4 Target users | Primary user is technical professionals; others are deferred. "Open to anyone" invites deferred users. | No feature changes for secondary users. The beta demonstrates the same find-and-verify job. (§4.2) |
| PRFAQ §6 Deferred scope | "single-user, local-first" | Already superseded for multi-user (Epic 8). Public hosting supersedes "local-first". (§4.2) |
| PRFAQ §10 Risks | No availability risk | Add a risk: the free allowance runs out daily. (§4.2) |
| Architecture §2 / §10 | No usage limiting; local-first, single-user | Add a usage limiter, upload limits, sign-up throttling, and provider-cap detection. (§4.3) |
| Architecture §3 Data model | No usage records | New `usage_counters` table. (§4.3) |
| Implementation plan §5 API | No limit responses or usage endpoint | New 429 and 503 responses; `GET /api/v1/usage`; registration requires accepting the privacy notice. (§4.3) |
| UI (no UX document) | No limits, banner or notice | Banner, privacy notice, remaining-questions counter, limit messages. (§4.4) |
| `docs/deployment.md` | Covers a VPS with Caddy only | Add a Hugging Face Spaces section. (§4.5) |

### Technical impact
- **Code:** a new usage-limit module and table, a change to how model-connector errors are classified, ingestion and upload checks, a registration change, a static privacy page, and UI updates.
- **Infrastructure:** Hugging Face Space configuration; the database path must be writable by the user Spaces runs the container as.
- **Evaluation:** evaluation runs draw on the same 50 daily requests as the demo. Run them only when the demo can spare the day.

---

## 3. Recommended Approach

**Direct adjustment (checklist option 1):** add Epic 9 within the current architecture.

| Option | Verdict |
|---|---|
| 1. Direct adjustment | **Selected.** Every story extends existing components (store, auth, model connector, UI). Effort: medium. Risk: low to medium. |
| 2. Rollback | Not viable. Nothing completed obstructs the change; multi-user (Epic 8) is a prerequisite. |
| 3. MVP review | Not needed. The MVP is complete and unaffected; this is additive. |

**Risks:**
- **Availability.** 50 requests a day is very small; the demo will often be out of questions. Mitigated by caps and honest messaging, not eliminated.
- **Hugging Face specifics not yet verified on a live Space:** container user ID and writable paths, frame and cookie behaviour, sleep behaviour. Story 9.6 verifies them first.

**Effort:** six stories of a similar size to Epic 8's.

---

## 4. Detailed Change Proposals

### 4.1 New stories: Epic 9, Public Beta on Hugging Face Spaces

All limits are configuration settings. **By default they are off** (no limits), so local and VPS installs behave as today; the Hugging Face Space turns them on. Defaults suggested for the Space are shown.

#### Story 9.1: Daily question limits
**Goal:** one visitor cannot exhaust the shared daily model allowance.

Acceptance criteria:
- [ ] Each user may ask at most **5 questions per day** (`limits.questions_per_user_per_day`).
- [ ] The whole site may ask at most **40 questions per day** (`limits.questions_per_site_per_day`), leaving headroom under OpenRouter's 50.
- [ ] Days reset at **00:00 UTC**, matching OpenRouter.
- [ ] A question counts when it is sent to the model. Questions answered without a model call (no passages retrieved) do not count.
- [ ] Follow-ups, conversation turns and standalone questions all count. The limit cannot be bypassed by any endpoint.
- [ ] Over a limit, the API returns **429** with `error` set to "Daily question limit reached", saying which limit (yours or the site's) and when it resets.
- [ ] `GET /api/v1/usage` returns the user's and the site's used and remaining counts, and the reset time.
- [ ] Counts are stored in SQLite (`usage_counters`: day, scope, user ID, count) and survive app restarts, though not Space rebuilds.
- [ ] When the limit settings are 0 or unset, there is no limit (today's behaviour).

#### Story 9.2: Handling OpenRouter's daily cap
**Goal:** never waste requests once the provider's daily allowance is gone, and tell visitors clearly.

Acceptance criteria:
- [ ] A 429 that signals the **daily** cap (body contains `free-models-per-day`, or `X-RateLimit-Remaining: 0`) is **not retried**. Short "busy" 429s are still retried, as today.
- [ ] After a daily-cap 429, the app marks the provider exhausted until the reset time OpenRouter reports (`X-RateLimit-Reset`), or 00:00 UTC if none is given.
- [ ] While exhausted, questions fail immediately, without calling OpenRouter, with **503** "The demo has used today's free AI allowance" and the reset time.
- [ ] The health check reports whether the model allowance is exhausted.
- [ ] Exhaustion is held in memory; a restart clears it, and the next daily-cap 429 sets it again.

#### Story 9.3: Upload limits
**Goal:** keep the free Space responsive and storage small.

Acceptance criteria:
- [ ] Maximum file size **5 MB** (`limits.max_upload_mb`), rejected with **413** before parsing.
- [ ] Maximum **50 pages** per PDF (`limits.max_pdf_pages`) and **400 passages** per document for any format (`limits.max_passages_per_document`), rejected with **422** and the limit in the message.
- [ ] Maximum **5 documents per user** (`limits.max_documents_per_user`), rejected with **409** "Document limit reached"; removing one frees a slot.
- [ ] Nothing is stored when a limit rejects an upload.
- [ ] The upload form shows the limits.

#### Story 9.4: Sign-up protection
**Goal:** slow down bots and runaway account creation without email verification.

Acceptance criteria:
- [ ] At most **3 registrations per hour per client IP** (`limits.registrations_per_ip_per_hour`), rejected with **429** "Too many sign-ups from your network".
- [ ] At most **500 accounts** in total (`limits.max_accounts`), rejected with **403** "The demo is full"; the data resets with the Space anyway.
- [ ] The client IP is read from `X-Forwarded-For` only when `limits.trust_proxy_headers` is on (it is on for Hugging Face); otherwise from the direct connection, so the address cannot be spoofed locally.
- [ ] The existing failed-sign-in throttle is unchanged.

#### Story 9.5: Privacy notice and demo banner
**Goal:** visitors know where their data goes and that it is temporary.

Acceptance criteria:
- [ ] When demo mode is on (`demo.enabled`), every screen shows a banner: *"Public demo. Data is deleted when the demo restarts. Don't upload confidential documents: passages are sent to third-party AI providers via OpenRouter's free tier, which may log them."*
- [ ] A privacy page (`/privacy`) explains what is stored, for how long, what is sent to OpenRouter, and how to delete documents and conversations.
- [ ] In demo mode, registration requires ticking "I have read the privacy notice". The API rejects registration without it (400), and records the acceptance time on the account.
- [ ] Signed-in users see "*N* of 5 questions left today" near the question box, updated after each answer.
- [ ] Limit errors from 9.1–9.4 appear as friendly messages, not raw errors.

#### Story 9.6: Hugging Face Spaces deployment
**Goal:** a working public Space that can be redeployed with one documented command.

Acceptance criteria:
- [ ] The Space runs the existing Docker image with Space settings (`sdk: docker`, the app port), and with demo mode and the limits turned on through Space variables.
- [ ] Secrets (`DOCURESEARCH_LLM_API_KEY`) are stored as Space secrets and never in the repository.
- [ ] The database lives in a directory writable by the user the Space runs the container as. This is verified on the live Space first, before the other configuration.
- [ ] The session cookie is `Secure` and works at the direct `*.hf.space` address. The Space description and README link to that address, because sign-in inside the embedded huggingface.co frame is not supported (E4).
- [ ] The Space is deployed from the `docuresearch/` folder of the GitHub repo with a single documented command (for example `git subtree push`). No separate copy of the code is maintained.
- [ ] `docs/deployment.md` gains a Hugging Face Spaces section covering setup, secrets, limits, redeploying, and the data-loss caveat.
- [ ] Smoke test on the live Space: register, upload, ask, check the citation, hit the per-user limit, and see the banner.

### 4.2 PRFAQ amendments
Applied through this proposal; the PRFAQ file itself is not edited (same practice as Epic 8).

**§4 Target users, appended:**
> **Public beta (post-MVP):** the hosted demo is open to anyone, but is designed and evaluated for the primary user's find-and-verify job. No features are added for secondary users.

**§6 MVP explicitly deferred: authentication bullet.**
OLD: "Authentication, billing, multi-tenancy, enterprise administration: out of MVP scope (single-user, local-first)."
NEW: "Billing and enterprise administration: out of scope. Accounts and private libraries were added in Epic 8. A public demo on Hugging Face Spaces is added in Epic 9, with usage limits and no payments."

**§10 Risks, new risk:**
> **Risk: the free model allowance runs out.** The public demo shares one free OpenRouter allowance (50 requests a day). **Mitigation:** per-user and site-wide daily limits, no wasted retries, and clear "come back after 00:00 UTC" messaging. The model-agnostic design allows a paid tier later as a configuration change.

### 4.3 Architecture and implementation-plan amendments
- **New component: usage limiter** (`app/limits/`). Used by the query and conversation endpoints before generation. Backed by the `usage_counters` table. Configurable, and off by default.
- **Model connector:** classifies 429s as *busy*, which is retried, or *daily cap*, which is not retried and marks the provider exhausted until reset.
- **Ingestion and upload:** size, page, passage and document-count checks before anything is stored.
- **Auth:** sign-up throttle per IP; an account cap; `privacy_accepted_at` on users, added through the existing in-place migration.
- **API (§5) additions:** `GET /api/v1/usage`; `/privacy`; 429 "Daily question limit reached"; 503 "AI allowance used"; 413 and 409 upload limits; 403 "The demo is full". Registration adds an `accept_privacy` field, required in demo mode.
- **Deployment:** Hugging Face Spaces joins the existing VPS (Caddy) option.

### 4.4 UI changes (no UX document; recorded here)
- **Demo banner** on every screen (demo mode only).
- **Sign-up:** privacy checkbox, with a link to `/privacy`.
- **Question box:** "*N* of 5 questions left today".
- **Upload form:** shows the limits.
- **Errors:** friendly limit messages.

### 4.5 Other artifacts
- `docs/deployment.md`: a Hugging Face Spaces section.
- `config.example.toml`: new `[limits]` and `[demo]` sections, documented, off by default.
- The evaluation command warns that a run uses the same daily allowance as the demo.
- After approval: AGENTS.md (deployment, status) and `sprint-status.yaml` (Epic 9 stories as backlog).

---

## 5. Implementation Handoff

**Scope:** moderate. A new epic is added to the backlog; there is no replanning of existing work.

| Role | Responsibility |
|---|---|
| Product owner (Didi) | Approve this proposal and the limit values. Create the Hugging Face account and Space. Add the OpenRouter key as a Space secret. Run the final smoke test. |
| Developer agent | Implement stories 9.1–9.6 in order, each with tests, following the BMad build workflow. Update `sprint-status.yaml` as stories complete. Write no secrets to the repo. |

**Sequence:** 9.1, then 9.2 (both protect the allowance), then 9.3 and 9.4, then 9.5, then 9.6. 9.6 starts with verifying the live Space, which can begin as soon as the Space exists.

**Success criteria:**
- [ ] Anyone can open the Space's direct address, register, upload a document, and get a cited answer.
- [ ] No visitor can use more than their daily share. The site never exceeds 40 questions a day, and never wastes requests after OpenRouter's daily cap.
- [ ] Every visitor sees the demo banner and accepts the privacy notice before uploading.
- [ ] All existing tests still pass, with new tests for every limit, including tests that limits cannot be bypassed via follow-ups or conversations.
- [ ] Local and VPS installs behave exactly as before unless the limits are turned on.

---

## Decisions on open questions (product owner, 2026-10-07)
1. **Limit values: confirmed as proposed.** 5 questions per user per day, 40 per site, 5 MB, 50 pages, 400 passages per document, 5 documents per user, 3 sign-ups per IP per hour, 500 accounts.
2. **Sample documents: not preloaded.** New accounts start empty; visitors upload their own documents.
