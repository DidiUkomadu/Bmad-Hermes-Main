# DocuResearch — Change Proposal: Multi-User Accounts and Data Isolation

**Status:** Approved by product owner (2026-10-02)
**Type:** Scope change (BMad Correct Course)
**Supersedes:** The "single-user; authentication and multi-tenancy excluded" MVP constraint in the PRFAQ, Architecture v1, Epics & Stories, and AGENTS.md. Those documents are not edited; this proposal amends them.

---

## 1. Trigger

The product owner wants several people to use one DocuResearch instance, each signing in and seeing only their own documents and conversations.

The MVP was single-user. It has no notion of users, so every document, passage, answer, citation and conversation is visible to anyone who can reach the server. The most serious consequence is that an answer for one user can be retrieved from, and cite, another user's documents.

## 2. Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Authentication method | Local email + password accounts | Works offline and with no third-party setup; consistent with local-first. OAuth can be added later. |
| D2 | Password storage | `scrypt` (Python stdlib) with a per-user random salt; constant-time comparison | Memory-hard, standard, no new dependency. |
| D3 | Session mechanism | Random token in an HttpOnly, SameSite=Lax cookie; only the token's SHA-256 hash is stored; default lifetime 7 days; deleted on sign-out | Server-side sessions are revocable. HttpOnly keeps the token away from page scripts, and SameSite=Lax blocks cross-site form submissions (CSRF). |
| D4 | Registration | Open self-registration, configurable (`auth.allow_registration`) | Easy for demos; can be closed for a deployed instance. |
| D5 | Data visibility | Every document and conversation belongs to exactly one user; no sharing | The safe default. Shared workspaces are a separate future feature. |
| D6 | Document and passage IDs | Derived from owner + filename for uploads made through the app; unchanged (filename only) when no owner is given | Two users can upload files with the same name. Evaluation-dataset IDs stay stable. |
| D7 | Existing data | Documents and conversations with no owner are assigned to the first account registered | Nothing already uploaded is lost when moving from single-user. |
| D8 | Password reset | Admin command: `python -m app.auth.cli reset-password <email>` | Email-based reset needs an email service, which the local-first MVP does not have. |
| D9 | Brute-force protection | After 5 failed sign-ins for an email, sign-in is refused for 5 minutes (in-memory) | Basic protection for a single-process server. |
| D10 | Error behaviour | A resource belonging to another user returns 404, never 403 | Existence of other users' data is not revealed. |

## 3. Requirements (acceptance criteria)

**Accounts and sessions**
- [ ] A person can register with email, password (minimum 8 characters) and an optional display name; emails are unique and case-insensitive.
- [ ] A person can sign in and out; sign-out invalidates the session on the server.
- [ ] Sessions expire after the configured lifetime.
- [ ] Passwords and session tokens are never stored in plain text.
- [ ] Repeated failed sign-ins are throttled (D9).
- [ ] Registration can be disabled by configuration.

**Isolation (the critical requirement)**
- [ ] Every API endpoint except health, sign-in, registration and the static UI requires a signed-in user (401 otherwise).
- [ ] Users list, open, rename and remove only their own documents.
- [ ] **Retrieval only searches the signed-in user's passages.** No answer, citation or carried-forward follow-up passage can come from another user's document.
- [ ] Citation lookup returns only the user's own passages (404 otherwise).
- [ ] Conversations are listed, opened, continued and deleted only by their owner.
- [ ] Two users can upload files with the same name without conflict.

**Migration**
- [ ] Existing unowned documents and conversations are assigned to the first registered user (D7).
- [ ] Existing databases are upgraded in place without data loss.

**UI**
- [ ] A sign-in / registration screen is shown to signed-out visitors.
- [ ] The signed-in user's name and a Sign out control are visible.
- [ ] Signing out (or a session expiring) returns to the sign-in screen, and the previous user's data is no longer displayed.

## 4. Impact on existing artifacts

| Artifact | Impact |
|---|---|
| PRFAQ | The "single-user" MVP scope is replaced by multi-user with private libraries. |
| Architecture v1 | New component: authentication (users, sessions). The content store gains an owner on documents and conversations. Retrieval and citation resolution gain an owner constraint. |
| Epics & Stories | New Epic 8 (below). Stories 2.1, 3.4, 5.1, 6.1 and 7.2–7.5 gain ownership checks. |
| Implementation plan | §5 API: new `/api/v1/auth/*` endpoints; all other endpoints require a session. §6 schema: new `users` and `auth_sessions` tables; `owner_id` on `documents` and `conversations`. |
| AGENTS.md | Scope section updated: authentication and multi-user isolation are in scope; billing and enterprise administration remain out. |
| Evaluation | Unaffected. The runner uses the pipeline directly without an owner, so dataset IDs are unchanged. |

## 5. Epic 8 — Multi-User Accounts and Data Isolation

- **8.1 Accounts:** users table, registration, password hashing, admin password reset.
- **8.2 Sessions:** sign-in, sign-out, session cookie and expiry, throttling of failed sign-ins.
- **8.3 Document ownership and retrieval isolation:** owner on documents, owner-scoped document IDs, retrieval and citation lookup restricted to the owner.
- **8.4 Conversation ownership:** owner on conversations; list, open, continue and delete restricted to the owner; carried-forward passages checked for ownership.
- **8.5 Sign-in UI:** sign-in and registration screens, current user display, sign-out.
- **8.6 Migration:** in-place schema upgrade; first registered user adopts unowned data.

## 6. Still out of scope

Billing, enterprise administration, roles and permissions, shared workspaces, OAuth / single sign-on, email verification and email-based password reset, and multi-process deployment (the sign-in throttle is in-memory).
