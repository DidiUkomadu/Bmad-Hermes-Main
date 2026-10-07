# Deploying DocuResearch

DocuResearch runs as one container that keeps everything (accounts, documents, passages, conversations) in a single SQLite database on a mounted volume. For other people to use it, put it behind HTTPS. The included `docker-compose.yml` does this with [Caddy](https://caddyserver.com), which obtains and renews certificates automatically.

## What you need

- A Linux server with Docker and the Compose plugin. **2 GB RAM** is the minimum; the embedding model and PyTorch use about 1 GB.
- A domain name (for example `docs.example.com`) whose DNS A record points to the server.
- Ports 80 and 443 open to the internet. Caddy needs both to obtain certificates.
- An LLM API key, such as OpenRouter.

## Deploy with HTTPS

```bash
git clone https://github.com/DidiUkomadu/Bmad-Hermes-Main.git
cd Bmad-Hermes-Main/docuresearch

cp .env.example .env      # then fill in DOCURESEARCH_LLM_BASE_URL, _MODEL and _API_KEY
chmod 600 .env

DOMAIN=docs.example.com docker compose up -d --build
```

Open `https://docs.example.com`. The first build takes several minutes because it downloads PyTorch and the embedding model. After each start, allow about 30 seconds while the app loads the embedding model: until then Caddy answers 502.

Then:

1. **Register your own account first.** The first account takes ownership of any documents and conversations already in the database.
2. Let your users register.
3. **Close registration** once everyone has an account:
   ```bash
   ALLOW_REGISTRATION=false DOMAIN=docs.example.com docker compose up -d
   ```

## Run without HTTPS (local network or testing only)

```bash
docker build -t docuresearch .
docker run -d --name docuresearch -p 8000:8000 \
  -v docuresearch-data:/data --env-file .env docuresearch
```

Use this only on a trusted network. Without HTTPS, passwords and session cookies travel unencrypted.

## Configuration

Set these in `.env` or in the `environment:` section of `docker-compose.yml`.

| Variable | Default | Purpose |
|---|---|---|
| `DOCURESEARCH_LLM_BASE_URL` | none | OpenAI-compatible endpoint, e.g. `https://openrouter.ai/api/v1` |
| `DOCURESEARCH_LLM_MODEL` | none | Model name |
| `DOCURESEARCH_LLM_API_KEY` | none | API key. Keep it in `.env`, never in the image or in git |
| `DOCURESEARCH_ALLOW_REGISTRATION` | `true` | Whether anyone can create an account |
| `DOCURESEARCH_COOKIE_SECURE` | `false`, but `true` in compose | Send the session cookie over HTTPS only |
| `DOCURESEARCH_DB_PATH` | `/data/docuresearch.db` in the image | Database location |
| `DOCURESEARCH_QUESTIONS_PER_USER_PER_DAY` | `0` (unlimited) | Questions each user may send to the model per UTC day (`[limits] questions_per_user_per_day`) |
| `DOCURESEARCH_QUESTIONS_PER_SITE_PER_DAY` | `0` (unlimited) | Questions all users together may send per UTC day (`[limits] questions_per_site_per_day`) |

## Operations

**Reset a forgotten password.** This also signs the user out everywhere.
```bash
docker compose exec app python -m app.auth.cli reset-password user@example.com
docker compose exec app python -m app.auth.cli list-users
```

**Back up.** Use SQLite's online backup, which is safe while the app is running, then copy the file out:
```bash
docker compose exec app python -c "import sqlite3; sqlite3.connect('/data/docuresearch.db').backup(sqlite3.connect('/data/backup.db'))"
docker compose cp app:/data/backup.db ./docuresearch-backup-$(date +%F).db
```
To restore, stop the app and copy a backup over `/data/docuresearch.db`.

**Upgrade.** The database schema is upgraded in place on startup.
```bash
git pull
DOMAIN=docs.example.com docker compose up -d --build
```

**Logs and health:**
```bash
docker compose logs -f app
curl https://docs.example.com/api/v1/health
```

## Limits to know before inviting users

- **One process.** SQLite and the in-memory sign-in throttle assume a single worker. Don't add `--workers`, and don't run several replicas against one database. This suits a team, not thousands of concurrent users.
- **The model provider sees document content.** Each question sends up to about 20 passages from that user's documents to the LLM provider. Free tiers may log prompts or use them for training. Tell your users, or use a paid provider with a no-retention policy.
- **Free-tier limits are shared.** All users draw on the same API key. OpenRouter's free models allow **50 requests per day** in total (1,000 per day once the account has bought 10 credits), resetting at 00:00 UTC. A busy day can exhaust it. Set the daily question limits (see Configuration; the public demo uses 5 per user and 40 per site, leaving headroom under 50). Once a limit is reached, questions get a 429 "Daily question limit reached" that names the limit and its reset time (00:00 UTC); `GET /api/v1/usage` shows the counts. Only questions sent to the model count. Without limits, questions fail with "Generation failed" when the provider's allowance runs out.
- **Uploads** are capped at 50 MB by Caddy. Scanned (image-only) PDFs are rejected because there is no OCR.
- **No email features.** There is no email verification and no self-service password reset (use the admin command above).
