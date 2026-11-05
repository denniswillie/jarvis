# Jarvis Supabase Setup

Jarvis uses a separate Supabase project from GitBio.

## Create project

1. Create a new Supabase project for Jarvis.
2. Store the Supabase URL, service role key, and Postgres connection string outside git.
3. Copy the values into `jarvis/backend/.env` on your local machine or EC2 host.

Required backend env values:

```dotenv
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_DB_URL=...
SUPABASE_STORAGE_BUCKET=jarvis-visual-frames
JARVIS_FILES_STORAGE_BUCKET=jarvis-files
```

## Apply initial schema

In Supabase SQL Editor, run:

```sql
-- Paste jarvis/backend/app/schema.sql here.
```

The canonical fresh schema enables `pgcrypto` if needed and creates the current Jarvis tables, including:

- `jarvis_sessions`
- `jarvis_messages`
- `jarvis_user_preferences`
- `visual_frames`

`backend/app/schema.sql` is the canonical **fresh-install** artifact. Its versioned
suffix is generated from `backend/app/migrations/sql/`; never edit an applied
migration. Add the next numbered migration and regenerate the artifact:

```bash
cd backend
python -m app.migrations render > /tmp/jarvis-schema.sql
mv /tmp/jarvis-schema.sql app/schema.sql
```

The original pre-migration schema remains above the generated marker. Do not
truncate the input file by redirecting `render` directly into `app/schema.sql`.
PostgreSQL 15 or newer is required for owner-scoped null-distinct constraints.

Alternatively, initialize an empty configured database transactionally with
`python -m app.migrations init`. This refuses an occupied schema. For an existing
Jarvis database at the pre-second-brain baseline, use `python -m app.migrations apply`.
Use `python -m app.migrations check` for read-only ledger/table readiness. Upgrades
are checksum-verified and serialized with a transaction-scoped advisory lock.
These commands are administrative operations, not model tools or startup hooks.

## Create storage buckets

Create these Supabase Storage buckets:

- `jarvis-visual-frames`
- `jarvis-files`

The visual frames bucket is for raw/ephemeral frame images. Later phases will
add expiration and promotion behavior. The files bucket is for retained
Jarvis-managed files.
