# Jarvis Backup and Restore

Jarvis durable state lives in the separate Jarvis Supabase Postgres database
and the `jarvis-visual-frames` and `jarvis-files` storage buckets. EC2 contains
runtime containers and secrets, not the source of truth.

## Database backup

Use Supabase managed backups when available. Before a risky schema or bulk-data
change, also create a logical backup from a trusted machine:

```bash
pg_dump "$SUPABASE_DB_URL" --format=custom --no-owner --no-acl \
  --file "jarvis-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

Store the dump encrypted outside EC2 and outside this repository. Never commit
a dump because it can contain conversations, preferences, reminders, OAuth
tokens, and file metadata.

## Database restore rehearsal

Restore into a disposable Supabase project or local Postgres first:

```bash
pg_restore --clean --if-exists --no-owner --no-acl \
  --dbname "$RESTORE_DATABASE_URL" jarvis-YYYYMMDDTHHMMSSZ.dump
```

Use `python -m app.migrations apply` and `python -m app.migrations check` from the
intended release to upgrade a restored baseline. Never rerun the fresh-install
`backend/app/schema.sql` against restored tables. Run backend tests and authenticated diagnostics
against the restored environment before switching production configuration.

## Storage backup

The files bucket contains retained user files and important promoted frames;
back it up. Raw visual frames are intentionally ephemeral and normally do not
need archival backup. Use the Supabase Storage API or an S3-compatible export
tool to copy bucket objects and preserve their object paths.

## Restore order

1. Restore Postgres into an isolated target.
2. Restore retained `jarvis-files` objects using their original paths.
3. Restore retained objects from `jarvis-visual-frames` only when required.
4. Configure a staging backend against the restored project.
5. Run `check_production_health.sh` and functional note/reminder/file checks.
6. Rotate credentials before using the restored environment as production.
