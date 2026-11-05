# Jarvis Secret Rotation

Secrets live in `backend/.env` on EC2 and in the matching provider consoles.
Never paste real values into deployment documentation, shell history, commits,
or diagnostics output.

## Rotation procedure

1. Create the replacement credential at the provider.
2. Update `backend/.env` on EC2 using a secure editor.
3. Run `cd deployment && ./restart.sh`.
4. Run the authenticated health and websocket checks.
5. Confirm the affected feature works.
6. Revoke the old credential only after validation.

## Credential-specific notes

- `JARVIS_STATIC_BEARER_TOKEN`: update EC2 first, then update the iOS secure setting immediately. Existing app connections will need to reconnect.
- `SUPABASE_DB_URL`: create/rotate the database password, update the pooler URL, restart, and confirm `/ready` before revoking the old password.
- `SUPABASE_SERVICE_ROLE_KEY`: rotate in Supabase, restart, then verify both storage buckets through `/diagnostics/config`.
- `OPENAI_API_KEY`: verify realtime audio, embeddings, reminders, and web search after rotation.
- `GEMINI_API_KEY`: verify a visual frame reaches analyzed state after rotation.
- Google OAuth client secret/state secret: update Google and EC2 together. Existing OAuth tokens may remain valid, but reconnect Calendar if Google invalidates them.

After any suspected exposure, rotate the affected credential immediately and
inspect application/provider logs. The diagnostics endpoints show only whether
configuration is present and working; they never return secret values.
