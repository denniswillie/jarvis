# Durable background tasks

This package owns task persistence and execution coordination, not voice sessions,
model selection, attention delivery, or the memory-maintenance queue. Closing a
voice session does not cancel its owner's tasks.

## Availability and administration

`SECOND_BRAIN_ENABLED=true` and `SECOND_BRAIN_TASKS_ENABLED=true` enable the optional
runtime. Initialization checks the migration ledger without applying DDL. Invalid
configuration or an unavailable task schema disables this subsystem, not voice.
Apply the reviewed migrations explicitly through `python -m app.migrations`.

Providers are trusted application adapters registered in
`JarvisRuntimeServices.background_task_providers`. At the S02 foundation stage this
map is empty: enabling the scheduler alone does not install a frontier model or
sandbox. A new request for an unavailable provider returns 503 without accepting
work. Replaying an already accepted create request can still return its durable
record. The HTTP fixture provider is never registered in production.

All task settings use the `SECOND_BRAIN_` environment prefix. Defaults are two
active attempts per owner, four globally, twenty pending tasks per owner, eight
attempts per task, thirty-second leases, half-second polling, and five-second
shutdown waits. Resource defaults are 900 active seconds, 40 tool steps, 16,000
output tokens per request, 32,000 output and 250,000 input tokens per task, and
2,000,000 reserved/known input-plus-output tokens per owner per UTC day.
These are resource limits, not currency-denominated billing promises.

The first claim initializes shared database concurrency limits. Subsequent workers
use those shared values rather than replacing them with their own environment.
After deliberately changing owner/global concurrency, an administrator can run:

```sh
python -m app.tasks configure-limits
```

Lowering a cap drains existing attempts instead of killing them. Existing tasks
retain their accepted resource budget across updates, resumes, and retries.
Resume is not permission to reset exhausted budgets or replace expired grants.
Budget/authority changes require a separate explicit policy operation; subsequent
conversation integration must not hide that requirement behind automatic retries.

## HTTP contract

All endpoints use the existing static-bearer authentication and derive the owner
from the authenticated principal, never the request body.

| Endpoint | Purpose |
| --- | --- |
| `POST /tasks` | Accept a UUID-identified request for a configured backend. |
| `GET /tasks` | List bounded owner-scoped task summaries. |
| `GET /tasks/{id}` | Read the current task revision and real execution state. |
| `GET /tasks/{id}/events` | Read ordered events, including results or clarification payloads. |
| `POST /tasks/{id}/updates` | Supply revised instructions with an expected revision. |
| `POST /tasks/{id}/cancel` | Cancel exactly one task, subject to reconciliation. |
| `POST /tasks/{id}/resume` | Explicitly resume eligible work without resetting consumption. |

Mutation bodies contain `request_id`, `expected_revision`, and optional `message`.
Updates require nonblank instructions. Retrying the same request ID and identical
parameters returns the existing task; reusing it with changed parameters conflicts.
A genuinely new request ID represents new intent even if the text is identical.
Queue saturation returns 429. Unknown or foreign task IDs return 404.

## Provider adapter obligations

Implement `TaskWorkerProvider.execute` and `reconcile`. Both run outside database
transactions. The worker context contains fenced checkpoint, reservation,
settlement, and action-ledger helpers. Adapters must follow this ordering:

1. Reserve bounded input/output/tool resources with a stable logical operation key.
2. Record dispatch intent before starting external provider I/O.
3. Save the actual provider handle and useful continuation state promptly.
4. Record measured usage and native tool receipts, rather than trusting model prose.
5. Return an explicit result, failure, clarification, or pause.

Recovery must inspect or cancel the existing handle, not create replacement work.
An absent handle after recorded dispatch is ambiguous, not proof that nothing ran.
Unknown usage remains charged. A newer task revision or recovery fence prevents an
old worker from publishing a current result. Recovery also rechecks authority and
active-time limits before continuing remote work.

External effects additionally use `TaskActionRepository`: prepare exact parameters
under one immutable operation grant, authorize, then atomically reserve a tool step
and cross the submission boundary. A grant cannot fan out into multiple sends under
different model call IDs. Receipt tokens permit recording late outcomes, not another
execution. Native reconciliation is an internal adapter operation, not a model tool.
An ambiguous effect stays unresolved until native evidence establishes its outcome.
Cancelling after submission cannot undo an already accepted effect.

Task success does not imply delivery to the user. Later attention and voice stages
consume the durable events and must respect task revision and result availability.

## Tests

Unit tests do not require a database. Native tests require explicit opt-in and use
disposable schemas on the configured database, with manifest-based cleanup:

```sh
python -m evals.second_brain.run_tests --live-db \
  --manifest-dir /tmp/jarvis-second-brain-manifests \
  --pattern 'test_second_brain_tasks*.py'
```

The native suite exercises real concurrent PostgreSQL transactions and subprocess
SIGKILL/restart against a loopback HTTP provider. It checks queued, waiting, and live
work plus a crash after an external effect but before receipt persistence. This is
real scheduler/ledger evidence, not evidence of frontier-model quality, E2B behavior,
email delivery, push delivery, or glasses acceptance.
