# Isolated interrupted-operation handoff experiment

This is a candidate integration, not a production change or an exactly-once guarantee.
The existing `native` mode remains the default and the control condition.

## What changes in operation_context mode

1. At an interruption, preserve the application's real admission state and read its
   durable execution records using the existing disposable owner/session scope.
2. Include those receipts and the unchanged user message in one realtime text input.
   A receipt distinguishes not started, started, completed, failed, and unknown.
3. Give the model `get_operation_status()`, a read-only way to check the known
   operations without invoking `remember` again. An optional `status_ref` selects
   one operation when needed.

Receipts expose one short `status_ref`. It is derived from the owner/session/call
identity, not from memory content, and remains stable when rebuilding durable
references. Database, provider, and saved-memory IDs stay out of the receipt's
status identity and result summary; raw records remain available in the audit.
A request stopped before admission also has a usable reference and returns
`not_started` normally, rather than failing UUID validation. Lookups without a
reference return the known operations, avoiding identifier copying altogether.
No expected test outcome or desired save count is sent to the model. Receipt
argument/result fields are data, not instructions.
Ordinary tool outputs are still sent; neither a send nor a cancellation is presented
as a provider acknowledgement of the application's completed work.

There is no matching of memory text, automatic retry suppression, write veto, changed
remember schema, new table, production configuration change, or deployment. If the
model asks for another save, it still reaches Jarvis's real executor and memory service.
The database assertions will therefore still expose duplicates.

## Comparison

Use the existing credential-stdin Docker invocation with these runner arguments:

```sh
python -B -m evals.gemini_live.interruption_runner \
  --keys-stdin --output /results/NEW_OUTPUT_DIRECTORY \
  --providers gemini --handoffs native operation_context --repeats 2 --followup-save
```

The four interruption boundaries and user prompts are identical in both modes.
A/B order reverses in the second repetition. The optional follow-up asks for a new
fact to be saved after the original status question. Original-save counts and the
additional save are checked separately, so a missing new save cannot compensate
for an earlier duplicate. Fresh-session readback remains a separate recovery check.

Controls check same-ID durable replay, distinct-ID independent writes, operation
status readback, reference reconstruction, and rejection of another fixture owner's
reference. A failed status lookup now fails the run, even with correct save counts. The existing
cleanup manifest and mandatory owner-scoped cleanup cover both modes.

## Limits

These are text-driven protocol tests, not microphone/VAD or glasses-playback tests.
Receipts reduce ambiguity but do not enforce business-operation idempotency. If the
candidate fails, keep the failing trace rather than changing the expected count.
Read-only lookups add execution records but must not add memory rows. Status and
answer meaning still require transcript review; transport completion alone is not
a semantic pass. Durable references can be reconstructed from database records;
pre-admission cancellation evidence is retained by the application context across
provider reconnections but is not persisted for recovery after an application-process
crash. A production receipt channel would need its own audio/context
ordering, latency, failure, and lifecycle tests before any migration.

Google documents cancellation as a provider tool-call event and matches normal
function responses by provider call ID. That is why this experiment separately
communicates the application's durable outcome rather than assuming a cancelled
provider call remains visible to the model. See the
[Live API protocol reference](https://ai.google.dev/api/live#bidigeneratecontenttoolcallcancellation).
