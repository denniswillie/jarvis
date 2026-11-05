# Glasses reliability evidence

`baselines.py` names the eleven observed failure cases and their acceptance
oracles. These are a regression manifest, not claims that the issues are fixed.
Use the staged plans in `planning_docs/glasses_reliability` for implementation.

Both audio paths emit `realtime_trace` JSON records with a schema version, UTC
timestamp, process identity, monotonic time, and available correlation IDs.
Phone diagnostics use the same prefix. Keep real session logs private.
No audio, image bytes, transcripts, tool arguments, URLs, or credentials belong
in these records. Codes and identifiers are bounded; arbitrary payload keys
and nonfinite clocks are discarded. Do not put user content in identifier fields.

Run the summary from the backend directory:

```sh
python -m evals.glasses_reliability.report < /private/path/session-traces.log
```

The summary joins GPT request IDs to response IDs and Gemini response IDs
directly. It reports response-write-to-first-forwarded-audio and provider-start
to first-forwarded-audio only when the correlated clocks share a process.
Missing or reversed timing evidence remains null, never zero. Duplicate first
audio records are deduplicated in the report; the live logger retains at most
256 recent response/item pairs, so an ancient retired ID can be logged again
after eviction. Provider completion is not playback completion.

Image selection, prewrite eligibility, successful socket write, and supported
provider acknowledgement are separate evidence. Gemini still-image writes have
no image-specific acknowledgement; a selected/written image does not prove OCR.

Phone audio logs distinguish receipt, enqueue, play request, observed render
clock, explicit stop, and queue drain. Existing buffer completion means consumed,
not necessarily played. The render clock is sampled on existing callbacks and
is not the exact first-sample time. Bluetooth/device latency and actual audible
onset remain physical acceptance checks. Camera traces carry capture timestamp,
sequence, visual session, and linked audio session without image content.

Stage 1 is diagnostics-only: no new wire fields, provider changes, turn-policy
changes, model instructions, or database writes. Deterministic tests use fake
providers and isolated loopback sockets. Native evaluation is a separate gate.

## Stage 2: visual reference contract and partial results

Stage 2 introduced request-facing reference contract 2. Ordered
frames use frame: references, existing entities entity:, active focus focus:,
and conditional reminders reminder:. The table belongs to one owner, session,
and window. Only exact supplied references or their authorized durable IDs
resolve; unknown, duplicate, ambiguous, or cross-scope IDs are never guessed.
The previous UUID-only prompt/parser remains available for compatibility and
comparison. The persisted observation contract remains version 1 with real IDs.

| Data | Invalid-result policy |
| --- | --- |
| JSON, required scene structure, entities, visible text | Reject the observation. |
| State changes and keyframes | Reject the invalid candidate, retain independent valid scene data. |
| Prospective matches and generic interrupt | Reject the invalid candidate; it cannot dispatch an alert. |
| Optional collection shape or size | Reject that collection; do not truncate it into a guessed subset. |

Every accepted result passes the existing final contract/reference validator.
Rejection logs contain a field, index, window ID, and application-owned category,
not the rejected model content. No inference repair/retry loop was added.

The opt-in reference_probe module compares both formats on the same private
images, alternating order. It uses the actual fast analyzer for the new path
and the existing legacy prompt/parser for the baseline. Reports keep per-trial
validation outcomes, image hashes, reference rejection counts and analysis time.
Transport and harness failures are separate and never count as successful model
responses. A small zero-error run does not establish a population error rate,
recognition quality, first-audible timing, or a glasses latency improvement.

## Stage 7: bounded quality and lifecycle evaluation

Current fast perception uses reference contract 3. Native prospective matches
copy one bound subject_id; the registry supplies its internal subject type.
Explicit legacy subject_type values still undergo strict identity validation.
Generic alerts must explicitly declare general scope while a watch is active.
Neither valid references nor declared scope establish semantic recognition.

Model-issued clear_visual_focus calls require the exact focus_id. An old stop
cannot clear a replacement or revoke that replacement's pending delivery.
Internal explicit-stop callers retain their existing unscoped cancellation path.

Use the existing eval extra (Pillow 11.3.0) for synthetic media and review helpers;
it is not a new production dependency. The native probes use real adapters and
isolated database owners. See the Stage 7 case, search, photo and review-format
documents in planning_docs/glasses_reliability for frozen rubrics and boundaries.
Run quality_probe, photo_probe, search_latency_probe and quality_review as Python
modules with their documented --help arguments. Fixture success, answer quality,
cleanup and physical playback are separate results; failed or unverified quality
cases must not disappear from a provider's denominator.
