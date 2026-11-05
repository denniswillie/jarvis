# Shared-view evaluation

Run the harness with `python -m evals.perception_memory.new_architecture_runner --companion-quality --output /tmp/jarvis-companion-report.json`.

This suite uses a temporary additional fixture bundle and never rewrites the legacy manifest, images, or frozen baseline report. It exercises actual coordinator preparation and direct current-image retrieval with deterministic provider doubles. Fixture answers are not model-quality evidence. Deterministic reports record preparation latency separately and leave actual response latency unmeasured.

The twelve cases cover fresh labels, silent scene changes, explicit image retrieval, stale/missing vision, acknowledgements, a blank Python function, a partial loop, TypeScript, unreadable code, screen prompt injection, and IDE-to-slide transitions. Unit tests separately exercise session admission, image acknowledgements, cancellation, continuation, source replacement, and retrieval timing.

## Negative-answer checks

The stale-view, missing-camera, and acknowledgement cases have evaluation-only
`allowed_answer_patterns` and `max_answer_words` requirements. Patterns match the
whole reply after whitespace and apostrophe normalization, case-insensitively.
An empty answer, an invented scene, a disclaimer followed by an unsupported claim,
or an acknowledgement followed by capability advertising fails. These failures
also count against current-view accuracy and the live report's overall assessment.

These are conservative regression checks, not a semantic judge. A valid reply
outside the documented patterns fails for human review rather than silently
passing. Preserve its original trace and review the wording before adding an
appropriate evaluation variant. Do not tune production prompts to these patterns.
Manual live review still checks visual reasoning and the other scenario-specific
acceptance instructions; matching terms or patterns alone is not proof of quality.
Legacy cases leave these optional requirements unset.

## Direct-image acceptance suite, contract v1

The separate direct-image suite adds current, historical, and lifecycle trials
without changing frozen baseline reports. List cases without accessing providers:

```sh
python -m evals.perception_memory.direct_image_access_runner --list
```

Live trials require deliberate provider and database opt-ins, an explicit model,
and an output directory. For the configured model used in the September 2026
validation, an example independent-count/label run is:

```sh
python -m evals.perception_memory.direct_image_access_runner \
  --live --allow-database-fixtures --model gpt-realtime-2.1 \
  --mode current --cases L01,L02,L03,L04 --strategy autonomous \
  --repeats 3 --output /tmp/jarvis-direct-image-trials
```

Use `--mode historical` or `--mode lifecycle` for the other groups.
`--strategy forced` explicitly forces the listed tool sequence for a mechanics
trial; it is not evidence of autonomous tool choice. `--passive` permits a
comparison with already attached current pixels. Keep image bytes, questions,
models, and settings identical within each comparison, and preserve every repeat.

The driver opens the production audio and visual WebSocket routes in an isolated
local ASGI app. It uses the configured Supabase database and storage and actual
OpenAI Realtime audio, but does not run production background jobs or alter a
running Jarvis instance. Spoken questions are synthesized with `tts-1` by default
and sent as 24 kHz PCM without human pacing. Exact visual answers are present only
in pixels and the private evaluation answer key, not the model's question,
session title, memory summary, or tool metadata. Historical search uses real
retrieval without an embedding provider in this isolated harness.

Every attempt creates a unique fixture owner and an on-disk manifest before
storage writes. Cleanup runs on failures and partial setup, deletes only tracked
storage paths and exact fixture-owned database rows, and checks for leftovers.
The same owner cleanup guard is used by the historical repository integration
tests, including rows left behind by repository side effects. No schema reset,
broad time-range deletion, or real-owner cleanup is performed.

Each attempt has an immutable directory containing a redacted provider/client
trace, actual completed user transcription, completed answer, tool calls, image
hashes, ACK/output/cutoff relationships, storage timing, and cleanup result.
Provider errors and incomplete trials remain failures, not successful answers.
The runner stops on driver/infrastructure or cleanup failure so repeated broken
setup cannot create more fixtures.

Transport gates and answer correctness are separate. The runner leaves accuracy
unscored for human review against the recorded rubric. First audio of the final
response and answer completion are measured from the last input-audio send;
first substantive audible-answer latency remains unmeasured without listening
or alignment. Small-sample summaries use medians and ranges, not strong
percentile or speedup claims. Physical glasses optics, microphone behavior, and
device playback interruption still require the user-owned handoff.

Requested image resolution uses `VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS`
(five seconds by default), independently of the shorter passive-image sync
deadline. Storage reads remain cancellable and revalidate the original evidence.
An image-backed provider response failure produces a recoverable client error
and at most one tool-free spoken failure notice with no conversation/image input.
The live report preserves the original failed provider response even when that
notice and the next normal turn complete. Negative-case acceptance must not be
confused with a successful image-interpretation transport gate.

## Live validation

Export viewable synthetic screens with `python -m evals.perception_memory.companion_fixtures --output /tmp/jarvis-companion-fixtures`. Follow each scenario's `live_test_instructions`; keep exact screen contents out of spoken questions and textual context.

Capture a `LiveObservationBundle` using the existing trace format, then score it with `--companion-quality --live --live-results /path/to/trace.json`. Every scenario needs an actual OpenAI Realtime provider event, the real spoken answer, observed tool calls, evidence provenance, and end-of-question-to-first-useful-answer latency. The target is seven seconds, not the legacy suite's three-second gate. Background frame descriptions are not required for image-only assistance.

Synthetic trace bundles in unit tests test the scorer, not live model behavior. Passing deterministic tests does not establish that the model reads code correctly or that glasses optics make a real screen legible. Run the live cases before declaring Stage 2C's grounded-answer acceptance complete.

## R5 fixture semantics and live completion

The TypeScript screen declares `findIndex(values: number[], target: number)`.
It does not promise sorted input or request binary search. A linear scan,
`values.indexOf(target)`, or a correct `values.findIndex(...)` implementation can
be appropriate. The deterministic example uses `indexOf`; it is not a template
the production agent must follow. The evaluation checks the visible parameters
as a coarse signal. Manual live review must still assess language, return-index
semantics, and whether the response answers the requested next increment.

Before scoring a captured live answer, the driver must establish completion:

- Await provider acknowledgement of the specific new scene before asking.
- Correlate the connection/session, user turn generation, requests, responses,
  output items, and tool continuations; session identity alone is insufficient.
- Wait for completed final provider output and delivery of its required audio
  items to the client. Preambles, commentary, and function-call responses alone
  do not complete a trial. Account for intentionally suppressed commentary.
- Record the actual user transcription (`user.transcript.done` on the app or
  completed input transcription on the provider trace), not an assumed echo.
- Read only newline-terminated JSONL records while traces are still growing.
- Preserve pre-question proactive output separately from the answer.
- Keep driver failure, infrastructure failure, response completion, and manually
  accepted answer quality separate. Never overwrite an earlier failed attempt.
- Report first-audio latency separately from substantive-answer latency; without
  audio alignment or listening review, the latter is unmeasured.

The scorer's compact trace format and provider-event checks do not independently
prove all these delivery conditions. Keep the detailed raw traces as evidence;
do not turn a fabricated or incomplete bundle into a live-pass claim. Retain
accepted R4B style limitations rather than tuning prompts to pass these checks.
