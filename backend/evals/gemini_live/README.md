# Gemini Live migration acceptance

Opt-in, paid provider evaluation. This is NOT an enabled production audio provider.
Keep GPT as Jarvis's default until the acceptance blockers and physical tests pass.

## Scope

- Original JPEGs are sent unchanged; hashes are included in reports.
- Initial-history images and live video frames are separate test conditions.
- Native audio is saved as 24 kHz mono PCM WAV. A transcript is not proof that
  equivalent speech was generated. Audit the WAV as well as the transcript.
- Generated speech is streamed at real-time pace as 16 kHz mono PCM16.
- Timing is last input audio byte to first returned audio byte, NOT physical
  end-of-speech to speaker playback. Setup time is excluded.
- Automatic server VAD is measured separately from manual audio turn boundaries.
- Text prompts use realtime text input without manual audio activity markers.
- Barge-in checks server interruption and the subsequent response, not iOS's
  playback queue, acoustic echo cancellation, or glasses routing.
- Reconnection uses a real server-issued resumption handle. It does not prove
  all-day stability, expiry recovery, or exactly-once replay across a crash.
- Memory probes use real Jarvis tool schemas, handlers and JarvisToolExecutor,
  backed by an in-memory fixture. No Supabase reads or writes are performed.
- The camera tool is a fixture handler testing provider image/tool continuation,
  NOT production frame admission, historical evidence, or provider ACK tracking.
- Tool-call IDs are retained, repeated IDs are not executed twice, and cancelled
  responses are suppressed without cancelling an already executed mutation.
  A NEW provider ID is a new invocation: this does not solve operation-level
  deduplication when the model retries a completed action after interruption.

## Run using the configured backend keys

From `backend`, supply directories containing the two exported JPEGs and the
speech fixtures, plus a NEW output directory:

```sh
bash scripts/run_gemini_live_eval.sh \
  /Users/denniswillie/Downloads/jarvis-playground-images.KA4Xlq \
  /Users/denniswillie/Downloads/jarvis-gemini-live-evaluation/speech \
  /Users/denniswillie/Downloads/jarvis-gemini-live-evaluation/new-run \
  all 3
```

The wrapper runs a disposable container with read-only source mounts and streams
keys from the existing backend through stdin. It does not rebuild/restart the
backend, persist credentials, or change its environment. Override
`JARVIS_BACKEND_CONTAINER` if its name differs.

Available groups: `smoke`, `vision`, `audio`, `interaction`, `tools`,
`extended-tools`, `baseline`, or `all`. Repeats are bounded to 1-5.
Individual scenarios have timeouts and do not automatically retry paid calls.
`execution: completed` means the scenario ran, NOT that it passed acceptance.
Review semantic accuracy, full audio delivery, and side effects independently.

The Python runner additionally accepts `--thinking medium` and
`--media-resolution MEDIA_RESOLUTION_HIGH`. These settings are recorded, not
silently substituted for default-model results.

## Speech fixtures

Required WAVs: `code`, `simple`, `long`, `interrupt` at 16 kHz mono PCM16, plus
`simple-24k` at 24 kHz for the OpenAI baseline. The September 11 fixtures were
generated with macOS `say -v Samantha -r 175`, then converted with ffmpeg.

| Fixture | Spoken input |
| --- | --- |
| code | Look at the code on the screen. Is the last line correct? Explain any mistakes you can actually see, and tell me the correction. Please answer out loud, briefly. |
| simple | What is two plus three? Answer in one short sentence. |
| long | Explain how binary search works, slowly and in detail, for about thirty seconds. |
| interrupt | Stop. Change of plan. What is seven plus two? Answer only that question. |

Expected answers are not injected into model context. Existing synthetic scene
fixtures are clearly distinguished from real glasses frames.

## Remaining production preparation

Do not emulate OpenAI image acknowledgements with local successful sends.
Jarvis's current response cutoffs, image deletion/pinning, playback truncation,
output phases, context refresh and durable tool lifecycle require explicit
Gemini-compatible contracts. The native acceptance client is not a drop-in
replacement for `OpenAIRealtimeConnection`.

Before enabling an opt-in production route, address partial native audio,
inconsistent live-frame answers, and operation replay after interruption. Then
exercise the actual registry, database, current/historical image tools, reminders,
context refresh and reconnect behavior end to end under isolated fixture owners.

## Physical acceptance checklist

Once an opt-in route is implemented and the automated blockers are resolved:

1. Ask five ordinary questions and five visual questions in a quiet room.
2. Pause naturally mid-sentence, then finish; check that Jarvis does not rush in.
3. Interrupt a long answer; check immediate silence and correct new answer.
4. Interrupt while a memory/tool operation is in flight; inspect for lost or
   duplicate effects, not just whether the speech stopped.
5. Capture a new view and request historical evidence; verify correct timestamps
   and no confusion between the two images.
6. Change a preference, start another session, and verify the updated memory.
7. Disconnect/reconnect network and audio routing; verify recovery without stale
   playback, duplicated answers, or lost task status.
8. Run an extended session and inspect reconnects, context compression, token
   usage, battery impact and first audible response latency.

Do not call these tests passed based on synthetic speech or provider events alone.

## Production-adapter paired probes

The existing launcher also accepts an explicit `--production-adapter` mode:

```sh
bash scripts/run_gemini_live_eval.sh --production-adapter \
  /Users/denniswillie/Downloads/jarvis-playground-images.KA4Xlq \
  /path/to/24k-speech-fixtures \
  /path/to/new-output-directory \
  all 2
```

Unlike the original provider-only runner, this mode uses the configured database
with tracked disposable owners and independent cleanup. It never uses the real
owner, starts the production runtime, deploys code, or changes provider enablement.
Credentials still travel through stdin. Its supported groups are `audio`, `vision`,
and `all`; repeats remain bounded to 1-5. The original mode keeps its existing
provider-only behavior and does not receive a database credential.

Supply mono PCM16 `simple-24k.wav` and `code-24k.wav` at 24 kHz. The latter is the
same spoken code question listed above, resampled without changing its words.
The two original JPEG filenames expected by the launcher remain unchanged.

`paired_probe.py` constructs the actual GPT and Gemini session coordinators with
real current-image tools, session storage, and experience attachment. Only the
phone sink, camera source, and pass-through transport observation are fixtures.
The GPT implementation, prompt builder, and output policy are unchanged.

Reports distinguish `transport_passed` from answer-quality acceptance. They retain
native and app WAVs, completion evidence, image/input hashes, timing, and cleanup.
GPT intentionally suppresses established commentary-phase audio; the independent
audit reuses its existing phase policy to identify eligible audio, retaining both
the raw and eligible recordings. Missing final-answer audio still fails. Gemini's
ordinary native turn audio is all eligible. An app terminal event alone is not
proof of native speech completion.

Latency uses one runner monotonic clock, from the end of streamed input to the
first app-sink audio. Raw first-provider audio and first eligible-provider audio
are separate observations, so muted commentary is not mistaken for network delay.
This does not measure device VAD, Bluetooth acoustics, or physical playback.
Slow failures and missing measurements remain visible in summaries.

`domain_speech_probe.py` supplies the production Gemini durable-interruption
matrix; `gpt_domain_speech_probe.py` supplies a matched unchanged-GPT control using
the same speech files, real handlers, and four observation boundaries. The GPT
control does not receive Gemini's local operation-status tool. Its durable-count
result is deliberately separate from independent spoken-status review.

All these probes are opt-in evaluation tools, not automatically run by ordinary
pytest or app startup. Keep manifests and reports when investigating a failure;
never overwrite an earlier result to make a later run appear uniformly passing.

## Stage 6 real-router and rollback probe

`evals.gemini_live.routing_probe` exercises FastAPI's actual ASGI WebSocket route,
real authentication and owner-scoped session lookup, both production provider
sessions, and real memory handlers against disposable database owners. It does
not start application lifespan/background queue consumers. The phone transport
is simulated; one named case deliberately fails Gemini's connection. Subsequent
GPT connections are real, not constructor mocks.

Provide credentials on stdin using the same existing-backend-to-disposable-
container pipeline as the other paid probes. The JSON contains `gemini`, `openai`,
`database`, and a `settings` object preserving the backend's public
`OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_VOICE`,
`OPENAI_REALTIME_TRANSCRIPTION_MODEL`, `JARVIS_AUDIO_SAMPLE_RATE`,
`JARVIS_AUDIO_FORMAT`, and, when present, `GEMINI_LIVE_MODEL` and
`GEMINI_LIVE_VOICE`. Never write this credential JSON to disk or put it in argv.

Within that disposable container:

```bash
python -B -m evals.gemini_live.routing_probe \
  --speech-dir /speech --output /results/NEW_RUN_DIRECTORY
```

The speech directory contains 24 kHz mono PCM16 `simple-24k.wav`, `save.wav`, and
`retrieve.wav`. The first two reuse the paired/interruption fixtures. The retrieval
question is: "Please explain the saved memory identified in my session
instructions. Use the memory tool to check the actual stored record." Only the
saved record's identifier, not its contents or a synthetic retrieval response,
is supplied in the test-owned session instructions. GPT must retrieve it through
the real `explain_memory` handler. Independent readback must show the same record
and no extra mutation.

An optional `--soak-seconds 660` adds a bounded natural-turnover observation with
periodic non-mutating speech. Allowed windows are 600 to 900 seconds. No GoAway or
socket close is injected into that case; it requires observed provider GoAway,
same-provider resumption, and completed speech afterwards. An inconclusive window
is not counted as a recovery pass. This is not an all-day/device battery test.

The report distinguishes routing failures, model/tool-input failures, actual
memory retrieval, and independently confirmed cleanup. A manifest is written
before fixture insertion. Resource-close failure cannot skip database cleanup;
cleanup failure makes the run fail and preserves the manifest for recovery.
