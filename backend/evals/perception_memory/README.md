# Perception and memory baseline

This package freezes Jarvis's legacy perception and memory contracts before the
new architecture changes them. It provides a deterministic, provider-free
baseline and an opt-in live observation scorer.

## Deterministic run

```bash
cd backend
python -m evals.perception_memory.generate_fixtures
python -m evals.perception_memory.baseline_runner
```

The deterministic mode validates every referenced synthetic JPEG through the
production `validate_visual_frame_event` boundary and runs it through the real
`VisualAnalysisPipeline`, context builder, interrupt dedupe, and visual-memory
search service. Only provider boundaries and persistence are replaced by
deterministic fakes. It does not contact providers, write application tables,
or upload objects. Reports default to
`evals/perception_memory/reports/latest.json`.

The frozen pre-architecture baseline intentionally preserves two failures:

- `omitted_caption_detail`: the requested pixels were omitted by the per-frame
  description, so the audio agent cannot recover the exact value.
- `voice_only_memory`: session-only speech memory is lost across a simulated
  session restart.

Expiry, session restart, and interruption dedupe are state transitions executed
by the harness, not labels embedded in a prewritten observation.

## Live run

Live execution is deliberately opt-in. Collect observations through the actual
iOS, OpenAI Realtime, Gemini, storage, and tool paths. The input is a
`LiveObservationBundle` containing a `trace_id` and one entry per scenario. Each
entry contains the human observation plus captured trace events. Tool calls and
provider usage are derived from those events and cannot be asserted only by the
human observation.

Minimal trace-event example:

```json
{
  "trace_id": "device-run-2026-08-15",
  "scenarios": {
    "static_scene_current_view": {
      "observation": {
        "answer": "A notebook, mug, and blue pen are on the desk.",
        "grounded_fixture_ids": ["desk_static"],
        "response_latency_ms": 920,
        "frame_to_description_lag_ms": 510,
        "session_duration_seconds": 8
      },
      "events": [
        {"at_seconds": 0.5, "type": "provider.response.completed", "provider": "gemini", "model": "gemini-flash-lite"},
        {"at_seconds": 1.4, "type": "provider.response.completed", "provider": "openai_realtime", "model": "realtime"}
      ]
    }
  }
}
```

```bash
python -m evals.perception_memory.baseline_runner \
  --live \
  --live-results /tmp/jarvis-live-observations.json \
  --output /tmp/jarvis-live-report.json
```

A live report is rejected unless it includes OpenAI Realtime observations for
current-view reasoning, historical recall, and latest-frame capture. This keeps
provider tool selection visible instead of replacing it with a deterministic
mock.

## Metrics

The report records current-view, temporal, recall, and grounding accuracy;
temporal localization error; current-view response latency; frame-analysis lag;
visual backlog; visual-search candidate and returned-description counts;
memory-level accuracy; duplicate and false interruptions; tool calls;
per-provider calls per minute; and estimated provider cost.

## Frozen legacy contracts

- `POST /sessions/start` returns `session` and `realtime` objects.
- Audio uses `/realtime/audio/ws` with bearer auth plus `session_id` and
  `client_kind` query parameters.
- Visual ingestion uses `/realtime/visual/ws` with bearer auth plus `session_id`,
  `visual_session_id`, and optional `audio_session_id` query parameters.
- Frames are `visual.frame.capture` events with `source`, `content_type`,
  `frame_sequence`, `captured_at`, `image_base64`, and optional dimensions.
- Visual lifecycle events retain the existing `visual.session.created`,
  `visual.frame.accepted`, `visual.frame.dropped`,
  `visual.frame.analysis_skipped`, `visual.frame.analyzed`, and
  `visual.frame.analysis_failed` names.
- Interrupt lifecycle events retain `visual.interrupt.started`,
  `visual.interrupt.response_created`, and `visual.interrupt.done`.
- Reminder state remains persisted and triggerable by the reminder scheduler.
- `capture_latest_visual_frame` returns metadata and a source frame ID, never
  image bytes or a signed URL.
- `search_memory` preserves the scenario's temporal recall intent and failed
  tool-result envelope.
- The iOS primary control starts/stops the audio session and glasses vision as
  one user action; visual unavailability must not corrupt durable audio state.

The automated Swift contract test freezes the ordering of the unified start and
stop coordinator. Device-only DAT behavior remains a manual check: start Jarvis
once, confirm audio and glasses vision both become active, stop once, and
confirm both stop without corrupting the durable session.

Synthetic fixtures intentionally contain no personal images. Regeneration is an
explicit operation because hashes are part of the baseline contract.
