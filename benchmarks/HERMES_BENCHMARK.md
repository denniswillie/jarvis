# Native Hermes comparison: 4 November 2026

## Result

All three models passed both workflows twice. Wonder was faster on the measured
programmatic video-editing workflow. This is a small application benchmark, not
evidence that one model is generally more intelligent.

| Metric | Muse Wonder 1.0 | GPT-5.6 Sol | GPT-6 Astra |
| --- | ---: | ---: | ---: |
| Fully passing workflow runs | 4/4 | 4/4 | 4/4 |
| Editing task times, seconds | 63.08, 74.45 | 79.71, 84.45 | 73.17, 79.26 |
| Median editing task time, seconds | 68.77 | 82.08 | 76.21 |
| Child coordination times, seconds | 14.82, 15.58 | 14.77, 17.15 | 26.26, 19.94 |
| Median child coordination time, seconds | 15.20 | 15.96 | 23.10 |
| Reported nonzero/error tool results | 0 | 0 | 1 |

The Astra tool error did not prevent a correct final result. All six editing
sandboxes were confirmed paused during cleanup, not deleted.

## What was measured

- The same pinned native Hermes source, task prompts, toolsets and fixture setup.
- All models configured with low reasoning and a 40-turn limit. Matching effort
  labels do not imply identical internal computation or token budgets.
- Fresh isolated profiles, no personal memory or serving database access.
- Identical source-video SHA-256 in all six editing trials.
- Editing: trim an existing six-second synthetic video; produce a four-second
  portrait H.264/AAC draft; receive guidance during a running command; reorder
  source clips and remove audio for an interim export; receive a second live
  instruction; produce a double-speed final export while preserving earlier files
  and the running sandbox. Check dimensions, rate, codec, duration, sampled clip
  order, audio properties, full decode, and source/interim hashes.
- Children: delegate independent ledger and timeline calculations to exactly two
  retained children; combine their answers; send a correction to the same ledger
  child and use its retained history without restarting the other child.
- Task wall time includes model calls, tools, control polling and orchestration.
  It excludes initial sandbox preparation and final external inspection/cleanup.
- Each editing run includes two fixed 12-second pauses for deterministic
  mid-work guidance. Those 24 seconds are included in the table, not model latency.
- Sol and Wonder order was reversed for the second repetition. Astra ran afterward
  under the same protocol, so provider load/time-of-run remains a confounder.

## Scoring correction

One Wonder answer contained an acknowledgment JSON object followed by the correct
combined JSON object. The original scorer incorrectly parsed everything from the
first opening brace to the last closing brace as one object. After user approval,
the scorer was corrected to extract the last complete object, a regression test
was added, and the rule was applied consistently to all saved trials. No paid
model run was repeated to obtain a pass. Original results remain intact.

## Scope and interpretation

Wonder's median editing time was approximately 16% lower than the existing Sol
baseline and 10% lower than Astra in this sample. Child coordination was close to
Sol and faster than Astra. Final correctness tied; there is no demonstrated
quality advantage here. Six offline scorer checks passed.

This is programmatic editing of synthetic footage, not a full GUI video-editor,
cloud publication, phone streaming, glasses, audible-latency or human aesthetic
quality test. Two repetitions per workflow are insufficient for a robust ranking,
p95 estimate or general performance claim. No cost advantage is established;
tokenization, cache accounting and reasoning consumption differ by provider.
Wonder Full was not tested because this API account exposes only regular Wonder.

Suggested accurate presentation:

> In our small, controlled Jarvis benchmark, Muse Wonder completed the same
> verified, multi-step video-editing workflow about 16% faster than our existing
> GPT-5.6 Sol configuration, while handling live corrections and retaining interim
> work. All compared models passed the final task checks.

## Private reproducibility artifacts

- `.jarvis-dev/hermes-benchmark-01/`: original Sol/Wonder protocol and eight trials.
- `.jarvis-dev/hermes-benchmark-astra-01/`: matching Astra protocol and four trials.
- `.jarvis-dev/hermes-benchmark-comparison-01.json`: consistent corrected scoring
  and aggregate timings, with references to untouched original receipts.
- `hermes_benchmark.py`, `test_hermes_benchmark.py` and
  `rescore_hermes_benchmark.py`: runner, offline checks and transparent rescoring.

All work remained in the migration worktree. No main checkout, phone build,
serving profile or personal database was modified.
