# Runtime/memory reuse experiments

Current result: the corrected follow-up selected Hermes and source-backed
Hindsight for integration planning. See
[FINAL_DECISION.md](/Users/denniswillie/gitbio/jarvis/planning_docs/runtime_memory_reuse/FINAL_DECISION.md).
This is an opt-in evaluation suite, not a production migration.

Opt-in, synthetic-only evaluation. This directory is not imported by the live
application. See planning_docs/runtime_memory_reuse for scope and acceptance.

Use a private `/tmp/jarvis-runtime-reuse.*` root and separate dependency
environments. `environment.py` starts only named experiment services and records
their identities before allocation; `cleanup` removes only those exact resources.
It never reads production database credentials or modifies deployment settings.
Existing OpenAI credentials are loaded privately for synthetic native requests.

Pinned evaluation dependencies: Hermes 0.19.0 in its own environment; Hindsight
API slim/client 0.9.2 in another. Honcho is pinned by OCI digest in the controller.
Do not combine these dependencies with Jarvis's environment.

Reproduction prerequisites: a fresh private run root with reports/home/cache/config
directories and a resources.json manifest containing empty containers/processes/
volumes lists; separate pinned hermes-env and hindsight-env environments; Docker;
the existing authorized model key. Do not reuse an old resource manifest for a
new run. The controller does not install dependencies.

Current sequence, using the appropriate isolated interpreter:

```sh
B=/Users/denniswillie/gitbio/jarvis/backend/evals/runtime_memory_reuse
R=/tmp/jarvis-runtime-reuse.RUN
H="$R/hermes-env/bin/python"
M="$R/hindsight-env/bin/python"
"$H" -m unittest discover -s "$B" -p test_contracts.py -v
"$H" "$B/environment.py" --root "$R" start
"$H" "$B/hermes_probe.py" --root "$R"
"$M" "$B/comparison.py" --root "$R" --repeats 3
"$H" "$B/worker_acceptance.py" --root "$R"
"$H" "$B/sandbox_probe.py" --root "$R"
"$M" "$B/lifecycle.py" --root "$R"
"$M" "$B/source_grounding.py" --root "$R"
"$M" "$B/honcho_deep_probe.py" --root "$R"
"$H" "$B/environment.py" --root "$R" cleanup
```

Commands are illustrative; use the absolute script paths from any private
working directory. Native evidence, errors, service logs, and resource manifests
remain in the private root. Final sanitized results belong in the plan report.
Missing model usage is unknown, not free. An empty retrieval or a fixture tool
success is not a passing native memory-engine benchmark.

The sequence above is illustrative, not an unattended cleanup guarantee: always
run cleanup on interruption/failure too. Native cases incur provider usage.
Inspect per-case errors and receipts rather than treating exit zero as acceptance.
The high-reasoning and original-source controls are separate tracks, not extra
independent corpora. Do not ingest test questions or oracles.

memory_probe.py and http_memory_probe.py retain the first exploratory approach;
they are not the current acceptance entry points. jarvis_probe.py remains a
limited native baseline control requiring its own disposable schema/fixture setup.

## Historical first-run notes: superseded by the corrected follow-up

The defects and pending status below describe the initial investigation.
The current receipt, startup, sandbox, and lifecycle corrections were rerun
successfully; final findings are in FINAL_DECISION.md. Original failed evidence
is retained to avoid rewriting the history of the evaluation.

These files are exploratory drivers, not a passing reusable acceptance harness. See the authoritative results and limitations in `planning_docs/runtime_memory_reuse/EXECUTION_REPORT.md` from the repository root.

Known issues from the first execution:

- Hermes callbacks return plain dictionaries where the native registry requires JSON text. Callback execution does not establish model receipt. The correction is pending the user's decision; affected integration cases must be rerun afterward.
- The image case exposed a concurrency gate unnecessarily. Its elapsed time is not valid image/model latency evidence. Future runs must restrict tools per case and reset gate state.
- Hindsight's original OpenAI provider configuration caused internal tool failures despite successful outer HTTP responses. The successful diagnostic used `openai-responses`, `gpt-5.4-mini`, low reasoning, and RRF with learned reranking disabled.
- Honcho needs migration readiness before the deriver starts and an explicit small-corpus flush policy. The successful diagnostic used a flush-enabled replacement worker.
- The initial Honcho SDK driver did not persist its ingestion response into the report state. Later source and queue checks supplied separate ingestion evidence.
- Driver exit status and agent completion flags are not scenario acceptance. Inspect actual errors, tool receipts, source references, deletion completion, and effects.

The isolated services were cleaned up. Private evidence and dependency environments remain under `/tmp/jarvis-runtime-reuse.V4dY8h`; temporary-directory retention is not guaranteed. No production runtime, configuration, or data was changed. Do not use these drivers as live migration code.
