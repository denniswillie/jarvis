# Dedicated frontier worker

**Historical architecture, not the current Meta setup.** The following describes
the retired 0.19 TaskRunner path. Use [the retained Meta setup](../META_SETUP.md)
for Hermes v0.22.6. Do not enable both worker systems.

This opt-in role runs Jarvis's durable TaskRunner and a separately pinned Hermes 0.19.0 interpreter. It does not start the API, audio sessions, legacy memory processing or a second generic agent loop.

## Boundaries

The API owns task admission/status/context capture. It advertises only configured model metadata and checks an owner/model-scoped expiring worker lease before new intake or resume. Replays of already accepted requests remain readable after worker loss. A lease is not a promise of completion.

The worker owns execution, model/tool gateways and private native state. Hermes receives no database or upstream provider credential. One worker process owns each private state root. Keep SECOND_BRAIN_WORKER_HOST_ID stable for that volume; never reuse it for a different host/storage pair. Recovery for an existing native attempt is restricted to its recorded host. A lost state volume remains an explicit recovery problem, not permission to repeat effects.

## Native command

Use the normal Jarvis environment for python -m app.tasks.worker run or check, with a separate pinned interpreter selected by SECOND_BRAIN_HERMES_PYTHON. Set SECOND_BRAIN_ENABLED, SECOND_BRAIN_TASKS_ENABLED and SECOND_BRAIN_FRONTIER_ENABLED to true only for the roles you are enabling. Configure SECOND_BRAIN_FRONTIER_MODEL, SECOND_BRAIN_WORKER_HOST_ID and an absolute private SECOND_BRAIN_HERMES_WORK_ROOT.

The worker checks schema readiness, native interfaces and configured model metadata access before advertising. The check command reads its own worker registration; it does not run inference. Credentials may use existing environment settings or private mounted files through JARVIS_FRONTIER_DB_URL_FILE and JARVIS_FRONTIER_OPENAI_API_KEY_FILE. Never print or commit those files.

## Container role

deployment/docker-compose.frontier.yml is an explicit standalone worker definition. It uses two virtual environments so Hermes dependency pins cannot alter voice. Set JARVIS_FRONTIER_SECRET_DIR to a private absolute directory containing database-url and openai-api-key, and provide JARVIS_FRONTIER_OWNER_ID and JARVIS_FRONTIER_HOST_ID.

Apply the reviewed migrations explicitly before enabling either role. Match the API's configured frontier model to the worker model. Do not change the API's foreground audio provider. Start only this named service, not unrelated development services. No ports are exposed.

The initial worker compose configuration deliberately leaves selected knowledge disabled. The optional docker-compose.frontier-knowledge.yml overlay adds the reviewed internal Hindsight network, exact private-origin opt-in and mounted API key. JARVIS_KNOWLEDGE_ENABLED still defaults to false; enabling requires the actual owned JARVIS_KNOWLEDGE_AREA_ID and accepted role/source configuration. Do not configure a conflicting inline Hindsight key. Candidate-container canonical-source and task integration passed. The actual owner's persistent worker is now processing the imported corpus; live API authority handoff and full voice/TTS acceptance remain open. See ../../planning_docs/second_brain/reports/M03_PERSONAL_IMPORT_REPORT.md and ../../planning_docs/second_brain/reports/M04_SELECTED_CONTAINER_ROLES_REPORT.md. Do not treat a healthy worker as complete native source coverage or a complete second-brain migration.

For retained images, optionally combine the base compose file with deployment/docker-compose.frontier-images.yml. Provide JARVIS_FRONTIER_SUPABASE_URL, an explicit JARVIS_FRONTIER_FRAME_BUCKET matching the existing frame bucket, and a private supabase-storage-key file beside the other secrets. The parent worker alone receives this credential. Native CLI use can supply existing Supabase settings or JARVIS_FRONTIER_STORAGE_KEY_FILE.

The worker resolves only exact images referenced by its canonical task context. Earlier current-camera selections are explicitly presented as captured snapshots, not proof of a currently live camera or a new capture. Expired/unavailable frames remain unavailable; no newer frame or scene description is substituted. Legacy record-associated images keep their existing record/evidence authorization checks.

## Shutdown and rollback

Disable new API frontier intake, then stop the worker with its configured grace period. It withdraws readiness before draining/stopping owned tasks. Preserve the state volume, task/action ledger, pending results and operation receipts. A restarted worker on the same identity/volume reconciles existing attempts instead of creating replacement effects. Do not use compose down -v as routine rollback.

Worker-boundary acceptance, including native compression/source recovery and the current stored-image container, is recorded in planning_docs/second_brain/reports/S04_REPORT.md. This does not enable foreground delegation or constitute live cutover. S05 owns the foreground integration; M04 remains the memory cutover stage.
