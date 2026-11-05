# Run the Meta version of Jarvis

This guide creates your own Jarvis setup. You bring the API accounts, database
and iPhone signing identity. It does not copy the developer's conversations,
recordings or credentials, and it does not start or stop another Jarvis instance.

## What you need

- Python 3.12 for the backend and Python 3.14 for Hermes, in separate environments.
- Docker, FFmpeg (`ffmpeg` and `ffprobe` on PATH), and internet access.
- Meta API access to `muse-spark-1.3` and `muse-wonder-1.0`.
- OpenAI access for GPT-Live and memory embeddings, plus a Gemini API key for vision.
- Your own Supabase project: PostgreSQL connection URL, project URL and service-role key.
- Private Storage buckets named `jarvis-visual-frames` and `jarvis-files`. Allow the
  file types you plan to use, including MP4, and a 500 MB upload limit for video.
- An HTTPS address forwarding to this backend, with WebSocket support. The phone
  and E2B background scripts must be able to reach it.

Keep service-role and model keys on the server. The phone uses only the backend
address and the bearer token generated below. Do not commit either environment
files or the private runtime directory.

## 1. Install the backend and Hermes

Run these commands from the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './backend[sandbox,setup]'
cp backend/.env.example backend/.env
chmod 600 backend/.env
```

Fill in `OPENAI_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` in `backend/.env`. Set `JARVIS_TIMEZONE` for yourself.
Add `JARVIS_FRONTIER_E2B_API_KEY_FILE=/absolute/path/to/your/e2b-key`.
Put your Meta API key and PostgreSQL URL in separate private files as well.
These are file paths in the commands below, not secrets pasted into shell arguments.

Install a fresh native Hermes runtime outside the repository:

```sh
NATIVE="$HOME/.local/share/jarvis/native-meta"
.venv/bin/python scripts/bootstrap_hermes.py \
  --destination "$NATIVE" --python python3.14
```

The installer downloads pinned public sources, applies the small static-prompt
compatibility patch, then uses Hermes's own `uv.lock` and dependency overrides.
Do not replace this with a plain `pip install hermes-agent`: the pinned release
has dependency constraints that its native installer resolves deliberately.
HTTPS downloads use verified certificates through `certifi`.

The destination must be new. `--download-only` stops after staging the sources;
`--install-only` resumes installation from an already-staged destination.
No model, sandbox, gateway or phone is started by this command.

## 2. Create your private configuration

Choose an empty database for a new setup. Do not point this command at somebody
else's running Jarvis database. Replace the example paths and HTTPS address:

```sh
CANDIDATE="$PWD/.jarvis-dev/meta-demo"
.venv/bin/python scripts/meta_setup.py prepare \
  --home "$CANDIDATE" \
  --base-env "$PWD/backend/.env" \
  --hermes-source "$NATIVE/source" \
  --hermes-python "$NATIVE/runtime/bin/python" \
  --hindsight-plugin "$NATIVE/hindsight-plugin" \
  --meta-key-file /absolute/path/to/meta-key \
  --database-url-file /absolute/path/to/database-url \
  --public-base-url https://your-own-backend.example
```

This creates a new owner ID, phone bearer token, Hermes profile, Files area ID,
memory-bank identity and private keys. It discovers and pins your local
`ffprobe` binary. The profile stores conversation history and sandbox bindings
when you use Jarvis; keep it between restarts.

The default backend port is **9100** and the memory-service port is **18889**.
Use `--port` and `--hindsight-port` if needed. Ports 9000 and 8888 are reserved
here to protect an existing demo. The destination must not already exist.

The database URL is deliberately not inherited from `backend/.env`. If you omit
`--database-url-file` or `--public-base-url`, edit the generated private
`backend.env` before continuing. The API refuses an unset database or placeholder
callback address.

Initialize the schema and your Files area:

```sh
.venv/bin/python scripts/meta_setup.py init-db --home "$CANDIDATE"
```

This uses the existing migration runner. It accepts an empty schema or one
already matching this release; it does not silently upgrade a partial or older
installation. Repeating it preserves the existing active Files area.

Recordings default to a 500,000,000-byte storage ceiling. If your Storage account
has a lower limit, set `SECOND_BRAIN_RECORDING_STORAGE_MAX_FILE_BYTES` accordingly
in the private `backend.env` before starting. After upgrading FFmpeg, refresh the
configured `ffprobe` path and SHA-256; a mismatched binary is rejected intentionally.

## 3. Start memory

Use the generated Compose project name for every command. It gives this setup
its own memory database, network and volume:

```sh
PROJECT=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["compose_project"])' "$CANDIDATE/setup.json")
docker compose --env-file "$CANDIDATE/compose.env" -p "$PROJECT" \
  -f deployment/docker-compose.knowledge.yml \
  -f deployment/docker-compose.knowledge-meta.yml up -d --build
```

Wait until the services report healthy in `docker compose ... ps`, then create
the memory bank and its three background summaries:

```sh
.venv/bin/python scripts/meta_setup.py init-memory --home "$CANDIDATE"
```

The summaries cover values/tradeoffs, habits, and active projects/routines. They
update asynchronously as memories arrive; creating them is not a claim that
Jarvis already knows you. No old conversations are imported. Model processing
uses your provider credits.

## 4. Start Jarvis, then Hermes

**Start the API first.** It creates the tool catalog Hermes needs.

```sh
.venv/bin/python scripts/meta_setup.py api --home "$CANDIDATE"
```

Once API startup finishes, open a second terminal in this repository, set
`CANDIDATE` to the same absolute directory, and run:

```sh
.venv/bin/python scripts/meta_setup.py gateway --home "$CANDIDATE"
```

The gateway command reports a clear error if the initial tool catalog is missing.
These commands run the existing servers in the foreground; there is no new
supervisor. Keep both terminals open. Forward your own HTTPS address to port 9100
(or the port you selected), including WebSocket traffic.

In the iPhone app, set that address and the generated
`JARVIS_STATIC_BEARER_TOKEN` from the private `backend.env`. See the root README
for Xcode signing and glasses pairing. Do not change an already-working phone's
address while recording a demo with another setup.

Start with a greeting, a current-view question, and a small request to Hermes.
Then check live guidance and a saved file. A healthy HTTP endpoint alone does
not prove audio, camera, cloud uploads or background callbacks work.

## What this configuration selects

| Part | Model / behavior |
| --- | --- |
| Router | Meta Spark 1.3, minimal reasoning, compact action-specific JSON |
| Hermes master and children | Meta Wonder 1.0, low reasoning, 40 tool turns |
| Hermes helper calls | Wonder for vision; Spark for compression, extraction and titles |
| Memory processing | Spark for extraction/consolidation; Wonder for reflection |
| Conversation | GPT-Live |
| Continuous visual analyzer | Gemini 3.1 Flash Lite |
| Memory embeddings | OpenAI `text-embedding-3-small` |

The router still uses the same backend validators and at most a lookup followed
by an action. It does not silently fall back to Luna. Hermes and Jarvis share one
Hindsight bank; automatic retention in the Hermes plugin is off to avoid storing
the same Voice dialogue twice. Memory reflection uses `tool_choice=auto` for
Meta compatibility. The old background memory workers stay disabled.

`SECOND_BRAIN_MEMORY_AUTHORITY=legacy` is retained as a compatibility setting for
the existing Files/recording path, not a claim that Hindsight has been replaced.
The new memory service and Hermes Hindsight tools are configured separately.

Music is disabled by default in this isolated profile. Spotify/Soloist requires
its own optional setup; do not connect a second profile to another demo's active
music process. E2B desktop creation does not guarantee a particular editor is
installed: Hermes may need to install/open the appropriate software.

## Stop or switch back

Stop the two foreground processes with Ctrl-C. To stop this memory service
without deleting its data:

```sh
docker compose --env-file "$CANDIDATE/compose.env" -p "$PROJECT" \
  -f deployment/docker-compose.knowledge.yml \
  -f deployment/docker-compose.knowledge-meta.yml stop
```

Do not use `down -v` if you want to retain memories. Preserve the private profile
and its keys to continue the same conversations. If you kept a separate working
GPT setup, switching back means using its original server/address/token; none of
these commands migrate or overwrite it.

## Qualification and limits

- The clean exported backend passes 194 focused checks: 94 Meta routing checks,
  46 existing-router checks and 54 shared-memory checks.
- Fresh installation from the pinned public native sources passed. Hermes is
  pinned to `f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab` (v0.22.6); the official
  Hindsight 1.2.3 plugin comes from `vectorize-io/hindsight` at
  `d952b6e655fccab4581ee0722b51c5f103873d12`.
- A clean PostgreSQL fixture accepted migrations 1-55 and passed schema readiness.
  The setup command also passed a fresh install and repeat run without duplicating
  the Files area.
- The current Spark interface passed six real-provider selections covering
  steer/queue/redirect, phone links, recording and music, in 1.3-2.6 seconds. An
  earlier run had four eight-second timeouts out of six; reliability under load
  is not established. These timings are not end-to-end audible latency.
- The fresh native Hindsight client 0.10.3 retained a synthetic fact and recalled
  its exact value from server 0.9.2. Earlier broader tests passed 17/19: deleting a
  source can leave a derived summary behind. Complete forgetting is unresolved.
- An earlier isolated GPT-Live -> Spark -> Wonder test created a file and spoke
  its saved result. It used browser WebRTC, disposable SQL and local object
  storage, not an iPhone or Supabase Storage.
- The exported iOS app built for the simulator with signing disabled. That is
  build evidence, not glasses, microphone, Bluetooth or phone acceptance.
- A fresh combined profile started the real API services and native gateway,
  exposed recording/app-control services, and completed a real Wonder turn.
  This used an isolated database and a deliberately unused storage endpoint;
  it did not test phone capture, cloud upload or external callback delivery.
  Physical-phone acceptance of this Meta configuration is still pending.

The root README and `benchmarks/HERMES_BENCHMARK.md` explain the demo/evaluation
scope. Raw credentials, transcripts, local profiles and test captures are not
part of the published repository.
