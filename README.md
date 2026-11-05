# Jarvis: Meta-powered wearable assistant

Jarvis is an assistant you can talk to through Meta Ray-Ban glasses. It can
look at what you are looking at, remember useful context, record a video, and
work on a computer for you. Its iPhone app lets you watch that computer and
guide the work as it happens.

For example, you can ask Jarvis to find a show, watch it browse, then ask it to
put the booking link on your phone. Or record a clip and ask it to edit the
video while you give feedback.

This is an experimental personal/hackathon project, not a production service.
Use your own accounts and credentials. Model calls, storage and running
sandboxes incur costs. Do not rely on visual interpretation for safety-critical
decisions.

## What uses Meta?

This version uses Meta models for routing requests, doing background work and
processing memory. It still uses OpenAI for conversation and Google for vision.

| Responsibility | Model or service |
| --- | --- |
| Glasses camera and wearable integration | Meta Ray-Ban glasses and Meta Wearables DAT |
| Decide which tool or background agent should handle a request | **Muse Spark 1.3** (`muse-spark-1.3`) |
| Research, computer use, file work and retained agent conversations | **Muse Wonder 1.0** (`muse-wonder-1.0`) through Hermes |
| Memory extraction and consolidation | **Muse Spark 1.3** through Hindsight |
| Memory reflection and synthesis | **Muse Wonder 1.0** through Hindsight |
| Live conversation and speech | OpenAI GPT-Live (`gpt-live-1`) |
| Live visual analysis | Google Gemini (`gemini-3.1-flash-lite`) |
| Memory embeddings | OpenAI embeddings |
| Agent computers | E2B desktop sandboxes |
| Application database and files | PostgreSQL / Supabase Storage |

Spark decides where a request should go. A simple action, such as stopping a
recording, goes straight to the relevant API. Longer work goes to Hermes,
which uses Wonder. Spark has its own compact response format rather than just
reusing the OpenAI configuration with a new model name.

## What you can do

- Talk through the glasses and ask about the current camera view.
- Ask an agent to research, browse, create files, or edit a recording.
- View its desktop on the phone and give follow-up guidance while it works.
- Start and stop recording by voice or in the app; view saved media in Audiovisual.
- Ask Jarvis to display a clickable link on the phone.
- Retain and retrieve personal context through shared Hindsight memory.
- Add optional server-side music playback and recording integration.

Desktop viewing shows applications that the agent actually opens. A terminal
command or headless edit does not automatically produce a graphical editing
timeline. Ask the agent to use a graphical editor when you want to watch it.

## How the pieces fit together

```text
Meta glasses <--> iPhone app <------------------> GPT-Live
                       |                            |
                  frames, controls             delegation
                       |                            |
                       +------ Jarvis FastAPI ------+
                                     |
                            Muse Spark router
                               /           \
                    Direct service APIs     Hermes / Muse Wonder
                    state, memory, app      retained master/children
                    controls, links                 |
                               |               E2B desktops
                               |               tools + files
                       Shared application services
                          /                  \
                  Live State               Hindsight
                  current context          long-term memory
                       ^                   Spark + Wonder
                       |
                  Gemini perception

Supabase: application records, captured media and published agent files
```

GPT-Live owns the speaking voice. Spark selects an operation; Hermes performs
longer work. Live State is a compact view of current information, not a second
memory database. Models retrieve domain state through tools rather than having
full state snapshots continuously injected into their instructions.

Follow-ups can steer ongoing work, queue a later request, or redirect the
current approach. Redirect is not an undo operation and does not guarantee that
an already-running external command stops instantly.

## What you will need

You need a Mac and Xcode to build the iPhone app. The easiest starting point
is to run the backend on that same Mac. You can host it elsewhere later;
the optional music-capture helper currently needs macOS.

| Requirement | Why it is needed |
| --- | --- |
| Python 3.12+ | Jarvis backend and setup tools |
| Python 3.14 | The separately pinned Hermes runtime used by this profile |
| Git, internet access and a working package installer | Fetch pinned upstream code and dependencies |
| Docker with Compose | Local Hindsight service |
| FFmpeg and ffprobe on the backend `PATH` | Recording inspection and video processing |
| Xcode and an iPhone | Build, sign and run the native app; app target is iOS 17.2+ |
| Supported Meta glasses, paired in Meta's app | Physical camera/audio use and DAT developer access |
| Meta API access to Spark and Wonder | Router, agents and memory model calls |
| OpenAI API access to GPT-Live and embeddings | Voice and memory embeddings |
| Gemini API key | Live visual analysis |
| Your own Supabase project | Application database and private media storage |
| E2B API key | Agent desktop sandboxes |
| An HTTPS origin reachable by the phone and E2B | API access and background-task callbacks |

Check that your provider accounts can use the models listed above before
starting. Having an API key does not necessarily give you access to every
model. If GPT-Live is unavailable to your account, this setup will not quietly
switch to a different voice service.

The app pins Meta DAT **1.0.1** and WebRTC **156.0.0**. Simulator tests do not
replace physical glasses testing.

## Get it running

Start here, then follow [the detailed setup guide](deployment/META_SETUP.md)
for the database and service commands. Both guides describe the same setup.
Some older files in `deployment/` describe an earlier worker system; you do not
need those recipes for this version.

### 1. Install the backend tools

From this repository's root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './backend[sandbox,setup]'
cp backend/.env.example backend/.env
chmod 600 backend/.env
```

Fill in `backend/.env` with your OpenAI and Gemini keys, Supabase connection
details, timezone, and the URL where your backend will be reachable. The setup
tool will also create an owner ID and a token for connecting your phone.
Keep the E2B and Meta keys in private files and use their full paths where
requested. Provider keys stay on the server, not in the phone app or Git.

Use a dedicated database and private Storage buckets. The standard bucket names
are `jarvis-visual-frames` and `jarvis-files`. Apply the versioned Jarvis migrations
using the setup guide before starting the application. Do not reset another
installation's database to make this one start.

### 2. Install pinned Hermes and its memory plugin

```sh
NATIVE="$HOME/.local/share/jarvis/native-meta"
.venv/bin/python scripts/bootstrap_hermes.py \
  --destination "$NATIVE" \
  --python python3.14
```

This uses Hermes **v0.22.6** and the catalog-pinned Hindsight **1.2.3** plugin.
It installs an independent runtime; it does not copy someone else's agent
history or require a privately provisioned profile. See the setup guide for
download/install options and any upstream availability limitations.

Upstream references: [Hermes](https://github.com/NousResearch/hermes-agent),
[Hindsight integration](https://hindsight.vectorize.io/sdks/integrations/hermes),
and [the official plugin catalog entry](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/hindsight.yaml).
The installer pins a revision rather than treating a changing catalog as a
reproducible release.

### 3. Create your own settings and history folder

```sh
NATIVE="$HOME/.local/share/jarvis/native-meta"
.venv/bin/python scripts/meta_setup.py prepare \
  --home "$PWD/.jarvis-dev/meta-demo" \
  --base-env "$PWD/backend/.env" \
  --hermes-source "$NATIVE/source" \
  --hermes-python "$NATIVE/runtime/bin/python" \
  --hindsight-plugin "$NATIVE/hindsight-plugin" \
  --meta-key-file "$HOME/.config/jarvis/meta-api-key" \
  --database-url-file "$HOME/.config/jarvis/database-url" \
  --public-base-url https://your-own-backend.example
```

Replace the key/database file paths and example URL with your own. This creates a fresh
profile: a private folder for your agent settings, history and credentials.
It does not start Jarvis yet. Follow the detailed guide to connect your
database, set the public callback URL, and start Hindsight with its own memory
bank. The callback URL is how an agent's sandbox contacts Jarvis when background
work finishes.

Jarvis saves conversation turns to memory. Hermes can read and write that same
memory through tools, but its automatic saving is off so conversations are not
stored twice. Keep this private folder and the memory bank between restarts.

### 4. Start the Meta services

After completing the database, Hindsight and callback steps, start the API
first from this repository's root:

```sh
.venv/bin/python scripts/meta_setup.py api --home "$PWD/.jarvis-dev/meta-demo"
```

Wait for API startup to finish, then start Hermes in a second terminal:

```sh
.venv/bin/python scripts/meta_setup.py gateway --home "$PWD/.jarvis-dev/meta-demo"
```

The isolated profile defaults to API port **9100** and Hindsight port **18889**.
It avoids the development GPT setup's ports 9000 and 8888. Use the generated
Compose project name and private configuration, not another installation's.

Expose the candidate API through your own HTTPS endpoint. For local development,
a tunnel can forward to port 9100. Configure the same origin in the generated
backend environment and on the phone. `localhost` on an iPhone is the iPhone,
not your Mac. Do not expose PostgreSQL or Hindsight directly to the internet.

### 5. Build and connect the iPhone app

1. Open `ios/Jarvis/Jarvis.xcodeproj` in Xcode and choose the `Jarvis` scheme.
2. Resolve its pinned Swift packages. Select your own signing team and unique bundle identifier; do not reuse the original developer's signing identity.
3. Connect and trust your iPhone, enable iOS Developer Mode if required, then build and run on that device.
4. Enter your Jarvis Meta backend's HTTPS origin and generated Jarvis bearer token in the app setup screen. This is the application token, not a Meta/OpenAI/Supabase key; do not enter `api.meta.ai` as the app's backend.
5. Pair the glasses in Meta's app, enable DAT developer access and complete Jarvis's registration/permission steps in Settings. Keep app registration and callback URL schemes consistent if you change them.
6. Select the GPT-Live voice route while stopped, then start Jarvis and test a greeting.

If you already have a working GPT installation, **do not switch its phone
configuration while recording a demo**. Keep that app/backend/tunnel running;
test the Meta candidate separately and change the phone origin only when ready.
Nothing in this repository requires deleting the GPT checkout or its data.

## Try it out

Test one capability at a time:

1. Greet Jarvis, stop, start again, and confirm both sessions reply through the glasses.
2. Look at a distinct object and ask what is in front of you. Check that the answer describes the current view.
3. Record a short clip, stop, and open the saved video in Audiovisual. An acknowledgment alone does not prove upload success.
4. Ask Hermes to create a small file. Confirm it is published, then open the workspace and request a visible browser action.
5. Guide ongoing work with a correction and ask for an interim file without closing the sandbox.
6. Ask for a link on the phone and open the popup.
7. Retain a harmless fact, recall it later, and test a correction. Review the forgetting limitation below before storing sensitive information.

The [setup guide](deployment/META_SETUP.md) lists the checks already run and
how to repeat them. You should still try the glasses yourself: a server health
check cannot tell you whether Bluetooth audio or the camera works on your device.

## Known limits and optional features

- **Memory deletion:** the tested Hindsight configuration can retain a derived summary after its source is deleted. Do not assume source deletion erases every derived memory.
- **Background camera:** opening another app or locking the phone is not qualified as uninterrupted camera streaming. A stale observation is not a current view.
- **Audio and recording:** the shared app includes recording echo reduction, but acoustic conditions and audio/video synchronization still need device testing. Do not change the Bluetooth route casually to improve one metric.
- **Recording size:** duration, bitrate and Supabase project/bucket limits all apply. A 30-minute duration limit is not a promise that every 30-minute clip fits the storage limit.
- **Music:** server-side Spotify Soloist and the macOS capture helper are optional and disabled by default in the isolated profile. They require separate provisioning, pairing and OS permissions. Their installed binaries and account credentials are not bundled. Use audio you are permitted to record or redistribute.
- **Desktop applications:** the E2B desktop provides a computer, not a guarantee that every editor is preinstalled. Agents may need to install appropriate software.
- **Notifications:** local app behavior does not establish APNs push support; remote push needs separate Apple provisioning.

The application sends relevant speech, images, files and memory content to the
configured providers. Review that data flow before using private material.
Sandbox agents can execute commands and access the internet. This single-owner
development setup is not a production multi-tenant security boundary.

## Repository map

| Path | Purpose |
| --- | --- |
| `ios/Jarvis/` | Native app, glasses, audio, recording and workspace UI |
| `backend/app/realtime/` | Voice sessions and routing, including Meta adapter |
| `backend/app/hermes_sessions/` | Retained agent conversations, native controls and sandbox integration |
| `backend/app/perception/` | Visual and conversational observation processing |
| `backend/app/live_state/` | Current context and service-facing operations |
| `backend/native/` | Native helper source |
| `deployment/META_SETUP.md` | Detailed installation, launch and qualification guide |
| `scripts/bootstrap_hermes.py` | Pinned native runtime/plugin installation |
| `scripts/meta_setup.py` | Private profile preparation and service launch |

Keep `.env`, `.jarvis-dev`, credentials, personal recordings and runtime databases
out of Git. Preserve your private state when stopping services; deleting Docker
volumes or sandbox resources is not an ordinary restart procedure. Third-party
components retain their upstream licenses and notices.
