# Jarvis read-only Mac project connector

Status: S10 implementation and native/contract acceptance complete. Outbound scoped
transport, canonical sources and shared worker/foreground tools passed real Mac
Python/TypeScript, Hermes coding and lifecycle checks. This is not live deployment;
no personal root has been enrolled. See planning_docs/second_brain/reports/S10_REPORT.md.

This package provides explicit local root enrollment and bounded file/status/search,
symbol, definition and reference operations. It does not provide a shell, edits,
dependency installation inside the inspected repository, account access or model keys.
It is independent of the backend's model and database dependencies.

## Install and run locally

Use a dedicated Python 3.12+ environment:

```sh
python3 -m venv "$HOME/.venvs/jarvis-project-connector"
"$HOME/.venvs/jarvis-project-connector/bin/pip" install ./mac_connector
install -d -m 700 "$HOME/.config/jarvis"
"$HOME/.venvs/jarvis-project-connector/bin/jarvis-project-connector" \
  --registry "$HOME/.config/jarvis/projects.json" enroll \
  --root /absolute/path/to/an/approved/repository
```

Enrollment creates a new mode-0600 registry without overwriting an existing file.
Multiple --root arguments enroll multiple independent roots. The displayed root_id
is the local inspection reference, not a remotely supplied host path. Backend
registration below returns separate server root IDs for Jarvis's shared tools.
The local registry contains the absolute path and pinned filesystem identity.
Remove the registry or the root's entry to revoke local inspection. The CLI rereads
enrollment before releasing an answer. A running connection propagates removed
entries to the backend on its next successful poll. Deleting the entire registry
stops local access; it does not guarantee that the backend received revocation.

```sh
jarvis-project-connector --registry "$HOME/.config/jarvis/projects.json" inspect \
  --root-id ROOT_ID --request '{"operation":"symbols","language":"python","query":"Controller"}'
```

Jedi 0.19.2 is an installed dependency. TypeScript/JavaScript navigation additionally
requires an operator-installed Node executable and TypeScript library. The tested
TypeScript version is 5.9.3. Install it outside source roots without lifecycle scripts:

```sh
npm install --prefix "$HOME/.local/share/jarvis-typescript" \
  --ignore-scripts --no-audit --no-fund --save-exact typescript@5.9.3
jarvis-project-connector --registry "$HOME/.config/jarvis/projects.json" inspect \
  --root-id ROOT_ID --node /absolute/path/to/node \
  --typescript "$HOME/.local/share/jarvis-typescript/node_modules/typescript/lib/typescript.js" \
  --request '{"operation":"definition","path":"src/main.ts","line":12,"column":7}'
```

Only operator configuration selects executables. A read request cannot provide a
command, executable, absolute file path or arbitrary extra input.

## Connect to the backend

The backend needs reviewed migration 0034 and its source-backed knowledge runtime.
Use an existing owned knowledge-area ID. No automatic production migration or live
memory switch is performed by this CLI.

Provide the owner's existing API bearer credential in the environment variable
JARVIS_PROJECT_BOOTSTRAP_TOKEN, not a command-line argument or a source file. It is
used only for enrollment and is not saved in the connector credential file.

```sh
jarvis-project-connector --registry "$HOME/.config/jarvis/projects.json" register \
  --url https://your-jarvis-backend.example --area-id OWNED_AREA_UUID \
  --credentials "$HOME/.config/jarvis/project-connection.json"
jarvis-project-connector --registry "$HOME/.config/jarvis/projects.json" serve \
  --credentials "$HOME/.config/jarvis/project-connection.json"
```

For TypeScript navigation, pass the same --node and --typescript operator paths to
serve as to inspect. The connection is outbound only. HTTPS is required except
explicit loopback fixture URLs; redirects and environment proxies are not followed.
Registration reserves a new mode-0600 credential file without overwriting another
file, and displays IDs rather than the token. Scoped credentials expire after
30 days by default and cannot use owner administration endpoints.

The owner can revoke an entire connector with DELETE /projects/connectors/ID using
the ordinary owner API bearer token. Revoke old enrollment before replacing its
credential. Revocation invalidates retained project originals and dependent results,
not merely future polls. Stop the CLI to go offline without revoking authorized
historical captures. If registration reports cleanup_required, revoke the reported
connector ID before enrolling again.

Model tools are list_project_roots, inspect_project and get_project_result. They
cannot enroll roots, alter permissions, install software or run a host shell.
Use the server root_id from listing, not a root ID copied from the local CLI.
Pending inspection is recoverable by request_id; it is not proof of a failed read.

Idle polling incrementally refreshes previously read files only. It does not upload
every file in an enrolled repository. A minimum ten-second per-file refresh interval
is not a maximum freshness guarantee; busy or disconnected connectors delay it.
Use a direct inspection when current disk evidence matters.

## Evidence and coverage

Every returned file or symbol has a root identity, relative case-exact path and file
content digest. Project queries also include a snapshot identity and capture time.
Git identity includes branch, commit and dirty state where available. Dirty files are
read from disk, not replaced with HEAD. Unsaved editor buffers are unavailable.

Read paging uses Unicode character offsets and requires the original digest after
the first page. Project continuation can fence against expected_snapshot. A changed
source/root fails rather than silently combining snapshots.

Local bounds default to 256 KiB per UTF-8 file, 4 MiB per snapshot, 1,000 files, 20,000
directory entries, depth 24 and at most 200 returned items. Results disclose exclusions
and truncation. Missing/offline/replaced roots, excluded sources, unavailable language
runtimes and invalid positions are unavailable/error results, never successful empty
repositories. Capture is revalidated before release but is not an atomic filesystem
transaction or proof of the current state after delivery.

Network requests allow at most 30 returned items and 32,000 characters per read
page. Full originals travel separately and must fit canonical storage bounds;
large or heavily escaped sources can be unavailable even within local byte limits.
A page is never published as a complete original. Canonical source digests differ
from raw file digests. Sources and inspection receipts use the existing deletion
and dependency system; Hindsight indexing is asynchronous, not required for a read.

Jedi's maintained project API and TypeScript's maintained language service supply
semantic navigation, not regular-expression symbol matching:
- https://jedi.readthedocs.io/en/v0.19.2/docs/api.html
- https://github.com/microsoft/TypeScript/wiki/Using-the-Language-Service-API

Static reference lists are explicitly incomplete for dynamic behavior. Python analysis
uses only copied permitted snapshot files and a fixed interpreter. TypeScript uses
an in-memory file host, without repository plugins, external dependency sources,
tsconfig path aliases, or standard-library type sources. These omissions are disclosed;
this is not a claim of complete build-configuration-aware project analysis.

## Safety boundary

Descriptor-relative, no-follow file reads stay beneath the pinned enrolled root.
Symlinks, hardlinks, special files, binaries, hidden files, common credential/key
names and generated/dependency directories are excluded. Case aliases are not accepted
even on case-insensitive filesystems. These filename rules are not a universal secret
scanner: the operator must approve the remaining source files for sharing.

Language children receive a minimal environment and parent-owned temporary workspace,
not ambient credentials. Jedi's persistent filesystem cache is disabled. Timeout
terminates the owned process group and removes temporary source copies. Source code is
parsed, not imported/executed. Tests prove source bytes and modification times remain
unchanged, and changes during analysis invalidate the pending result.

## Tests

```sh
JARVIS_CONNECTOR_TEST_NODE=/absolute/path/to/node \
JARVIS_CONNECTOR_TEST_TYPESCRIPT=/absolute/path/to/typescript/lib/typescript.js \
python -m pytest mac_connector/tests -q --tb=short
```

TypeScript tests skip without explicit runtime paths; a skip is not language acceptance.
Fixtures create temporary synthetic Python/TypeScript repositories only. No personal
repository or remote account needs enrollment for these tests.
