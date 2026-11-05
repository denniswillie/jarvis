# jarvis.clip.v1 — bounded local clip template

Implemented source for the first M3 renderer fixture. **17 isolated tests, compilation and local synthetic MR01/MR02 rendering/inspection pass.** Evidence and the one corrected color-profile failure are in `jarvis-side-chat/implementation/MEDIA_LOCAL_QUALIFICATION.md`. This directory is not a production task worker, source/grant resolver, plugin installer, phone wire, editor UI or publication integration; real capture and real-sample acceptance remain unqualified.

## Fixed behavior

One retained source, 1–8 ordered cuts, 0–16 literal captions, 720×1280 at24fps, black letterboxing with `contain`. Captions use bundled Noto Sans600 with no remote font request. The template keeps the single inspected mixed audio stream, or mutes only when the recipe explicitly requests it. Multiple audio streams with `preserve` fail: this prevents quietly losing the wearer or Jarvis track. Supporting the real mixed recording remains a required Core qualification gate.

`recipe` is unchanged output of Python `ValidatedMediaRecipe.template_props()`. Cut frames use the **composition's24fps clock**, independent of encoded source fps. A cut `[48,168)` selects `[2,7)`seconds even if the input was encoded at30fps. `Sequence.from` places the cut on the output timeline; `OffthreadVideo.trimBefore` and `durationInFrames` select its source time. No `renderMedia.frameRange` is used, because that API uses inclusive bounds. Captions are half-open output-frame intervals. Integer-millisecond recipes only express exact24fps boundaries every125ms/three frames; this package does not round them or normalize variable-frame-rate media.

Each caption is a React text child. HTML, braces and URL-looking text are displayed literally. The first qualification should use the English marker captions in the saved fixtures. Other scripts/emoji and extreme multiline text need visual/font coverage review before promising their appearance; no visual qualification is claimed by the prop tests.

## Internal input envelope

```json
{
  "recipe": "replace this string with the unchanged template_props() object",
  "sources": [{
    "sourceId": "original",
    "assetPath": "inputs/<64 lowercase hex SHA-256>.mp4",
    "contentSha256": "<same SHA-256>",
    "durationMs": 30000,
    "audioStreams": 1
  }]
}
```

The `sources` array is **trusted staging input**, not a model-authored capability or alternative source registry. The owning adapter must first resolve the exact retained `file_id`/`version_id`/SHA-256, current access and allowed operation; inspect actual bytes; match the recipe's pinned template against installed inventory; and enforce the task's existing grants and limits. Neither a data object nor a matching hash establishes those facts. No mutable source generation is added to retained identity.

The harness accepts one hash-named MP4 beneath an explicitly supplied staging directory, rejects symlink redirection, hashes the original and the isolated copy, and serves only that copy using Remotion `staticFile()`. That function derives the URL on the local bundle server. Neither recipe nor mapping accepts external URLs, URL credentials, executable/filter text, custom templates or arbitrary filesystem paths. The CLI's executable/directory arguments belong to the trusted operator; do not expose this CLI directly as an unrestricted model tool.

Input caps are64KiB raw input JSON and300second/256MiB source; output remains720frames/30seconds. The five-minute input expansion is a new source candidate, not qualification of the earlier60second/64MiB image. Trusted Files/task/transport admission must independently authorize and inspect these bounds; the existing32MiB output publication limit is unchanged and remains enforced by its owner. Exact source permissions, stream metadata truth, VFR behavior and decoding remain upstream and actual-render checks, respectively. CPU, memory, renderer settings and process deadlines are unchanged; a longer input is not permission to extend an execution deadline.

## Pinned local installation and pure checks

Tested here with Node24.2.0 and npm11.3.0. Direct dependencies are exact: Remotion/core renderer/bundler4.0.539; React/react-dom19.3.1; Fontsource Noto Sans5.3.0. `package-lock.json` pins transitive resolution/integrity. `.npmrc` keeps the cache in this directory and disables install scripts, audit and funding requests. No CLI package, browser installer, runtime `npx` or downloaded media fixture is used.

```sh
cd /Users/denniswillie/jarvis/media/clip-template
npm ci --ignore-scripts --no-audit --no-fund
npm test
```

The tests use inert local bytes and never execute their fake browser/binary files. They cover actual React text escaping, Python-produced10second/12second props, timing/count/identity constraints, source hash/path failures, fresh output requirements and renderer options using injected inert functions. Test-created files are inside ignored `.qualification/` and are removed after each case. The compile-only check bundles the source/CSS/fonts into ignored `.qualification/compile-check/bundle`; it starts no browser and renders no frames.

Saved `fixtures/*-props-*.json` were generated from the previously accepted pure Python validator with synthetic source metadata. Their `aaaa…` hash is a placeholder with no corresponding media file. They are data fixtures, **not playable media or proof of installed-template identity**. Use actual inspected bytes/hash when preparing a real run.

## Coordinated render invocation

The owning Core lane must supply a compatible provisioned browser and the matching Remotion platform compositor directory (its `ffmpeg`, `ffprobe`, `remotion` executable and shared libraries). The presence of Homebrew FFmpeg alone does not establish that bundle. The harness requires all paths explicitly and throws if Remotion attempts a browser download. It uses `chromeMode: 'chrome-for-testing'` to select new headless mode for the provisioned full Chrome binary. Inspected Remotion source creates a new temporary profile; the qualification wrapper additionally confined TMPDIR and tracked its detached process groups. Never use a signed-in profile.

```sh
node scripts/render.mjs \
  --input /absolute/fixture/input.json \
  --staging-dir /absolute/fixture/staged \
  --browser /absolute/provisioned/chrome \
  --binaries-dir /absolute/provisioned/remotion-binaries \
  --work-dir /absolute/fixture/new-attempt \
  --output /absolute/fixture/new-attempt/output.mp4
```

The work directory must not exist; output must be its `output.mp4`. Existing input/output is never overwritten. The operation copies the selected source, bundles fixed code and local fonts, selects the fixed composition and requests H.264/AAC/yuv420p at24fps with explicit limited-range BT.709 conversion, PNG frame capture, software encoding, one rendering worker, CRF18 and192kbps/48kHz audio. Explicit BT.709 is required: the first real encode with Remotion4.x defaults produced full-range yuvj420p despite the yuv420p option. The corrected profile passed actual ffprobe/decode inspection. This is a deterministic configuration, **not a claim of bit-identical output across browser/platform/encoder builds**. Every different provisioned environment still needs qualification.

The API's30second timeout is per render frame, not an overall task deadline. The local qualification used a separate240second per-render subprocess deadline, fresh process groups and cleanup of only run-owned temporary Chrome profile groups. A production adapter must supply equivalent bounded cancellation through the existing task owner; this harness is not that owner. The reproduced run script and deadline/cleanup observations are preserved with the local evidence.

A failed attempt may leave work/partial output for diagnosis. It is never published or marked available. The returned local path means the renderer call returned; Core still needs decode/frame/audio inspection and playback before adopting a usable output. A retry must use the existing task/action reconciliation rules; do not create a new render simply because its response was lost. This local harness intentionally adds no durable task state or retry engine.

## Next real evidence and integration decisions

1. Core chooses and records the actual browser/platform/binaries/template bundle pin. This package's source version, lock hash and compile check are not the recipe's `bundle_sha256`. The eventual inventory must identify the immutable code/font bundle separately from per-job staged source media, and check it before running. No fake inventory entry is created here.
2. Use one permitted30second clip with a single mixed sound track, time markers and identifiable sound events. Run MR01 cuts2–7s and12–17s; then MR02 extends the second cut to19s. Inspect240/288 decoded frames, boundary markers, caption timing, preserved audio and unchanged original. Repeat explicit mute and a rejected missing-audio preserve input. A deterministic marker clip qualifies renderer behavior; the first real sample and live capture still have separate required evidence.
3. Parent/Core reuse existing task receipts, exact file adoption, cancellation/reconciliation and retained media delivery. Phone consumes the eventual exact output reference; this folder does not change its wire. Source bytes/version/hash, recipe predecessor/output refs, publication approval and account/caption/version receipts stay in their existing owning subsystems.

Actual capture/audio synchronization, retained large-file transport, wearable playback, phone playback/offline save, automatic highlight analysis, full-concert scale and Instagram publication remain unqualified. Clip editing only follows explicit chosen intervals; it does not decide which moments are highlights.

## Primary API evidence inspected

- [OffthreadVideo timing, audio and mute options](https://www.remotion.dev/docs/offthreadvideo)
- [Composition and source timing rules](https://www.remotion.dev/docs/timing)
- [Sequence placement](https://www.remotion.dev/docs/sequence)
- [renderMedia options, output and browser fallback](https://www.remotion.dev/docs/renderer/render-media)
- [Platform binaries directory](https://www.remotion.dev/docs/options/binaries-directory)
- [Bundler and matching exact Remotion versions](https://www.remotion.dev/docs/bundler)
- [Explicit color-space conversion and PNG capture](https://www.remotion.dev/docs/options/color-space)

Also inspected the locally installed4.0.539 `OffthreadVideoForRendering` implementation: source time divides the composed media frame by `videoConfig.fps`. The source agrees with the documented composition-clock interpretation. Source inspection and pure tests do not replace decoded-frame qualification.
