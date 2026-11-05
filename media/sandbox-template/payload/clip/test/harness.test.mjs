import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, writeFile, readFile, rm, symlink, stat, truncate, rename} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {preflight, renderFixture, refuseBrowserDownload} from '../scripts/harness.mjs';
import {FIXTURE_LIMITS} from '../src/props.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
async function setup(t) {
  // Own test files only; fake bytes and fake executables are NEVER executed.
  const fixtureRoot = path.join(root, '.qualification');
  await mkdir(fixtureRoot, {recursive: true});
  const dir = await mkdtemp(path.join(fixtureRoot, 'pure-'));
  t.after(() => rm(dir, {recursive: true, force: true}));
  const source = Buffer.from('INERT TEST BYTES, NOT VIDEO');
  const sha = createHash('sha256').update(source).digest('hex');
  const input = JSON.parse(await readFile(new URL('../fixtures/mr01-props-10s.json', import.meta.url), 'utf8'));
  input.sources[0].contentSha256 = sha;
  input.sources[0].assetPath = `inputs/${sha}.mp4`;
  const options = {
    inputFile: path.join(dir, 'input.json'), stagingDir: path.join(dir, 'staging'),
    browserExecutable: path.join(dir, 'browser'), binariesDirectory: path.join(dir, 'binaries'),
    workDir: path.join(dir, 'attempt'), outputPath: path.join(dir, 'attempt/output.mp4'),
  };
  await mkdir(path.join(options.stagingDir, 'inputs'), {recursive: true});
  await mkdir(options.binariesDirectory);
  for (const file of [options.browserExecutable, ...['ffmpeg', 'ffprobe', 'remotion'].map((name) => path.join(options.binariesDirectory, name))]) {
    await writeFile(file, 'INERT NEVER EXECUTE', {mode: 0o700});
  }
  const sourcePath = path.join(options.stagingDir, input.sources[0].assetPath);
  await writeFile(sourcePath, source);
  await writeFile(options.inputFile, JSON.stringify(input));
  return {dir, options, input, source, sourcePath};
}

test('preflight is read-only; accepts exact local bytes without claiming media decode', async (t) => {
  const s = await setup(t);
  const checked = await preflight(s.options);
  assert.equal(checked.sourcePath, s.sourcePath);
  assert.deepEqual(checked.inputProps, s.input);
  await assert.rejects(stat(s.options.workDir), {code: 'ENOENT'});
});

test('preflight accepts exact 256MiB bytes and rejects one extra byte before renderer loading', async (t) => {
  const s = await setup(t);
  assert.equal(FIXTURE_LIMITS.maxSourceBytes, 256 * 1024 * 1024);
  // A sparse inert file tests the real stat/streaming-hash path with bounded RAM.
  // It is not an encoded five-minute video and no decoder/browser is invoked.
  await truncate(s.sourcePath, 256 * 1024 * 1024);
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(s.sourcePath)) hash.update(chunk);
  const sha = hash.digest('hex');
  const exactPath = path.join(s.options.stagingDir, 'inputs', `${sha}.mp4`);
  await rename(s.sourcePath, exactPath);
  s.input.sources[0].contentSha256 = sha;
  s.input.sources[0].assetPath = `inputs/${sha}.mp4`;
  s.input.sources[0].durationMs = 300_000;
  await writeFile(s.options.inputFile, JSON.stringify(s.input));
  const checked = await preflight(s.options);
  assert.equal(checked.sourcePath, exactPath);
  assert.deepEqual(checked.inputProps, s.input);
  await assert.rejects(stat(s.options.workDir), {code: 'ENOENT'});

  await truncate(exactPath, 256 * 1024 * 1024 + 1);
  let loaded = false;
  await assert.rejects(renderFixture(s.options, async () => { loaded = true; }), /byte allowance/);
  assert.equal(loaded, false);
  await assert.rejects(stat(s.options.workDir), {code: 'ENOENT'});
});

test('fails malformed props before loading renderer or writing a work directory', async (t) => {
  const s = await setup(t);
  s.input.recipe.segments[0].sourceId = 'missing';
  await writeFile(s.options.inputFile, JSON.stringify(s.input));
  let loaded = false;
  await assert.rejects(renderFixture(s.options, async () => { loaded = true; }), /mapping/);
  assert.equal(loaded, false);
  await assert.rejects(stat(s.options.workDir), {code: 'ENOENT'});
});

test('fails raw oversized or invalid UTF8 JSON before renderer', async (t) => {
  const s = await setup(t);
  await writeFile(s.options.inputFile, ' '.repeat(65537));
  await assert.rejects(preflight(s.options), /64 KiB/);
  await writeFile(s.options.inputFile, Buffer.from([0xff]));
  await assert.rejects(preflight(s.options), /encoded data/);
});

test('rejects wrong file hash and symlink source redirection', async (t) => {
  const s = await setup(t);
  await writeFile(s.sourcePath, 'changed');
  await assert.rejects(preflight(s.options), /hash mismatch/);
  await rm(s.sourcePath);
  const outside = path.join(s.dir, 'elsewhere.mp4');
  await writeFile(outside, s.source);
  await symlink(outside, s.sourcePath);
  await assert.rejects(preflight(s.options), /symlink/);
});

test('requires explicit browser/binaries and fresh work/output paths', async (t) => {
  const s = await setup(t);
  await assert.rejects(preflight({...s.options, browserExecutable: undefined}), /explicit/);
  await assert.rejects(preflight({...s.options, browserExecutable: '/missing-browser'}), {code: 'ENOENT'});
  await assert.rejects(preflight({...s.options, outputPath: s.sourcePath}), /Output/);
  await assert.rejects(preflight({...s.options, outputPath: 'output.mp4'}), /absolute/);
  await mkdir(s.options.workDir);
  await assert.rejects(preflight(s.options), /never overwrite/);
});

test('inert renderer seam gets local bundle, explicit tools and fixed encoding; originals unchanged', async (t) => {
  const s = await setup(t);
  const calls = [];
  const result = await renderFixture(s.options, async () => ({
    bundle: async (options) => {
      calls.push(['bundle', options]);
      assert.equal(options.enableCaching, false);
      assert.equal(options.symlinkPublicDir, false);
      assert.deepEqual(await readFile(path.join(options.publicDir, s.input.sources[0].assetPath)), s.source);
      return options.outDir;
    },
    selectComposition: async (options) => {
      calls.push(['select', options]);
      assert.equal(options.browserExecutable, s.options.browserExecutable);
      assert.equal(options.binariesDirectory, s.options.binariesDirectory);
      assert.equal(options.chromeMode, 'chrome-for-testing');
      assert.throws(options.onBrowserDownload, /downloads are disabled/);
      return {...s.input.recipe, id: 'jarvis-clip-v1'};
    },
    renderMedia: async (options) => {
      calls.push(['render', options]);
      assert.equal(options.outputLocation, s.options.outputPath);
      assert.equal(options.codec, 'h264'); assert.equal(options.audioCodec, 'aac');
      assert.equal(options.pixelFormat, 'yuv420p'); assert.equal(options.muted, false);
      assert.equal(options.colorSpace, 'bt709'); assert.equal(options.imageFormat, 'png');
      assert.equal(options.chromeMode, 'chrome-for-testing');
      assert.equal(options.overwrite, false); assert.equal(options.concurrency, 1);
      assert.equal(options.browserExecutable, s.options.browserExecutable);
      assert.equal(options.binariesDirectory, s.options.binariesDirectory);
      assert.equal(Object.hasOwn(options, 'frameRange'), false); // Never apply half-open cuts to inclusive API.
      assert.throws(options.onBrowserDownload, /downloads are disabled/);
    },
  }));
  assert.deepEqual(calls.map(([name]) => name), ['bundle', 'select', 'render']);
  assert.deepEqual(result, {outputPath: s.options.outputPath});
  assert.deepEqual(await readFile(s.sourcePath), s.source);
  assert.equal(await readFile(s.options.inputFile, 'utf8'), JSON.stringify(s.input));
  await assert.rejects(stat(s.options.outputPath), {code: 'ENOENT'}); // Fake API creates no playable file.
  await assert.rejects(renderFixture(s.options), /never overwrite/); // No second attempt overwrites anything.
});

test('explicit mute reaches renderer and composition mismatch fails before render', async (t) => {
  const s = await setup(t);
  s.input.recipe.audio = 'mute'; s.input.sources[0].audioStreams = 0;
  await writeFile(s.options.inputFile, JSON.stringify(s.input));
  let rendered = false;
  const fake = (wrong) => async () => ({
    bundle: async (options) => options.outDir,
    selectComposition: async () => ({...s.input.recipe, fps: wrong ? 30 : 24}),
    renderMedia: async (options) => { rendered = true; assert.equal(options.muted, true); },
  });
  await renderFixture(s.options, fake(false));
  assert.equal(rendered, true);
  const next = {...s.options, workDir: path.join(s.dir, 'next'), outputPath: path.join(s.dir, 'next/output.mp4')};
  rendered = false;
  await assert.rejects(renderFixture(next, fake(true)), /metadata differs/);
  assert.equal(rendered, false);
});

test('browser download fallback fails closed', () => assert.throws(refuseBrowserDownload, /downloads are disabled/));
